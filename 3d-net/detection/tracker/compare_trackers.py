"""Compare SORT, ByteTrack, OC-SORT, DeepSort and BotSort on a FiftyOne image dataset.

Ground truth tracks (with globally unique ``index`` values) from ``gt_field``
are used both as detector input and as reference.  For each session the script
runs every tracker, computes HOTA / CLEAR / Identity metrics per session, then
reports aggregated results across all sessions.

Usage
-----
    python compare_trackers.py                            # whole dataset, ground_truth field
    python compare_trackers.py --split val                # val split only
    python compare_trackers.py --split train --gt-field ground_truth
"""

from __future__ import annotations

import argparse
import itertools
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import fiftyone as fo
import supervision as sv
from boxmot.trackers.botsort.botsort import BotSort as _BotSort
from deep_sort_realtime.deepsort_tracker import DeepSort as _DeepSort
from trackers.core.bytetrack.tracker import ByteTrackTracker
from trackers.core.ocsort.tracker import OCSORTTracker
from trackers.core.sort.tracker import SORTTracker
from trackers.eval import box_iou, compute_clear_metrics, compute_hota_metrics, compute_identity_metrics


DATASET_NAME = "3d_net_rgbt"
GT_FIELD     = "ground_truth"


# ──────────────────────────────────────────────────────────────────────────────
# Tracker wrappers — uniform interface: update(xyxy, conf, img) → (xyxy, ids)
# ──────────────────────────────────────────────────────────────────────────────

class _SupervisionTrackerWrapper:
    """Wraps SORT / ByteTrack / OC-SORT from the trackers library."""

    def __init__(self, tracker_cls, **kwargs):
        self._cls    = tracker_cls
        self._kwargs = kwargs
        self._tracker = tracker_cls(**kwargs)

    def reset(self):
        self._tracker = self._cls(**self._kwargs)

    def update(
        self,
        xyxy: np.ndarray,
        conf: np.ndarray,
        class_id: np.ndarray,
        img: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(xyxy) == 0:
            dets = sv.Detections.empty()
        else:
            dets = sv.Detections(xyxy=xyxy, confidence=conf, class_id=class_id)

        result = self._tracker.update(dets)
        if result.tracker_id is None or len(result.tracker_id) == 0:
            return np.empty((0, 4)), np.empty((0, 4)), np.empty(0, dtype=int)
        mask = result.tracker_id >= 0
        input_xyxy = result.xyxy[mask]
        ids = result.tracker_id[mask]

        # Build Kalman-state bounding boxes from internal track objects
        internal_tracks = getattr(self._tracker, "tracks", None) or getattr(self._tracker, "trackers", [])
        id_to_state = {t.tracker_id: t.get_state_bbox() for t in internal_tracks if hasattr(t, "get_state_bbox")}
        pred_rows = [id_to_state[tid] if tid in id_to_state else input_xyxy[i] for i, tid in enumerate(ids)]
        pred_xyxy = np.array(pred_rows, dtype=float).reshape(-1, 4) if pred_rows else np.empty((0, 4))

        return input_xyxy, pred_xyxy, ids


class _BotSortWrapper:
    """Wraps boxmot's BotSort with the uniform tracker interface.

    Set ``with_reid=True`` to enable OSNet-based appearance features
    (``osnet_x0_25_msmt17.pt`` is auto-downloaded on first use).
    """

    def __init__(self, track_buffer: int = 30, match_thresh: float = 0.8, with_reid: bool = False):
        self._track_buffer  = track_buffer
        self._match_thresh  = match_thresh
        self._with_reid     = with_reid
        self._tracker       = self._make()

    def _make(self) -> _BotSort:
        return _BotSort(
            reid_weights=None,
            device=torch.device("cpu"),
            half=False,
            track_buffer=self._track_buffer,
            match_thresh=self._match_thresh,
            with_reid=self._with_reid,
            frame_rate=6,
        )

    def reset(self):
        self._tracker = self._make()

    def update(
        self,
        xyxy: np.ndarray,
        conf: np.ndarray,
        class_id: np.ndarray,
        img: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(xyxy) == 0:
            dets = np.empty((0, 6), dtype=np.float32)
        else:
            dets = np.column_stack([xyxy, conf, class_id]).astype(np.float32)

        out = self._tracker.update(dets, img)  # (M, 8): x1 y1 x2 y2 id conf cls det_ind
        if out is None or len(out) == 0:
            return np.empty((0, 4)), np.empty((0, 4)), np.empty(0, dtype=int)
        det_idx = out[:, 7].astype(int)
        return xyxy[det_idx], out[:, :4], out[:, 4].astype(int)


class _DeepSortWrapper:
    def __init__(self, max_age: int = 5, n_init: int = 1):
        self._max_age = max_age
        self._n_init  = n_init
        self._tracker = _DeepSort(max_age=max_age, n_init=n_init)

    def reset(self):
        self._tracker = _DeepSort(max_age=self._max_age, n_init=self._n_init)

    def update(
        self,
        xyxy: np.ndarray,
        conf: np.ndarray,
        class_id: np.ndarray,
        img: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(xyxy) == 0:
            self._tracker.update_tracks([], frame=img)
            return np.empty((0, 4)), np.empty((0, 4)), np.empty(0, dtype=int)

        x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
        bbs = [([x1[i], y1[i], x2[i] - x1[i], y2[i] - y1[i]], float(conf[i]), int(class_id[i]))
               for i in range(len(xyxy))]
        tracks = self._tracker.update_tracks(bbs, frame=img)

        out_input_xyxy, out_pred_xyxy, out_ids = [], [], []
        for t in tracks:
            if not t.is_confirmed() or t.original_ltwh is None:
                continue
            lx, ly, lw, lh = t.original_ltwh
            out_input_xyxy.append([lx, ly, lx + lw, ly + lh])
            out_pred_xyxy.append(t.to_ltrb())
            out_ids.append(t.track_id)

        if not out_ids:
            return np.empty((0, 4)), np.empty((0, 4)), np.empty(0, dtype=int)
        return np.array(out_input_xyxy), np.array(out_pred_xyxy), np.array(out_ids, dtype=int)


def make_trackers(max_age: int) -> dict[str, _SupervisionTrackerWrapper | _DeepSortWrapper | _BotSortWrapper]:
    return {
        "SORT":      _SupervisionTrackerWrapper(SORTTracker,      lost_track_buffer=max_age, minimum_consecutive_frames=2, frame_rate=6),
        "ByteTrack": _SupervisionTrackerWrapper(ByteTrackTracker, lost_track_buffer=max_age, minimum_consecutive_frames=2, frame_rate=6),
        "OC-SORT":   _SupervisionTrackerWrapper(OCSORTTracker,    lost_track_buffer=max_age, minimum_consecutive_frames=2, frame_rate=6),
        "DeepSort":     _DeepSortWrapper(max_age=max_age, n_init=2),
        "BotSort":      _BotSortWrapper(track_buffer=max_age, with_reid=False),
        "BotSort+ReID": _BotSortWrapper(track_buffer=max_age, with_reid=True),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Session key / ordering helpers (mirrors ds_track.py)
# ──────────────────────────────────────────────────────────────────────────────

def _session_key(s: fo.Sample) -> tuple:
    return (s.get_field("location") or "", s.get_field("date") or "", s.get_field("session") or "")


def _frame_index(s: fo.Sample) -> int:
    return s.get_field("frame_index") or 0


# ──────────────────────────────────────────────────────────────────────────────
# Per-session evaluation
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SessionResult:
    gt_ids:     list[np.ndarray] = field(default_factory=list)
    tr_ids:     list[np.ndarray] = field(default_factory=list)
    sims:       list[np.ndarray] = field(default_factory=list)       # GT vs input detection boxes
    sims_pred:  list[np.ndarray] = field(default_factory=list)       # GT vs tracker-predicted boxes
    latencies:  list[float]      = field(default_factory=list)  # ms per frame
    det_mask:   list[bool]       = field(default_factory=list)  # True if frame had detections


def eval_session(
    samples: list[fo.Sample],
    gt_field: str,
    trackers: dict,
) -> dict[str, SessionResult]:
    results = {name: SessionResult() for name in trackers}

    for sample in sorted(samples, key=_frame_index):
        img = cv2.imread(sample.local_path)
        if img is None:
            logging.warning("Could not read %s", sample.local_path)
            continue

        img_h, img_w = img.shape[:2]
        dets_field: fo.Detections | None = sample.get_field(gt_field)

        gt_boxes = np.empty((0, 4))
        gt_ids   = np.empty(0, dtype=int)
        det_xyxy = np.empty((0, 4))
        det_conf = np.empty(0)
        det_cls  = np.empty(0, dtype=int)

        if dets_field and dets_field.detections:
            rows = []
            ids  = []
            for det in dets_field.detections:
                if det.index is None:
                    continue
                x, y, w, h = det.bounding_box
                x1, y1 = x * img_w, y * img_h
                x2, y2 = (x + w) * img_w, (y + h) * img_h
                rows.append([x1, y1, x2, y2])
                ids.append(det.index)

            if rows:
                gt_boxes = np.array(rows)
                gt_ids   = np.array(ids, dtype=int)
                det_xyxy = gt_boxes.copy()
                det_conf = np.ones(len(rows))
                det_cls  = np.zeros(len(rows), dtype=int)

        for name, tracker in trackers.items():
            t0 = time.perf_counter()
            tr_input_xyxy, tr_pred_xyxy, tr_ids = tracker.update(det_xyxy, det_conf, det_cls, img)
            elapsed_ms = (time.perf_counter() - t0) * 1e3

            if len(gt_ids) > 0 and len(tr_ids) > 0:
                sim      = box_iou(gt_boxes, tr_input_xyxy)
                sim_pred = box_iou(gt_boxes, tr_pred_xyxy)
            elif len(gt_ids) == 0 and len(tr_ids) == 0:
                sim = sim_pred = np.empty((0, 0))
            else:
                sim = sim_pred = np.zeros((len(gt_ids), len(tr_ids)))

            results[name].gt_ids.append(gt_ids)
            results[name].tr_ids.append(tr_ids)
            results[name].sims.append(sim)
            results[name].sims_pred.append(sim_pred)
            results[name].latencies.append(elapsed_ms)
            results[name].det_mask.append(len(gt_ids) > 0)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────────────

def aggregate(all_results: list[dict[str, SessionResult]]) -> tuple[dict[str, dict], dict[str, SessionResult]]:
    combined: dict[str, SessionResult] = defaultdict(SessionResult)
    for session_results in all_results:
        for name, res in session_results.items():
            combined[name].gt_ids.extend(res.gt_ids)
            combined[name].tr_ids.extend(res.tr_ids)
            combined[name].sims.extend(res.sims)
            combined[name].sims_pred.extend(res.sims_pred)
            combined[name].latencies.extend(res.latencies)
            combined[name].det_mask.extend(res.det_mask)

    metrics: dict[str, dict] = {}
    for name, res in combined.items():
        hota     = compute_hota_metrics(res.gt_ids, res.tr_ids, res.sims)
        clear    = compute_clear_metrics(res.gt_ids, res.tr_ids, res.sims)
        identity = compute_identity_metrics(res.gt_ids, res.tr_ids, res.sims)

        hota_pred     = compute_hota_metrics(res.gt_ids, res.tr_ids, res.sims_pred)
        clear_pred    = compute_clear_metrics(res.gt_ids, res.tr_ids, res.sims_pred)
        identity_pred = compute_identity_metrics(res.gt_ids, res.tr_ids, res.sims_pred)
        pred_metrics  = {k + "_pred": v for d in (hota_pred, clear_pred, identity_pred) for k, v in d.items()}

        lat = np.array(res.latencies)
        lat_det = lat[np.array(res.det_mask)] if res.det_mask else np.empty(0)

        def _lat_stats(a: np.ndarray, suffix: str) -> dict:
            if len(a) == 0:
                return {f"Lat_mean{suffix}": float("nan"), f"Lat_p50{suffix}": float("nan"), f"Lat_p95{suffix}": float("nan")}
            return {
                f"Lat_mean{suffix}": float(np.mean(a)),
                f"Lat_p50{suffix}":  float(np.percentile(a, 50)),
                f"Lat_p95{suffix}":  float(np.percentile(a, 95)),
            }

        latency_metrics = {**_lat_stats(lat, ""), **_lat_stats(lat_det, "_det")}
        metrics[name] = {**hota, **clear, **identity, **latency_metrics, **pred_metrics}

    return metrics, dict(combined)


def print_results(metrics: dict[str, dict]) -> None:
    perf_keys    = ["HOTA", "DetA", "AssA", "MOTA", "MOTP", "IDF1", "IDP", "IDR"]
    lat_keys_all = ["Lat_mean",     "Lat_p50",     "Lat_p95"]
    lat_keys_det = ["Lat_mean_det", "Lat_p50_det", "Lat_p95_det"]
    available    = [k for k in perf_keys if any(k in m for m in metrics.values())]

    col_w = max(len(n) for n in metrics) + 2

    def _print_table(title: str, key_suffix: str, show_latency: bool) -> None:
        header = (
            f"{'Tracker':<{col_w}}"
            + "".join(f"  {k:>8}" for k in available)
        )
        if show_latency:
            header += (
                "  │"
                + "".join(f"  {k:>10}" for k in lat_keys_all)
                + "  │"
                + "".join(f"  {k:>12}" for k in lat_keys_det)
            )
        print(f"\n{title}")
        print(header)
        print("-" * len(header))
        for name, m in metrics.items():
            row = f"{name:<{col_w}}"
            for k in available:
                row += f"  {m.get(k + key_suffix, float('nan')):>8.4f}"
            if show_latency:
                row += "  │"
                for k in lat_keys_all:
                    row += f"  {m.get(k, float('nan')):>8.2f}ms"
                row += "  │"
                for k in lat_keys_det:
                    row += f"  {m.get(k, float('nan')):>10.2f}ms"
            print(row)
        print()

    _print_table("Input detection boxes (passed to trackers)", "",      show_latency=True)
    _print_table("Tracker-predicted bounding boxes",           "_pred", show_latency=False)


# ──────────────────────────────────────────────────────────────────────────────
# Latency plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_latencies(combined: dict[str, SessionResult], out_path: str = "latency_distributions.png") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    titles = ["All frames", "Frames with detections only"]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, title, det_only in zip(axes, titles, [False, True]):
        all_lats = []
        for name, res in combined.items():
            lat = np.array(res.latencies)
            if det_only:
                lat = lat[np.array(res.det_mask)]
            if len(lat) > 0:
                all_lats.append(lat)
        if not all_lats:
            continue
        global_min = max(min(l.min() for l in all_lats), 1e-3)
        global_max = max(l.max() for l in all_lats)
        bins = np.logspace(np.log10(global_min), np.log10(global_max), 50)

        for (name, res), color in zip(combined.items(), colors):
            lat = np.array(res.latencies)
            if det_only:
                lat = lat[np.array(res.det_mask)]
            if len(lat) == 0:
                continue
            ax.hist(lat, bins=bins, alpha=0.5, label=name, color=color, weights=np.ones(len(lat)) / len(lat))

        ax.set_xscale("log")
        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Fraction of frames")
        ax.set_title(title)
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved latency plot to {out_path}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare SORT / ByteTrack / OC-SORT / DeepSort / BotSort on a FiftyOne image dataset.",
    )
    parser.add_argument("--name",     default=DATASET_NAME)
    parser.add_argument("--gt-field", default=GT_FIELD,
                        help="Sample field with ground-truth Detections (default: %(default)s)")
    parser.add_argument("--split",    default="",
                        help="Split to evaluate on; omit or leave empty to use all data (default: all)")
    parser.add_argument("--max-age",  type=int, default=30,
                        help="Max missed frames before a track is dropped (default: %(default)s)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not fo.dataset_exists(args.name):
        raise SystemExit(f"Dataset '{args.name}' does not exist.")

    dataset = fo.load_dataset(args.name)
    collection: fo.Dataset | fo.DatasetView = dataset
    if args.split:
        collection = dataset.match(fo.ViewField("split") == args.split)
        logging.info("Filtered to split='%s': %d samples.", args.split, len(collection))

    samples = list(collection.iter_samples(progress=True))
    samples.sort(key=lambda s: (*_session_key(s), _frame_index(s)))

    trackers = make_trackers(args.max_age)
    all_results: list[dict[str, SessionResult]] = []

    for key, group in itertools.groupby(samples, key=_session_key):
        session_samples = list(group)
        logging.info("Session %s — %d frames", key, len(session_samples))

        session_results = eval_session(session_samples, args.gt_field, trackers)
        all_results.append(session_results)

        for tracker in trackers.values():
            tracker.reset()

    logging.info("Computing metrics...")
    metrics, combined = aggregate(all_results)
    print_results(metrics)
    plot_latencies(combined)


if __name__ == "__main__":
    main()
