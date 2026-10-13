import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .config import Config
from .evaluator import Evaluator
from .frame_provider import FrameProvider
from .models import BBox, Detection, FrameResult
from .perf_monitor import PerfMonitor
from .publisher import DetectionPublisher
from .tracker import Tracker
from .inference_recorder import InferenceRecorder

logger = logging.getLogger(__name__)

if sys.stdout.isatty():
    GREEN, YELLOW, RED, BOLD, RESET = "\033[32m", "\033[33m", "\033[31m", "\033[1m", "\033[0m"
else:
    GREEN = YELLOW = RED = BOLD = RESET = ""

_MODEL_INPUT_SIZE = 640

# TODO: avoid this warning:
# 2026-03-26 11:34:08,668 - WARNING - pyhailort.py:1031 - Given input data dtype (float32) is different than inferred dtype (uint8). conversion for every frame will reduce performance


class InferenceService:
    def __init__(self, config: Config, frame_provider: FrameProvider):
        self._cfg = config
        self._provider = frame_provider
        self._publisher: Optional[DetectionPublisher] = None
        self._tracker: Optional[Tracker] = None
        self._monitor: Optional[PerfMonitor] = None
        self._recorder: Optional[InferenceRecorder] = None
        self._session: str = ""

    def run(self) -> None:
        cfg = self._cfg
        self._session = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._provider.configure_pipeline(n_workers=cfg.service.pipeline_workers, modality=cfg.model.modality)
        self._provider.start()
        self._publisher = DetectionPublisher(cfg.zmq.endpoint)
        tracker_fps = cfg.service.max_fps or cfg.camera.fps
        self._tracker = Tracker(cfg.tracker, tracker_fps, cfg.model.class_names)
        self._monitor = PerfMonitor(
            window_frames=cfg.monitoring.window_frames,
            report_interval_s=cfg.monitoring.report_interval_s,
            log_path=Path(cfg.monitoring.log_dir) if cfg.monitoring.log_dir else None,
            hw_poll_every_frames=cfg.monitoring.hw_poll_every_frames,
            session=self._session,
        )

        print(f"{BOLD}================================")
        print("Inference Service")
        print(f"================================{RESET}")
        print(self._provider.describe())
        print(f"Model   : {cfg.model.hef_path}")
        print(f"  conf    : {cfg.model.conf_threshold}")
        print(f"  classes : {cfg.model.class_names}")
        print(f"Tracker : {cfg.tracker.mode.value}")
        print(f"ZMQ     : {cfg.zmq.endpoint}")
        if cfg.service.max_fps is None:
            print("Max FPS : uncapped")
        else:
            print(f"Max FPS : {cfg.service.max_fps}")
        if cfg.evaluation.enabled:
            eval_out = Path(cfg.monitoring.log_dir) / "<session>" / "evaluation" if cfg.monitoring.log_dir else "logs/<session>/evaluation"
            print(f"{YELLOW}Eval    : enabled  IoU={cfg.evaluation.iou_threshold}  out={eval_out}{RESET}")
        print()

        try:
            self._infer_loop()
        except (KeyboardInterrupt, SystemExit):
            print("\nShutting down...")
        finally:
            self.stop()

        if cfg.evaluation.enabled:
            self._run_evaluation()

    def _infer_loop(self) -> None:
        # Lazy import: hailo_platform is only available on the RPi with the Hailo kit.
        from hailo_platform import (  # type: ignore[import]
            ConfigureParams,
            FormatType,
            HailoStreamInterface,
            HEF,
            InferVStreams,
            InputVStreamParams,
            OutputVStreamParams,
            VDevice,
        )

        cfg = self._cfg
        hef = HEF(cfg.model.hef_path)
        interval: Optional[float] = None
        if cfg.service.max_fps is not None and cfg.service.max_fps > 0:
            interval = 1.0 / cfg.service.max_fps
        crop_w, crop_h = self._provider.crop_size

        with VDevice() as target:
            configure_params = ConfigureParams.create_from_hef(
                hef, interface=HailoStreamInterface.PCIe
            )
            network_group = target.configure(hef, configure_params)[0]
            network_group_params = network_group.create_params()

            input_info = hef.get_input_vstream_infos()[0]
            logger.info("Hailo input : %s  shape=%s", input_info.name, input_info.shape)
            for o in hef.get_output_vstream_infos():
                logger.info("Hailo output: %s  shape=%s", o.name, o.shape)

            physical_devices = target.get_physical_devices()
            if physical_devices and self._monitor:
                self._monitor.set_hailo_device(physical_devices[0])

            input_params = InputVStreamParams.make_from_network_group(
                network_group, quantized=True, format_type=FormatType.UINT8
            )
            output_params = OutputVStreamParams.make_from_network_group(
                network_group, quantized=False, format_type=FormatType.FLOAT32
            )

            with network_group.activate(network_group_params):
                with InferVStreams(network_group, input_params, output_params) as pipeline:
                    session = self._session
                    frame_idx = 0
                    t_next = time.monotonic()

                    if cfg.inference_recorder.enabled:
                        recorder_fps = (
                            cfg.inference_recorder.video_fps
                            or cfg.service.max_fps
                            or cfg.camera.fps
                        )
                        self._recorder = InferenceRecorder(
                            cfg.inference_recorder, session, crop_w, crop_h,
                            recorder_fps,
                        )

                    while True:
                        if interval is not None:
                            now = time.monotonic()
                            if now < t_next:
                                time.sleep(t_next - now)
                            t_next = time.monotonic() + interval

                        t0 = time.monotonic()
                        item = self._provider.next_prepared()
                        t_frame = time.monotonic() - t0

                        if item is None:
                            if self._provider.exhausted:
                                break
                            print(
                                f"\r{YELLOW}[WAIT]{RESET} no frame pair available    ",
                                end="", flush=True,
                            )
                            continue

                        frame_idx += 1
                        frame_id = f"{session}_{frame_idx:07d}"
                        input_data = {input_info.name: item.tensor}

                        # ── Hailo inference ───────────────────────────────────
                        t1 = time.monotonic()
                        results = pipeline.infer(input_data)
                        output_key = list(results.keys())[0]
                        batch_result = results[output_key][0]
                        t_infer = time.monotonic() - t1

                        # ── Post-process ──────────────────────────────────────
                        t2 = time.monotonic()
                        detections = self._postprocess(
                            batch_result,
                            cfg.model.class_names,
                            cfg.model.conf_threshold,
                            crop_w,
                            crop_h,
                            item.lb_scale,
                            item.lb_pad_x,
                            item.lb_pad_y,
                        )
                        t_post = time.monotonic() - t2

                        # ── Track ─────────────────────────────────────────────
                        t3 = time.monotonic()
                        if self._tracker.enabled:
                            detections = self._tracker.update(detections)
                        t_track = time.monotonic() - t3

                        if self._recorder:
                            self._recorder.write(item.pair, detections)

                        self._publisher.publish(FrameResult(
                            timestamp=datetime.now(timezone.utc),
                            frame_id=frame_id,
                            detections=detections,
                            source_file=item.pair.frame_stem,
                        ))

                        # ── Status line ───────────────────────────────────────
                        if self._cfg.monitoring.log_status_line:
                            n = len(detections)
                            det_str = f"{GREEN}{n} det{RESET}" if n else f"{RED}0 det{RESET}"
                            print(
                                f"[INF] {frame_id}"
                                f"  drift={item.pair.drift_s*1000:4.1f}ms"
                                f"  frame_provider={t_frame*1000:4.1f}ms"
                                f"  align={item.t_align*1000:4.1f}ms"
                                f"  pre={item.t_pre*1000:4.1f}ms"
                                f"  infer={t_infer*1000:5.1f}ms"
                                f"  post={t_post*1000:4.1f}ms"
                                f"  track={t_track*1000:4.1f}ms"
                                f"  {det_str}",
                            )
                            for det in detections:
                                b = det.bbox
                                tid_str = f" [T{det.track_id}]" if det.track_id is not None else ""
                                print(
                                    f"  {det.name:<12} {det.confidence:.2f}"
                                    f"  [{b.x1},{b.y1} → {b.x2},{b.y2}]{tid_str}"
                                )

                        if self._monitor:
                            self._monitor.record_frame(
                                frame_id=frame_id,
                                drift_s=item.pair.drift_s,
                                frame_provider_s=t_frame,
                                align_s=item.t_align,
                                pre_s=item.t_pre,
                                infer_s=t_infer,
                                post_s=t_post,
                                track_s=t_track,
                                detections=detections,
                                frame_stem=item.pair.frame_stem,
                            )
                            self._monitor.maybe_report()

    def _postprocess(
        self,
        batch_result,
        class_names: List[str],
        conf_threshold: float,
        crop_w: int,
        crop_h: int,
        lb_scale: float,
        lb_pad_x: int,
        lb_pad_y: int,
    ) -> List[Detection]:
        """
        Convert Hailo NMS output to Detection objects.

        batch_result is a list indexed by class; each entry is an ndarray of shape (N, 5)
        where each row is [y1_norm, x1_norm, y2_norm, x2_norm, score] (YXYX, normalised).
        Coordinates are de-projected from the letterboxed 640×640 space back to crop dimensions.
        """
        detections: List[Detection] = []
        for cls_idx, cls_dets in enumerate(batch_result):
            if len(cls_dets) == 0:
                continue
            name = class_names[cls_idx] if cls_idx < len(class_names) else f"class_{cls_idx}"
            for det in cls_dets:
                y1_n, x1_n, y2_n, x2_n, score = det
                if score < conf_threshold:
                    continue
                x1 = max(0, min(crop_w, int((x1_n * _MODEL_INPUT_SIZE - lb_pad_x) / lb_scale)))
                y1 = max(0, min(crop_h, int((y1_n * _MODEL_INPUT_SIZE - lb_pad_y) / lb_scale)))
                x2 = max(0, min(crop_w, int((x2_n * _MODEL_INPUT_SIZE - lb_pad_x) / lb_scale)))
                y2 = max(0, min(crop_h, int((y2_n * _MODEL_INPUT_SIZE - lb_pad_y) / lb_scale)))
                detections.append(Detection(
                    name=name,
                    label=cls_idx,
                    bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    confidence=float(score),
                ))
        return detections

    def stop(self) -> None:
        self._provider.stop()
        if self._publisher:
            self._publisher.close()
        if self._monitor:
            self._monitor.close()
        if self._recorder:
            self._recorder.close()

    def _run_evaluation(self) -> None:
        cfg = self._cfg
        if not self._monitor or not self._monitor.stats_csv_path:
            logger.warning("Evaluation requested but monitoring log_dir is not configured — skipping")
            return

        evaluator = Evaluator(
            class_names=cfg.model.class_names,
            iou_threshold=cfg.evaluation.iou_threshold,
            iou_range=cfg.evaluation.iou_range,
            output_dir=Path(cfg.monitoring.log_dir) / self._session / "evaluation",
            pipeline_workers=cfg.service.pipeline_workers,
        )
        evaluator.run_from_logs(
            stats_csv=self._monitor.stats_csv_path,
            detections_csv=self._monitor.detections_csv_path,
            labels_dir=self._provider.labels_dir,
        )
        summary = evaluator.summarize()
        self._print_eval_summary(summary)

    def _print_eval_summary(self, summary: dict) -> None:
        print(f"\n{BOLD}================================")
        print("Evaluation Summary")
        print(f"================================{RESET}")
        threshold_key = next((k for k in summary if k.startswith("mAP_")), None)
        if threshold_key:
            print(f"  {threshold_key:<12}: {summary[threshold_key]:.4f}")
        if "mAP_50_95" in summary:
            print(f"  {'mAP_50_95':<12}: {summary['mAP_50_95']:.4f}")
        print(f"  {'frames':<12}: {summary['frames_evaluated']}  (skipped: {summary['frames_skipped']})")
        print()
        print(f"  {'class':<14} {'AP@50':>7}  {'prec':>7}  {'recall':>7}  {'n_gt':>6}")
        print(f"  {'-'*14} {'-'*7}  {'-'*7}  {'-'*7}  {'-'*6}")
        for class_name, metrics in summary.get("per_class", {}).items():
            ap_key = next((k for k in metrics if k.startswith("AP_") and k != "AP_50_95"), "AP_50")
            print(
                f"  {class_name:<14} {metrics.get(ap_key, 0):>7.4f}"
                f"  {metrics['precision']:>7.4f}"
                f"  {metrics['recall']:>7.4f}"
                f"  {metrics['n_gt']:>6}"
            )
        timing = summary.get("timing", {})
        if timing:
            pipeline_workers = summary.get("pipeline_workers", 1)
            critical_stages = ["frame_provider_ms", "infer_ms", "post_ms", "track_ms", "total_ms"]
            parallel_stages = ["align_ms", "pre_ms"]

            print()
            print(f"  Timing  (Frame Provider Workers={pipeline_workers})")
            print(f"  {'stage':<18} {'mean ms':>8}  {'p95 ms':>8}  {'p99 ms':>8}")
            print(f"  {'-'*18} {'-'*8}  {'-'*8}  {'-'*8}")
            for stage in critical_stages:
                if stage not in timing:
                    continue
                stats = timing[stage]
                print(
                    f"  {stage:<18} {stats['mean']:>8.1f}"
                    f"  {stats['p95']:>8.1f}"
                    f"  {stats['p99']:>8.1f}"
                )

            if "fps" in timing:
                fps_stats = timing["fps"]
                print()
                print(f"  {'fps':<18} {fps_stats['mean']:>8.2f}"
                      f"  {fps_stats['p95']:>8.2f}"
                      f"  {fps_stats['p99']:>8.2f}  (mean / p95 / p99)")

            diag_rows = [(s, timing[s]) for s in parallel_stages if s in timing]
            if diag_rows:
                print()
                print(f"  Frame Provider Worker diagnostics  (not included in total_ms)")
                print(f"  {'stage':<18} {'mean ms':>8}  {'p95 ms':>8}  {'p99 ms':>8}")
                print(f"  {'-'*18} {'-'*8}  {'-'*8}  {'-'*8}")
                for stage, stats in diag_rows:
                    print(
                        f"  {stage:<18} {stats['mean']:>8.1f}"
                        f"  {stats['p95']:>8.1f}"
                        f"  {stats['p99']:>8.1f}"
                    )


