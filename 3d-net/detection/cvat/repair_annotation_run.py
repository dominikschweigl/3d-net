"""
Repair a FiftyOne annotation run whose results object is None (lost/never saved).

Reconstructs the CVATAnnotationResults by querying CVAT for the task matching
the annotation run key, then re-saves it so load_annotations works normally.

Usage:
    python repair_annotation_run.py --key hausern_05_05_26_aufnahme2_8331_8949_20260505_182756
"""
import argparse
import os
from collections import OrderedDict

import fiftyone as fo
import fiftyone.utils.cvat as fouc

DATASET_NAME = "3d_net_rgbt"
LABEL_FIELD   = "ground_truth"


def find_cvat_task_by_name(api: fouc.CVATAnnotationAPI, task_name: str) -> int | None:
    resp = api.get(api.tasks_url, params={"search": task_name, "page_size": 10})
    results = resp.json().get("results", [])
    for t in results:
        if t["name"] == task_name:
            return t["id"]
    return None


def get_job_ids(api: fouc.CVATAnnotationAPI, task_id: int) -> list[int]:
    resp = api.get(api.base_api_url + "/jobs", params={"task_id": task_id})
    return [j["id"] for j in resp.json().get("results", [])]


def get_cvat_frames(api: fouc.CVATAnnotationAPI, task_id: int) -> list[str]:
    resp = api.get(api.task_data_meta_url(task_id))
    return [f["name"] for f in resp.json().get("frames", [])]


def build_frame_id_map(
    cvat_frames: list[str],
    samples_by_filename: dict[str, fo.Sample],
) -> dict[int, dict]:
    # Keys must be int — CVATAnnotationResults stores them as int in memory
    frame_id_map = {}
    for frame_idx, frame_name in enumerate(cvat_frames):
        basename = os.path.basename(frame_name)
        sample = samples_by_filename.get(basename)
        if sample is None:
            raise ValueError(f"No FiftyOne sample for CVAT frame: {frame_name}")
        frame_id_map[frame_idx] = OrderedDict(sample_id=sample.id)
    return frame_id_map


def repair(dataset: fo.Dataset, broken_key: str, dry_run: bool = False) -> None:
    results = dataset.load_annotation_results(broken_key)
    if results is not None:
        print(f"Run '{broken_key}' already has results — nothing to repair.")
        return

    info = dataset.get_annotation_info(broken_key)
    config_dict = info.config.serialize()

    # Re-use a working run's API connection (credentials are in FiftyOne's keyring)
    working_keys = [k for k in dataset.list_annotation_runs() if k != broken_key]
    if not working_keys:
        raise RuntimeError("No working annotation run to borrow CVAT credentials from.")
    working_results = None
    for k in working_keys:
        working_results = dataset.load_annotation_results(k)
        if working_results is not None:
            break
    if working_results is None:
        raise RuntimeError("All annotation runs have missing results.")

    api = working_results.connect_to_api()

    task_name = config_dict.get("task_name", broken_key)
    print(f"Searching CVAT for task: {task_name}")
    task_id = find_cvat_task_by_name(api, task_name)
    if task_id is None:
        raise RuntimeError(f"CVAT task not found: {task_name}")
    print(f"Found CVAT task ID: {task_id}")

    job_ids = get_job_ids(api, task_id)
    print(f"Job IDs: {job_ids}")

    cvat_frames = get_cvat_frames(api, task_id)
    print(f"CVAT frames: {len(cvat_frames)}")

    # Find the FiftyOne samples that belong to this run
    # Match by filename since CVAT frame names == FiftyOne sample basenames
    sample_ids_in_run = list(
        dataset.get_annotation_info(broken_key)
        .config.serialize()
        .get("label_schema", {})
        .keys()
    )
    # Fall back: query via frame filenames
    basenames = {os.path.basename(f) for f in cvat_frames}
    view = dataset.match(fo.ViewField("filepath").re_match(
        "(" + "|".join(re.escape(b) for b in basenames) + ")$"
    ))

    # Simpler: filter by the same location/date/session/frame range
    # derived from the task name pattern: <loc>_<date>_<session>_<start>_<end>_<ts>
    parts = task_name.split("_")
    # hausern_05_05_26_aufnahme2_8331_8949_20260505_182756
    # loc=hausern, date=05_05_26, session=aufnahme2, start=8331, end=8949
    loc     = parts[0]
    date    = "_".join(parts[1:4])        # 05_05_26
    session = parts[4]                    # aufnahme2
    start   = int(parts[5])
    end     = int(parts[6])

    view = dataset.view().match(
        (fo.ViewField("location") == loc)
        & (fo.ViewField("date")     == date)
        & (fo.ViewField("session")  == session)
        & (fo.ViewField("frame_index") >= start)
        & (fo.ViewField("frame_index") <= end)
    )
    print(f"FiftyOne samples in range: {len(view)}")
    if len(view) != len(cvat_frames):
        raise RuntimeError(
            f"Sample count mismatch: FiftyOne={len(view)} vs CVAT={len(cvat_frames)}"
        )

    samples_by_filename = {
        os.path.basename(s.filepath): s for s in view.iter_samples(progress=False)
    }

    frame_id_map = build_frame_id_map(cvat_frames, samples_by_filename)
    id_map = {LABEL_FIELD: {s.id: [] for s in view.iter_samples(progress=False)}}
    server_id_map = {}
    labels_task_map = OrderedDict({LABEL_FIELD: [task_id]})
    # int keys required — CVATAnnotationResults stores task_id/frame_id as int
    job_ids_map = {task_id: job_ids}

    if dry_run:
        print("[DRY RUN] Would reconstruct results and call load_annotations.")
        print(f"  task_id={task_id}, jobs={job_ids}, frames={len(frame_id_map)}")
        return

    backend_config = fouc.CVATBackendConfig.from_dict(config_dict)

    reconstructed = fouc.CVATAnnotationResults(
        samples=view,
        config=backend_config,
        anno_key=broken_key,
        id_map=id_map,
        server_id_map=server_id_map,
        project_ids=[],
        task_ids=[task_id],
        job_ids=job_ids_map,
        frame_id_map={task_id: frame_id_map},
        labels_task_map=labels_task_map,
    )
    reconstructed.save()
    print("Saved reconstructed results.")

    dataset.load_annotations(broken_key, unexpected="prompt")
    dataset.save()
    print(f"Done — annotations loaded for: {broken_key}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair a broken FiftyOne annotation run.")
    parser.add_argument("--name",    default=DATASET_NAME, help="FiftyOne dataset name")
    parser.add_argument("--key",     required=True,        help="Annotation run key to repair")
    parser.add_argument("--dry-run", action="store_true",  help="Show what would happen without doing it")
    args = parser.parse_args()

    dataset = fo.load_dataset(args.name)
    repair(dataset, args.key, dry_run=args.dry_run)


if __name__ == "__main__":
    import re
    main()
