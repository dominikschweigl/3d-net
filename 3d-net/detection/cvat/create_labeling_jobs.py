
import fiftyone as fo
import fiftyone.core.storage as fos
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

fos.init_storage()

DATASET_NAME      = "3d_net_rgbt"
CVAT_PROJECT_NAME = "3d_net"

CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle", "person"]
DRY_RUN = False  # Set to True to skip annotate() calls and just print task names

LOCATION = "hausern"
DATE     = "05_05_26"

# Frame index ranges (inclusive) per session that define individual CVAT tasks
SESSIONS: dict[str, list[tuple[int, int]]] = {
    "aufnahme1": [
        (5763, 6360),
        (6363, 7113),
        (7116, 7710),
        (7713, 8415),
        (8418, 9468),
        (9471, 10095),
        (10098, 10422),
        (10425, 10989),
        (10992, 11574),
        (11577, 12189),
        (12192, 12870),
        (12873, 13449),
        (13452, 14049),
        (14052, 14649),
        (14652, 15339),
        (15342, 15939),
        (15942, 16539),
        (16542, 17085),
        (17088, 17685),
        (17688, 17862),
    ],
    "aufnahme2": [
        (648, 1272),
        (1275, 1869),
        (1872, 2436),
        (2439, 3039),
        (3042, 3702),
        (3705, 4272),
        (4275, 4878),
        (4881, 5592),
        (5595, 6240),
        (6243, 6840),
        (6843, 7029),
        (7437, 8328),
        (8331, 8949),
        (8952, 9684),
    ],
}

dataset = fo.load_dataset(DATASET_NAME)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for session, ranges in SESSIONS.items():
    session_view = dataset.view().match(
        (fo.ViewField("location") == LOCATION)
        & (fo.ViewField("date") == DATE)
        & (fo.ViewField("session") == session)
    )

    for start, end in ranges:
        range_view = session_view.match(
            (fo.ViewField("frame_index") >= start)
            & (fo.ViewField("frame_index") <= end)
        )

        task_name = f"{LOCATION}_{DATE}_{session}_{start}_{end}_{timestamp}"

        if DRY_RUN:
            print(f"[DRY RUN] Would create task: {task_name}  frames={start}-{end}  ({len(range_view)} samples)")
            continue

        range_view.annotate(
            task_name,
            label_field="ground_truth",
            classes=CLASSES,
            launch_editor=False,
            project_name=CVAT_PROJECT_NAME,
            organization="TEC",
            label_type="detections",
            task_name=task_name,
            media_field="filepath",
            use_cache=False,
        )
        print(f"Created task: {task_name}")
