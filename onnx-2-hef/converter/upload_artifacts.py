"""
script to upload artifacts to a ClearML task
"""

import sys
from clearml import Task

SOURCE_TASK_ID = sys.argv[1]

task = Task.get_task(task_id=SOURCE_TASK_ID)
upload_state = task.upload_artifact(
    name="har", artifact_object="/model_quant.har", wait_on_upload=True
)
print(f"uploaded har: {upload_state}")
upload_state = task.upload_artifact(
    name="report", artifact_object="/report.html", wait_on_upload=True
)
print(f"uploaded report: {upload_state}")
upload_state = task.upload_artifact(
    name="evaluation quantized",
    artifact_object="/evaluation_quantized.txt",
    wait_on_upload=True,
)
print(f"uploaded evaluation: {upload_state}")
task.upload_artifact(name="log", artifact_object="/hailo_sdk.client.log")
