# Fusion method performance comparison

## M3FD

### RGB
238 epochs completed in 1.228 hours.
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.1.0a0+32f93b1 CUDA:0 (NVIDIA GeForce RTX 2080 Ti, 11012MiB)
2026-05-12 17:42:39
YOLO11n summary (fused): 238 layers, 2,583,127 parameters, 0 gradients, 6.32 GFLOPs
val: Scanning /root/.clearml/venvs-builds/3.10/task_repository/3d-net-clearml.git/workspace/dataset_rgb/labels/val.cache... 628 images, 0 backgrounds, 0 corrupt: 100% 628/628 [00:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  68% 27/40 [00:05<00:01, 10.99it/s]
2026-05-12 17:42:44
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 40/40 [00:06<00:00,  6.48it/s]
                   all        628       4555      0.834      0.554      0.635      0.436
                person        446       1708      0.814      0.425      0.562      0.287
                   car        443       2588      0.847      0.772      0.839      0.604
                   bus         53         61      0.829      0.638      0.686      0.552
            motorcycle         55         58      0.764      0.392      0.407      0.257
                 truck        109        140      0.916      0.542      0.679      0.479
Speed: 0.4ms preprocess, 3.9ms inference, 0.0ms loss, 1.0ms postprocess per image

### Thermal
300 epochs completed in 1.208 hours.
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.1.0a0+32f93b1 CUDA:0 (NVIDIA RTX A5000, 24248MiB)
YOLO11n summary (fused): 238 layers, 2,583,127 parameters, 0 gradients, 6.32 GFLOPs
val: Scanning /root/.clearml/venvs-builds/3.10/task_repository/3d-net-clearml.git/workspace/dataset_thermal/labels/val.cache... 628 images, 0 backgrounds, 0 corrupt: 100% 628/628 [00:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   5% 2/40 [00:00<00:08,  4.58it/s]
2026-05-12 17:41:04
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 40/40 [00:04<00:00,  9.72it/s]
                   all        628       4555      0.849      0.607      0.705      0.462
                person        446       1708      0.912      0.686      0.833      0.481
                   car        443       2588      0.871      0.759      0.828      0.577
                   bus         53         61      0.723      0.541      0.616      0.425
            motorcycle         55         58       0.84      0.414      0.506      0.292
                 truck        109        140      0.898      0.636      0.742      0.538
2026-05-12 17:41:09
Speed: 0.3ms preprocess, 2.7ms inference, 0.0ms loss, 0.5ms postprocess per image

### Early Fusion
300 epochs completed in 1.895 hours.
onnx model exported to workspace/dataset_rgbt/Results/yolo11n-RGBT-earlyfusion_8b1e2932/weights/best.onnx
2026-05-01 11:01:49
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.1.0a0+32f93b1 CUDA:0 (NVIDIA GeForce RTX 2080 Ti, 11012MiB)
YOLO11n-RGBT-earlyfusion summary (fused): 240 layers, 2,583,271 parameters, 0 gradients, 6.35 GFLOPs
val: Scanning /root/.clearml/venvs-builds/3.10/task_repository/3d-net-clearml.git/workspace/dataset_rgbt/visible/val.cache... 628 images, 0 backgrounds, 0 corrupt: 100% 628/628 [00:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   0% 0/40 [00:00<?, ?it/s]
2026-05-01 11:01:54
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  18% 7/40 [00:05<00:14,  2.21it/s]
2026-05-01 11:01:59
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 40/40 [00:09<00:00,  4.44it/s]
2026-05-01 11:02:04
                   all        628       4555      0.806      0.579      0.652       0.44
                person        446       1708      0.879      0.668      0.799      0.458
                   car        443       2588      0.861      0.786      0.854       0.61
                   bus         53         61      0.785      0.557      0.578      0.417
            motorcycle         55         58       0.67      0.328      0.355      0.227
                 truck        109        140      0.835      0.557      0.673      0.489
Speed: 0.5ms preprocess, 3.0ms inference, 0.0ms loss, 1.1ms postprocess per image


### Mid Fusion
300 epochs completed in 2.137 hours.
onnx model exported to workspace/dataset_rgbt/Results/yolo11n-RGBT-midfusion_4707fde3/weights/best.onnx
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.1.0a0+32f93b1 CUDA:0 (NVIDIA GeForce RTX 2080 Ti, 11012MiB)
YOLO11n-RGBT-midfusion summary (fused): 324 layers, 3,785,663 parameters, 0 gradients, 9.31 GFLOPs
2026-05-01 13:42:48
val: Scanning /root/.clearml/venvs-builds/3.10/task_repository/3d-net-clearml.git/workspace/dataset_rgbt/visible/val.cache... 628 images, 0 backgrounds, 0 corrupt: 100% 628/628 [00:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   8% 3/40 [00:03<00:49,  1.33s/it]
2026-05-01 13:42:53
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 40/40 [00:08<00:00,  4.77it/s]
2026-05-01 13:42:59
                   all        628       4555      0.788       0.62      0.671      0.458
                person        446       1708        0.8      0.761      0.814      0.478
                   car        443       2588       0.82      0.807      0.856      0.616
                   bus         53         61      0.781       0.59      0.622      0.454
            motorcycle         55         58      0.663       0.34      0.363      0.212
                 truck        109        140      0.875        0.6      0.702      0.529
Speed: 0.5ms preprocess, 3.5ms inference, 0.0ms loss, 1.8ms postprocess per image

### Mid P3 Fusion
300 epochs completed in 1.752 hours.
onnx model exported to workspace/dataset_rgbt/Results/yolo11n-RGBT-midP3fusion_b4558e7c/weights/best.onnx
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.1.0a0+32f93b1 CUDA:0 (NVIDIA RTX A5000, 24248MiB)
YOLO11n-RGBT-midP3fusion summary (fused): 272 layers, 2,690,111 parameters, 0 gradients, 8.13 GFLOPs
val: Scanning /root/.clearml/venvs-builds/3.10/task_repository/3d-net-clearml.git/workspace/dataset_rgbt/visible/val.cache... 628 images, 0 backgrounds, 0 corrupt: 100% 628/628 [00:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   2% 1/40 [00:00<00:32,  1.20it/s]
2026-05-02 21:43:21
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  65% 26/40 [00:06<00:01,  9.69it/s]
2026-05-02 21:43:26
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 40/40 [00:08<00:00,  4.97it/s]
                   all        628       4555       0.86      0.559      0.654      0.451
                person        446       1708       0.91      0.657       0.81      0.468
                   car        443       2588      0.883      0.787      0.864      0.621
                   bus         53         61      0.829      0.525       0.54      0.424
            motorcycle         55         58      0.753      0.276      0.342      0.219
                 truck        109        140      0.928      0.553      0.716      0.521
2026-05-02 21:43:32
Speed: 0.6ms preprocess, 3.6ms inference, 0.0ms loss, 1.0ms postprocess per image

### Mid CMA 3 fusion
300 epochs completed in 1.787 hours.
ONNX model exported to workspace/dataset_rgbt/Results/yolo11n-RGBT-midCMA3fusion_698d0d4b/weights/best.onnx
2026-05-04 09:53:13
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.1.0a0+32f93b1 CUDA:0 (NVIDIA RTX A5000, 24248MiB)
YOLO11n-RGBT-midCMA3fusion summary (fused): 355 layers, 3,732,046 parameters, 0 gradients, 9.18 GFLOPs
val: Scanning /root/.clearml/venvs-builds/3.10/task_repository/3d-net-clearml.git/workspace/dataset_rgbt/visible/val.cache... 628 images, 0 backgrounds, 0 corrupt: 100% 628/628 [00:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  20% 8/40 [00:03<00:07,  4.13it/s]
2026-05-04 09:53:19
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 40/40 [00:05<00:00,  6.85it/s]
                   all        628       4555      0.801      0.598      0.652      0.449
                person        446       1708      0.862      0.708      0.811      0.477
                   car        443       2588       0.85      0.803      0.858       0.62
                   bus         53         61      0.737      0.557      0.599      0.441
            motorcycle         55         58      0.686      0.345      0.349      0.216
                 truck        109        140      0.871      0.577      0.643       0.49
Speed: 0.4ms preprocess, 3.5ms inference, 0.0ms loss, 0.7ms postprocess per image

### Mid CFT 3 Fusion
300 epochs completed in 3.148 hours.
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.1.0a0+32f93b1 CUDA:0 (NVIDIA RTX A5000, 24248MiB)
YOLO11n-RGBT-midCFT3fusion summary: 705 layers, 13,088,575 parameters, 0 gradients, 978.96 GFLOPs
val: Scanning /root/.clearml/venvs-builds/3.10/task_repository/3d-net-clearml.git/workspace/dataset_rgbt/visible/val.cache... 628 images, 0 backgrounds, 0 corrupt: 100% 628/628 [00:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   2% 1/40 [00:00<00:20,  1.93it/s]
2026-05-04 17:26:52
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 40/40 [00:06<00:00,  6.37it/s]
2026-05-04 17:26:58
2026-05-04 15:26:54,128 - clearml.Task - INFO - Completed model upload to azure://clearmlstorage.blob.core.windows.net/trainingartefacts/YOLO11-3DNet/yolo11nrgbt_3dnet_train.2257f92166ed49b694e33c3e3db993fc/models/best.pt
                   all        628       4555      0.815      0.586      0.649      0.439
                person        446       1708      0.851      0.724      0.819      0.483
                   car        443       2588      0.831      0.808      0.853      0.617
                   bus         53         61      0.827       0.47      0.535      0.375
            motorcycle         55         58      0.736      0.345      0.366      0.223
                 truck        109        140      0.828      0.585      0.671      0.496
Speed: 0.4ms preprocess, 4.8ms inference, 0.0ms loss, 1.0ms postprocess per image

### Late Fusion
300 epochs completed in 2.603 hours.
onnx model exported to workspace/dataset_rgbt/Results/yolo11n-RGBT-latefusion_7827a223/weights/best.onnx
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.1.0a0+32f93b1 CUDA:0 (NVIDIA GeForce RTX 2080 Ti, 11012MiB)
2026-05-01 11:46:41
YOLO11n-RGBT-latefusion summary (fused): 420 layers, 5,124,399 parameters, 0 gradients, 12.21 GFLOPs
val: Scanning /root/.clearml/venvs-builds/3.10/task_repository/3d-net-clearml.git/workspace/dataset_rgbt/visible/val.cache... 628 images, 0 backgrounds, 0 corrupt: 100% 628/628 [00:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   5% 2/40 [00:02<00:55,  1.46s/it]
2026-05-01 11:46:47
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  90% 36/40 [00:08<00:00,  8.25it/s]
2026-05-01 11:46:52
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 40/40 [00:09<00:00,  4.40it/s]
                   all        628       4555      0.803      0.623       0.69      0.467
                person        446       1708      0.848      0.707      0.808      0.473
                   car        443       2588      0.844      0.803      0.862      0.619
                   bus         53         61      0.703      0.607      0.628      0.485
            motorcycle         55         58      0.776      0.414      0.496      0.279
                 truck        109        140      0.845      0.585      0.657      0.477
Speed: 0.5ms preprocess, 4.2ms inference, 0.0ms loss, 1.0ms postprocess per image

### Score Fusion
300 epochs completed in 2.162 hours.
onnx model exported to workspace/dataset_rgbt/Results/yolo11n-RGBT-scorefusion_1534b0e8/weights/best.onnx
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.1.0a0+32f93b1 CUDA:0 (NVIDIA RTX A5000, 24248MiB)
2026-05-03 09:10:25
YOLO11n-RGBT-scorefusion summary (fused): 471 layers, 5,165,950 parameters, 0 gradients, 12.58 GFLOPs
val: Scanning /root/.clearml/venvs-builds/3.10/task_repository/3d-net-clearml.git/workspace/dataset_rgbt/visible/val.cache... 628 images, 0 backgrounds, 0 corrupt: 100% 628/628 [00:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  30% 12/40 [00:03<00:04,  6.53it/s]
2026-05-03 09:10:31
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 40/40 [00:06<00:00,  6.46it/s]
                   all        628       4555      0.784      0.582      0.664      0.445
                person        446       1708      0.833      0.728      0.809      0.472
                   car        443       2588       0.79      0.792      0.846      0.599
                   bus         53         61      0.823      0.475      0.601      0.439
            motorcycle         55         58      0.622       0.31      0.359      0.227
                 truck        109        140      0.851      0.607      0.703      0.489
Speed: 0.4ms preprocess, 3.6ms inference, 0.0ms loss, 0.6ms postprocess per image

## RTDOD

### RGB
300 epochs completed in 4.246 hours.
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.1.0a0+32f93b1 CUDA:0 (NVIDIA RTX A5000, 24248MiB)
YOLO11n summary (fused): 238 layers, 2,583,322 parameters, 0 gradients, 6.32 GFLOPs
val: Scanning /root/.clearml/venvs-builds/3.10/task_repository/3d-net-clearml.git/workspace/dataset_rgb/labels/val.cache... 1221 images, 201 backgrounds, 0 corrupt: 100% 1221/1221 [00:00
2026-05-12 20:53:33
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  17% 13/77 [00:04<00:10,  6.29it/s]
2026-05-12 20:53:38
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 77/77 [00:08<00:00,  9.05it/s]
                   all       1221      13737      0.833       0.72      0.784      0.489
                person        518       4741      0.891      0.788      0.877      0.584
                   car        676       6231       0.91      0.872      0.927      0.664
            motorcycle        615       1610      0.833      0.766      0.845      0.601
               bicycle        226        281       0.75       0.52      0.545       0.28
                   bus        118        241      0.881      0.644      0.782      0.421
                 truck        211        633      0.735      0.733      0.729      0.381
2026-05-12 20:53:44
Speed: 0.2ms preprocess, 1.4ms inference, 0.0ms loss, 1.8ms postprocess per image

### Thermal
300 epochs completed in 5.462 hours.
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.1.0a0+32f93b1 CUDA:0 (NVIDIA GeForce RTX 2080 Ti, 11012MiB)
YOLO11n summary (fused): 238 layers, 2,583,322 parameters, 0 gradients, 6.32 GFLOPs
val: Scanning /root/.clearml/venvs-builds/3.10/task_repository/3d-net-clearml.git/workspace/dataset_thermal/labels/val.cache... 1221 images, 201 backgrounds, 0 corrupt: 100% 1221/1221 [00:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   4% 3/77 [00:02<01:07,  1.10it/s]
2026-05-19 12:28:27
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  47% 36/77 [00:08<00:02, 18.69it/s]
2026-05-19 12:28:33
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 77/77 [00:12<00:00,  6.40it/s]
                   all       1221      13737      0.748      0.592      0.638      0.303
                person        518       4741      0.728      0.585      0.645      0.288
                   car        676       6231      0.827      0.757      0.789      0.437
            motorcycle        615       1610      0.799      0.596      0.662      0.334
               bicycle        226        281       0.55      0.274      0.323     0.0856
                   bus        118        241      0.804      0.697      0.775      0.403
                 truck        211        633      0.778      0.643       0.63      0.269
2026-05-19 12:28:38
Speed: 0.2ms preprocess, 1.7ms inference, 0.0ms loss, 1.0ms postprocess per image

### Early Fusion
300 epochs completed in 6.944 hours.
onnx model exported to workspace/dataset_rgbt/Results/yolo11n-RGBT-earlyfusion_db5ee497/weights/best.onnx
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.1.0a0+32f93b1 CUDA:0 (NVIDIA GeForce RTX 2080 Ti, 11012MiB)
YOLO11n-RGBT-earlyfusion summary (fused): 240 layers, 2,583,466 parameters, 0 gradients, 6.35 GFLOPs
val: Scanning /root/.clearml/venvs-builds/3.10/task_repository/3d-net-clearml.git/workspace/dataset_rgbt/visible/val.cache... 1221 images, 201 backgrounds, 0 corrupt: 100% 1221/1221 [00:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  64% 49/77 [00:22<00:02,  9.44it/s]
2026-05-02 14:13:55
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 77/77 [00:25<00:00,  3.06it/s]
                   all       1221      13737      0.833      0.708      0.784      0.489
                person        518       4741      0.904      0.774       0.88      0.583
                   car        676       6231      0.926      0.838      0.922      0.646
            motorcycle        615       1610      0.887       0.74      0.853      0.599
               bicycle        226        281       0.59      0.498      0.441      0.219
                   bus        118        241      0.863      0.679      0.832      0.466
                 truck        211        633      0.828       0.72      0.777      0.422
2026-05-02 14:14:01
Speed: 0.4ms preprocess, 2.9ms inference, 0.0ms loss, 1.9ms postprocess per image

### Mid Fusion
300 epochs completed in 7.967 hours.
onnx model exported to workspace/dataset_rgbt/Results/yolo11n-RGBT-midfusion_b9801ea8/weights/best.onnx
2026-05-02 15:15:39
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.1.0a0+32f93b1 CUDA:0 (NVIDIA GeForce RTX 2080 Ti, 11012MiB)
YOLO11n-RGBT-midfusion summary (fused): 324 layers, 3,785,858 parameters, 0 gradients, 9.31 GFLOPs
val: Scanning /root/.clearml/venvs-builds/3.10/task_repository/3d-net-clearml.git/workspace/dataset_rgbt/visible/val.cache... 1221 images, 201 backgrounds, 0 corrupt: 100% 1221/1221 [00:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  99% 76/77 [00:25<00:00,  9.55it/s]
2026-05-02 15:16:06
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 77/77 [00:26<00:00,  2.96it/s]
                   all       1221      13737      0.806      0.759      0.813      0.506
                person        518       4741      0.865      0.818      0.892      0.596
                   car        676       6231      0.895      0.859      0.917      0.639
            motorcycle        615       1610      0.838      0.784      0.859      0.614
               bicycle        226        281      0.615      0.506      0.516      0.257
                   bus        118        241      0.797       0.83      0.877      0.506
                 truck        211        633      0.823      0.758      0.814      0.425
Speed: 0.3ms preprocess, 3.6ms inference, 0.0ms loss, 2.1ms postprocess per image

### Mid P3 Fusion
300 epochs completed in 5.694 hours.
onnx model exported to workspace/dataset_rgbt/Results/yolo11n-RGBT-midP3fusion_8f1ac657/weights/best.onnx
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.1.0a0+32f93b1 CUDA:0 (NVIDIA RTX A5000, 24248MiB)
YOLO11n-RGBT-midP3fusion summary (fused): 272 layers, 2,690,306 parameters, 0 gradients, 8.13 GFLOPs
val: Scanning /root/.clearml/venvs-builds/3.10/task_repository/3d-net-clearml.git/workspace/dataset_rgbt/visible/val.cache... 1221 images, 201 backgrounds, 0 corrupt: 100% 1221/1221 [00:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   0% 0/77 [00:00<?, ?it/s]
2026-05-03 01:50:20
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   9% 7/77 [00:05<00:35,  1.95it/s]
2026-05-03 01:50:25
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 77/77 [00:09<00:00,  7.74it/s]
2026-05-03 01:50:30
                   all       1221      13737      0.828       0.75      0.815      0.509
                person        518       4741      0.881      0.811      0.895      0.601
                   car        676       6231      0.909      0.853      0.917      0.645
            motorcycle        615       1610      0.878       0.77      0.864      0.614
               bicycle        226        281      0.667      0.541      0.555      0.322
                   bus        118        241      0.812      0.788      0.865      0.465
                 truck        211        633      0.821      0.736      0.795      0.405
Speed: 0.2ms preprocess, 1.4ms inference, 0.0ms loss, 0.5ms postprocess per image

### Mid CMA3 Fusion
300 epochs completed in 6.385 hours.
ONNX model exported to workspace/dataset_rgbt/Results/yolo11n-RGBT-midCMA3fusion_8b739b3c/weights/best.onnx
2026-05-04 14:39:47
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.1.0a0+32f93b1 CUDA:0 (NVIDIA RTX A5000, 24248MiB)
YOLO11n-RGBT-midCMA3fusion summary (fused): 355 layers, 3,732,241 parameters, 0 gradients, 9.18 GFLOPs
val: Scanning /root/.clearml/venvs-builds/3.10/task_repository/3d-net-clearml.git/workspace/dataset_rgbt/visible/val.cache... 1221 images, 201 backgrounds, 0 corrupt: 100% 1221/1221 [00:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   3% 2/77 [00:02<01:42,  1.37s/it]
2026-05-04 14:39:53
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  62% 48/77 [00:08<00:01, 15.32it/s]
2026-05-04 14:39:58
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 77/77 [00:10<00:00,  7.03it/s]
                   all       1221      13737      0.858      0.741      0.817      0.501
                person        518       4741      0.886      0.791       0.89      0.589
                   car        676       6231      0.924      0.849       0.92      0.645
            motorcycle        615       1610       0.89       0.75      0.859      0.613
               bicycle        226        281       0.72      0.544      0.567      0.297
                   bus        118        241      0.886      0.768      0.859      0.461
                 truck        211        633      0.843      0.741      0.805      0.399
Speed: 0.2ms preprocess, 2.0ms inference, 0.0ms loss, 0.9ms postprocess per image

### Mid CFT3 Fusion
300 epochs completed in 11.104 hours.
ONNX export skipped (EXPORT_ONNX=0)
2026-05-04 23:31:55,281 - clearml.Task - INFO - Completed model upload to azure://clearmlstorage.blob.core.windows.net/trainingartefacts/YOLO11-3DNet/yolo11nrgbt_3dnet_train.c1ab6c74aca3424590b2d02340c20996/models/best.pt
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.1.0a0+32f93b1 CUDA:0 (NVIDIA RTX A5000, 24248MiB)
YOLO11n-RGBT-midCFT3fusion summary: 705 layers, 13,088,770 parameters, 0 gradients, 978.96 GFLOPs
val: Scanning /root/.clearml/venvs-builds/3.10/task_repository/3d-net-clearml.git/workspace/dataset_rgbt/visible/val.cache... 1221 images, 201 backgrounds, 0 corrupt: 100% 1221/1221 [00:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   1% 1/77 [00:00<00:34,  2.21it/s]
2026-05-05 01:32:05
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  38% 29/77 [00:07<00:03, 12.75it/s]
2026-05-05 01:32:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 77/77 [00:11<00:00,  6.97it/s]
2026-05-05 01:32:16
                   all       1221      13737      0.841       0.75      0.817       0.51
                person        518       4741      0.899       0.79      0.887       0.59
                   car        676       6231      0.913      0.845      0.917      0.646
            motorcycle        615       1610      0.876      0.771      0.861      0.609
               bicycle        226        281      0.674      0.543      0.547      0.296
                   bus        118        241      0.877      0.788      0.873      0.504
                 truck        211        633      0.807      0.764      0.815      0.418
Speed: 0.2ms preprocess, 4.2ms inference, 0.0ms loss, 1.1ms postprocess per image

### Late Fusion
300 epochs completed in 9.195 hours.
onnx model exported to workspace/dataset_rgbt/Results/yolo11n-RGBT-latefusion_6fd7b832/weights/best.onnx
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.1.0a0+32f93b1 CUDA:0 (NVIDIA GeForce RTX 2080 Ti, 11012MiB)
2026-05-02 16:28:36
YOLO11n-RGBT-latefusion summary (fused): 420 layers, 5,124,786 parameters, 0 gradients, 12.21 GFLOPs
val: Scanning /root/.clearml/venvs-builds/3.10/task_repository/3d-net-clearml.git/workspace/dataset_rgbt/visible/val.cache... 1221 images, 201 backgrounds, 0 corrupt: 100% 1221/1221 [00:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  16% 12/77 [00:20<00:30,  2.13it/s]
2026-05-02 16:28:57
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 77/77 [00:26<00:00,  2.94it/s]
2026-05-02 16:29:03
                   all       1221      13737      0.834      0.726      0.811      0.509
                person        518       4741      0.898       0.79      0.894      0.605
                   car        676       6231      0.917      0.847      0.922      0.646
            motorcycle        615       1610      0.873      0.729      0.852      0.615
               bicycle        226        281      0.614      0.463      0.517      0.265
                   bus        118        241      0.868      0.793      0.883      0.517
                 truck        211        633      0.836      0.736      0.796      0.403
Speed: 0.3ms preprocess, 4.3ms inference, 0.0ms loss, 2.8ms postprocess per image

### Score Fusion
300 epochs completed in 7.410 hours.
onnx model exported to workspace/dataset_rgbt/Results/yolo11n-RGBT-scorefusion_bba4ac03/weights/best.onnx
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.1.0a0+32f93b1 CUDA:0 (NVIDIA RTX A5000, 24248MiB)
YOLO11n-RGBT-scorefusion summary (fused): 471 layers, 5,166,340 parameters, 0 gradients, 12.58 GFLOPs
val: Scanning /root/.clearml/venvs-builds/3.10/task_repository/3d-net-clearml.git/workspace/dataset_rgbt/visible/val.cache... 1221 images, 201 backgrounds, 0 corrupt: 100% 1221/1221 [00:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   3% 2/77 [00:02<01:24,  1.12s/it]
2026-05-03 14:36:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  47% 36/77 [00:07<00:03, 13.26it/s]
2026-05-03 14:36:05
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 77/77 [00:10<00:00,  7.04it/s]
                   all       1221      13737      0.808       0.71      0.773      0.482
                person        518       4741      0.879      0.772      0.871       0.57
                   car        676       6231      0.809      0.851      0.892      0.632
            motorcycle        615       1610      0.866      0.728      0.831      0.592
               bicycle        226        281      0.694      0.488      0.478      0.252
                   bus        118        241      0.875      0.699       0.84      0.455
                 truck        211        633      0.723       0.72      0.725      0.394
2026-05-03 14:36:11
Speed: 0.2ms preprocess, 2.3ms inference, 0.0ms loss, 0.7ms postprocess per image


## Visdrone Dronevehicle

### RGB
300 epochs completed in 11.488 hours.
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.1.0a0+32f93b1 CUDA:0 (NVIDIA GeForce RTX 2080 Ti, 11012MiB)
YOLO11n summary (fused): 238 layers, 2,582,737 parameters, 0 gradients, 6.32 GFLOPs
val: Scanning /root/.clearml/venvs-builds/3.10/task_repository/3d-net-clearml.git/workspace/dataset_rgb/labels/val.cache... 1469 images, 2 backgrounds, 0 corrupt: 100% 1469/1469 [00:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   0% 0/92 [00:00<?, ?it/s]
2026-05-19 02:51:18
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 92/92 [00:21<00:00,  4.26it/s]
2026-05-19 02:51:40
                   all       1469      24490      0.868       0.77       0.85      0.525
                   car       1401      21313      0.914      0.847      0.918      0.533
                 truck        588       2388      0.805      0.609      0.718      0.411
                   bus        291        789      0.886      0.854      0.913      0.631
Speed: 0.3ms preprocess, 1.9ms inference, 0.0ms loss, 2.2ms postprocess per image

### Thermal
300 epochs completed in 9.355 hours.
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.1.0a0+32f93b1 CUDA:0 (NVIDIA GeForce RTX 2080 Ti, 11012MiB)
YOLO11n summary (fused): 238 layers, 2,582,737 parameters, 0 gradients, 6.32 GFLOPs
val: Scanning /root/.clearml/venvs-builds/3.10/task_repository/3d-net-clearml.git/workspace/dataset_thermal/labels/val.cache... 1469 images, 2 backgrounds, 0 corrupt: 100% 1469/1469 [00:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   2% 2/92 [00:00<00:34,  2.57it/s]
2026-05-19 22:18:22
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 92/92 [00:23<00:00,  3.99it/s]
                   all       1469      24490      0.918      0.894      0.945      0.696
                   car       1401      21313      0.954      0.962      0.983      0.707
                 truck        588       2388      0.867      0.792      0.884      0.618
                   bus        291        789      0.935      0.928      0.966      0.762
2026-05-19 22:18:38
Speed: 0.3ms preprocess, 1.9ms inference, 0.0ms loss, 1.3ms postprocess per image

### Early Fusion
300 epochs completed in 7.850 hours.
onnx model exported to workspace/dataset_rgbt/Results/yolo11n-RGBT-earlyfusion_2386ffa3/weights/best.onnx
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.1.0a0+32f93b1 CUDA:0 (NVIDIA RTX A5000, 24248MiB)
YOLO11n-RGBT-earlyfusion summary (fused): 240 layers, 2,582,881 parameters, 0 gradients, 6.34 GFLOPs
val: Scanning /root/.clearml/venvs-builds/3.10/task_repository/3d-net-clearml.git/workspace/dataset_rgbt/visible/val.cache... 1469 images, 2 backgrounds, 0 corrupt: 100% 1469/1469 [00:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  52% 48/92 [04:50<00:05,  8.64it/s]
2026-05-02 15:21:22
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 92/92 [04:54<00:00,  3.20s/it]
2026-05-02 15:21:28
                   all       1469      24490      0.905      0.907      0.945      0.693
                   car       1401      21313      0.946      0.967      0.985      0.711
                 truck        588       2388      0.844      0.817      0.883      0.613
                   bus        291        789      0.926      0.935      0.969      0.754
Speed: 7.6ms preprocess, 23.8ms inference, 0.0ms loss, 3.3ms postprocess per image

### Mid Fusion
300 epochs completed in 10.664 hours.
onnx model exported to workspace/dataset_rgbt/Results/yolo11n-RGBT-midfusion_8fdbb0fb/weights/best.onnx
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.1.0a0+32f93b1 CUDA:0 (NVIDIA RTX A5000, 24248MiB)
YOLO11n-RGBT-midfusion summary (fused): 324 layers, 3,785,273 parameters, 0 gradients, 9.31 GFLOPs
val: Scanning /root/.clearml/venvs-builds/3.10/task_repository/3d-net-clearml.git/workspace/dataset_rgbt/visible/val.cache... 1469 images, 2 backgrounds, 0 corrupt: 100% 1469/1469 [00:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  40% 37/92 [00:27<00:05,  9.90it/s]
2026-05-02 18:00:58
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 92/92 [00:32<00:00,  2.82it/s]
2026-05-02 18:01:04
                   all       1469      24490      0.916       0.91      0.948      0.697
                   car       1401      21313      0.951      0.964      0.986      0.717
                 truck        588       2388      0.851      0.831      0.888      0.617
                   bus        291        789      0.945      0.935      0.969      0.757
Speed: 0.4ms preprocess, 4.9ms inference, 0.0ms loss, 4.7ms postprocess per image

### Mid P3 Fusion
300 epochs completed in 9.255 hours.
onnx model exported to workspace/dataset_rgbt/Results/yolo11n-RGBT-midP3fusion_c559f2cd/weights/best.onnx
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.1.0a0+32f93b1 CUDA:0 (NVIDIA RTX A5000, 24248MiB)
2026-05-03 05:24:48
YOLO11n-RGBT-midP3fusion summary (fused): 272 layers, 2,689,721 parameters, 0 gradients, 8.13 GFLOPs
val: Scanning /root/.clearml/venvs-builds/3.10/task_repository/3d-net-clearml.git/workspace/dataset_rgbt/visible/val.cache... 1469 images, 2 backgrounds, 0 corrupt: 100% 1469/1469 [00:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  78% 72/92 [00:30<00:01, 13.62it/s]
2026-05-03 05:25:20
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 92/92 [00:31<00:00,  2.90it/s]
                   all       1469      24490      0.921      0.908      0.951      0.707
                   car       1401      21313      0.956      0.968      0.987      0.725
                 truck        588       2388      0.857      0.825      0.895      0.629
                   bus        291        789      0.948      0.932       0.97      0.766
Speed: 0.4ms preprocess, 4.6ms inference, 0.0ms loss, 2.9ms postprocess per image

### Mid CMA3 Fusion
300 epochs completed in 12.643 hours.
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.1.0a0+32f93b1 CUDA:0 (NVIDIA RTX A5000, 24248MiB)
YOLO11n-RGBT-midCMA3fusion summary (fused): 355 layers, 3,731,656 parameters, 0 gradients, 9.18 GFLOPs
val: Scanning /root/.clearml/venvs-builds/3.10/task_repository/3d-net-clearml.git/workspace/dataset_rgbt/visible/val.cache... 1469 images, 2 backgrounds, 0 corrupt: 100% 1469/1469 [00:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   0% 0/92 [00:00<?, ?it/s]
2026-05-18 21:26:02
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 92/92 [00:26<00:00,  3.53it/s]
2026-05-18 21:26:07
                   all       1469      24490      0.922      0.912       0.95      0.701
                   car       1401      21313      0.954      0.965      0.986      0.722
                 truck        588       2388      0.869      0.826      0.896      0.631
                   bus        291        789      0.944      0.944      0.968       0.75
Speed: 0.3ms preprocess, 5.8ms inference, 0.0ms loss, 2.9ms postprocess per image

### Mid CFT3 Fusion
300 epochs completed in 31.108 hours.
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.1.0a0+32f93b1 CUDA:0 (NVIDIA GeForce RTX 2080 Ti, 11012MiB)
2026-05-05 04:37:58
YOLO11n-RGBT-midCFT3fusion summary: 705 layers, 13,088,185 parameters, 0 gradients
val: Scanning /root/.clearml/venvs-builds/3.10/task_repository/3d-net-clearml.git/workspace/dataset_rgbt/visible/val.cache... 1469 images, 2 backgrounds, 0 corrupt: 100% 1469/1469 [00:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   1% 1/92 [00:00<01:25,  1.07it/s]
2026-05-05 04:38:05
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   2% 2/92 [00:06<05:05,  3.40s/it]
2026-05-05 04:38:15
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   4% 4/92 [00:20<08:23,  5.73s/it]
2026-05-05 04:38:21
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  55% 51/92 [00:27<00:05,  7.89it/s]
2026-05-05 04:38:27
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  99% 91/92 [00:32<00:00,  8.23it/s]
2026-05-05 04:38:32
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 92/92 [00:32<00:00,  2.83it/s]
                   all       1469      24490      0.928      0.898      0.947      0.696
                   car       1401      21313      0.955      0.962      0.985      0.718
                 truck        588       2388      0.875      0.795      0.885      0.611
                   bus        291        789      0.955      0.937      0.972      0.758

### Late Fusion
300 epochs completed in 11.966 hours.
onnx model exported to workspace/dataset_rgbt/Results/yolo11n-RGBT-latefusion_3a493382/weights/best.onnx
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.1.0a0+32f93b1 CUDA:0 (NVIDIA RTX A5000, 24248MiB)
YOLO11n-RGBT-latefusion summary (fused): 420 layers, 5,123,625 parameters, 0 gradients, 12.20 GFLOPs
val: Scanning /root/.clearml/venvs-builds/3.10/task_repository/3d-net-clearml.git/workspace/dataset_rgbt/visible/val.cache... 1469 images, 2 backgrounds, 0 corrupt: 100% 1469/1469 [00:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  70% 64/92 [00:26<00:02, 11.64it/s]
2026-05-02 19:18:54
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 92/92 [00:29<00:00,  3.15it/s]
                   all       1469      24490      0.921      0.899      0.944      0.696
                   car       1401      21313      0.959      0.959      0.986      0.718
                 truck        588       2388      0.861      0.802      0.882      0.612
                   bus        291        789      0.945      0.937      0.964      0.756
Speed: 0.3ms preprocess, 6.4ms inference, 0.0ms loss, 2.8ms postprocess per image

### Score Fusion
300 epochs completed in 12.995 hours.
onnx model exported to workspace/dataset_rgbt/Results/yolo11n-RGBT-scorefusion_c6ceb1b6/weights/best.onnx
2026-05-03 20:13:35
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.1.0a0+32f93b1 CUDA:0 (NVIDIA RTX A5000, 24248MiB)
YOLO11n-RGBT-scorefusion summary (fused): 471 layers, 5,165,170 parameters, 0 gradients, 12.57 GFLOPs
val: Scanning /root/.clearml/venvs-builds/3.10/task_repository/3d-net-clearml.git/workspace/dataset_rgbt/visible/val.cache... 1469 images, 2 backgrounds, 0 corrupt: 100% 1469/1469 [00:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  95% 87/92 [00:29<00:00, 11.51it/s]
2026-05-03 20:14:07
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 92/92 [00:29<00:00,  3.08it/s]
                   all       1469      24490      0.914      0.878      0.937      0.684
                   car       1401      21313       0.94      0.955      0.982      0.712
                 truck        588       2388      0.857      0.773      0.866        0.6
                   bus        291        789      0.945      0.907      0.962      0.741
Speed: 0.4ms preprocess, 6.1ms inference, 0.0ms loss, 2.6ms postprocess per image

## 3D-Net

### RGB
300 epochs completed in 5.128 hours.
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.1.0a0+32f93b1 CUDA:0 (NVIDIA GeForce RTX 2080 Ti, 11012MiB)
YOLO11n summary (fused): 238 layers, 2,583,322 parameters, 0 gradients, 6.32 GFLOPs
val: Scanning /root/.clearml/venvs-builds/3.10/task_repository/3d-net-clearml.git/workspace/dataset_rgb/labels/val.cache... 1889 images, 907 backgrounds, 0 corrupt: 100% 1889/1889 [00:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   0% 0/119 [00:00<?, ?it/s]
2026-05-18 14:42:25
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  62% 74/119 [00:05<00:01, 24.26it/s]
2026-05-18 14:42:30
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 119/119 [00:08<00:00, 14.63it/s]
                   all       1889       1404      0.937      0.937       0.97      0.808
                person         85        191          1      0.986      0.995      0.819
                   car        625        798      0.818      0.982      0.974       0.86
            motorcycle          4          4      0.876          1      0.995      0.822
               bicycle         31         31      0.967      0.956      0.989      0.713
                   bus          7          7          1      0.996      0.995      0.915
                 truck        357        373      0.963        0.7      0.874      0.721
2026-05-18 14:42:35
Speed: 0.2ms preprocess, 1.7ms inference, 0.0ms loss, 0.5ms postprocess per image

### Thermal
300 epochs completed in 3.202 hours.
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.1.0a0+32f93b1 CUDA:0 (NVIDIA RTX A5000, 24248MiB)
YOLO11n summary (fused): 238 layers, 2,583,322 parameters, 0 gradients, 6.32 GFLOPs
val: Scanning /root/.clearml/venvs-builds/3.10/task_repository/3d-net-clearml.git/workspace/dataset_thermal/labels/val.cache... 1889 images, 907 backgrounds, 0 corrupt: 100% 1889/1889 [00:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   0% 0/119 [00:00<?, ?it/s]
2026-05-19 12:20:48
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  82% 98/119 [00:05<00:00, 24.57it/s]
2026-05-19 12:20:54
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 119/119 [00:06<00:00, 18.86it/s]
                   all       1889       1404      0.713      0.632      0.643      0.345
                person         85        191      0.212      0.089      0.079     0.0302
                   car        625        798      0.793      0.855       0.81      0.476
            motorcycle          4          4      0.579       0.75      0.624      0.146
               bicycle         31         31      0.812      0.699      0.732       0.29
                   bus          7          7          1      0.795      0.995      0.692
                 truck        357        373      0.881      0.606      0.617      0.433
Speed: 0.2ms preprocess, 1.3ms inference, 0.0ms loss, 0.6ms postprocess per image

### Early Fusion
300 epochs completed in 3.398 hours.
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.1.0a0+32f93b1 CUDA:0 (NVIDIA RTX A5000, 24248MiB)
YOLO11n-RGBT-earlyfusion summary (fused): 240 layers, 2,583,466 parameters, 0 gradients, 6.35 GFLOPs
val: Scanning /root/.clearml/venvs-builds/3.10/task_repository/3d-net-clearml.git/workspace/dataset_rgbt/visible/val.cache... 1889 images, 907 backgrounds, 0 corrupt: 100% 1889/1889 [00:00
2026-05-13 15:21:44
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  17% 20/119 [00:04<00:07, 13.40it/s]
2026-05-13 15:21:49
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 119/119 [00:08<00:00, 13.76it/s]
2026-05-13 15:21:55
                   all       1889       1404      0.963      0.966      0.987      0.801
                person         85        191          1      0.997      0.995      0.813
                   car        625        798       0.88       0.97       0.97      0.853
            motorcycle          4          4      0.905          1      0.995       0.76
               bicycle         31         31          1      0.948      0.988      0.666
                   bus          7          7          1      0.996      0.995      0.928
                 truck        357        373      0.994      0.884      0.977      0.785
Speed: 0.3ms preprocess, 1.1ms inference, 0.0ms loss, 0.6ms postprocess per image

### Mid Fusion
300 epochs completed in 4.539 hours.
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.1.0a0+32f93b1 CUDA:0 (NVIDIA RTX A5000, 24248MiB)
YOLO11n-RGBT-midfusion summary (fused): 324 layers, 3,785,858 parameters, 0 gradients, 9.31 GFLOPs
val: Scanning /root/.clearml/venvs-builds/3.10/task_repository/3d-net-clearml.git/workspace/dataset_rgbt/visible/val.cache... 1889 images, 907 backgrounds, 0 corrupt: 100% 1889/1889 [00:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   2% 2/119 [00:01<01:29,  1.31it/s]
2026-05-13 16:30:23
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  79% 94/119 [00:06<00:01, 22.77it/s]
2026-05-13 16:30:28
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 119/119 [00:08<00:00, 14.19it/s]
                   all       1889       1404      0.952      0.929      0.978      0.797
                person         85        191      0.998      0.995      0.995       0.81
                   car        625        798      0.858       0.97      0.971      0.855
            motorcycle          4          4      0.897          1      0.995      0.811
               bicycle         31         31          1      0.924      0.995        0.7
                   bus          7          7      0.995          1      0.995      0.867
                 truck        357        373      0.966      0.685      0.917      0.737
Speed: 0.3ms preprocess, 1.7ms inference, 0.0ms loss, 0.4ms postprocess per image

### Mid P3 Fusion
300 epochs completed in 5.346 hours.
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.1.0a0+32f93b1 CUDA:0 (NVIDIA GeForce RTX 2080 Ti, 11012MiB)
YOLO11n-RGBT-midP3fusion summary (fused): 272 layers, 2,690,306 parameters, 0 gradients, 8.13 GFLOPs
val: Scanning /root/.clearml/venvs-builds/3.10/task_repository/3d-net-clearml.git/workspace/dataset_rgbt/visible/val.cache... 1889 images, 907 backgrounds, 0 corrupt: 100% 1889/1889 [00:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   1% 1/119 [00:00<01:39,  1.18it/s]
2026-05-13 21:12:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  34% 40/119 [00:06<00:04, 17.03it/s]
2026-05-13 21:12:16
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 119/119 [00:11<00:00, 10.78it/s]
2026-05-13 21:12:21
                   all       1889       1404      0.964      0.904      0.955      0.793
                person         85        191          1      0.993      0.995       0.81
                   car        625        798      0.881      0.954      0.973      0.862
            motorcycle          4          4      0.945          1      0.995      0.911
               bicycle         31         31          1      0.953      0.993      0.679
                   bus          7          7      0.977      0.857      0.909        0.8
                 truck        357        373       0.98      0.664      0.864      0.698
Speed: 0.4ms preprocess, 2.7ms inference, 0.0ms loss, 0.5ms postprocess per image

### Mid CMA3 Fusion
205 epochs completed in 3.847 hours.
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.1.0a0+32f93b1 CUDA:0 (NVIDIA RTX A5000, 24248MiB)
YOLO11n-RGBT-midCMA3fusion summary (fused): 355 layers, 3,732,241 parameters, 0 gradients, 9.18 GFLOPs
val: Scanning /root/.clearml/venvs-builds/3.10/task_repository/3d-net-clearml.git/workspace/dataset_rgbt/visible/val.cache... 1889 images, 907 backgrounds, 0 corrupt: 100% 1889/1889 [00:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   3% 3/119 [00:02<01:50,  1.05it/s]
2026-05-17 11:45:22
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  92% 109/119 [00:08<00:00, 19.04it/s]
2026-05-17 11:45:28
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 119/119 [00:09<00:00, 12.49it/s]
                   all       1889       1404      0.912      0.926      0.972      0.766
                person         85        191      0.991          1      0.995      0.776
                   car        625        798      0.777      0.981      0.962      0.822
            motorcycle          4          4      0.822          1      0.995      0.785
               bicycle         31         31      0.989          1      0.995      0.638
                   bus          7          7          1      0.857      0.995       0.87
                 truck        357        373       0.89      0.719      0.891      0.707
Speed: 0.3ms preprocess, 2.1ms inference, 0.0ms loss, 0.4ms postprocess per image

### Mid CFT3 Fusion
300 epochs completed in 15.248 hours.
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.1.0a0+32f93b1 CUDA:0 (NVIDIA GeForce RTX 2080 Ti, 11012MiB)
2026-05-17 23:05:21
YOLO11n-RGBT-midCFT3fusion summary: 705 layers, 13,088,770 parameters, 0 gradients, 978.96 GFLOPs
val: Scanning /root/.clearml/venvs-builds/3.10/task_repository/3d-net-clearml.git/workspace/dataset_rgbt/visible/val.cache... 1889 images, 907 backgrounds, 0 corrupt: 100% 1889/1889 [00:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   3% 3/119 [00:03<02:09,  1.12s/it]
2026-05-17 23:05:26
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  49% 58/119 [00:08<00:05, 10.65it/s]
2026-05-17 23:05:32
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 119/119 [00:13<00:00,  8.60it/s]
2026-05-17 23:05:37
                   all       1889       1404      0.925      0.947      0.981      0.798
                person         85        191      0.991          1      0.995        0.8
                   car        625        798      0.798      0.984      0.972      0.849
            motorcycle          4          4      0.834          1      0.995      0.824
               bicycle         31         31      0.979          1      0.995      0.676
                   bus          7          7          1      0.973      0.995      0.886
                 truck        357        373      0.951      0.723      0.933      0.756
Speed: 0.4ms preprocess, 4.2ms inference, 0.0ms loss, 0.8ms postprocess per image

### Late Fusion
300 epochs completed in 5.266 hours.
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.1.0a0+32f93b1 CUDA:0 (NVIDIA RTX A5000, 24248MiB)
YOLO11n-RGBT-latefusion summary (fused): 420 layers, 5,124,786 parameters, 0 gradients, 12.21 GFLOPs
val: Scanning /root/.clearml/venvs-builds/3.10/task_repository/3d-net-clearml.git/workspace/dataset_rgbt/visible/val.cache... 1889 images, 907 backgrounds, 0 corrupt: 100% 1889/1889 [00:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   2% 2/119 [00:01<01:40,  1.16it/s]
2026-05-13 17:14:08
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  72% 86/119 [00:07<00:01, 20.26it/s]
2026-05-13 17:14:14
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 119/119 [00:09<00:00, 13.08it/s]
                   all       1889       1404      0.971      0.918      0.984      0.798
                person         85        191          1      0.988      0.995      0.811
                   car        625        798       0.91      0.936      0.971      0.848
            motorcycle          4          4      0.944          1      0.995      0.792
               bicycle         31         31          1      0.894       0.99      0.689
                   bus          7          7      0.977          1      0.995      0.882
                 truck        357        373      0.996      0.689      0.956      0.766
Speed: 0.3ms preprocess, 2.1ms inference, 0.0ms loss, 0.4ms postprocess per image

### Score Fusion
300 epochs completed in 5.335 hours.
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.1.0a0+32f93b1 CUDA:0 (NVIDIA RTX A5000, 24248MiB)
YOLO11n-RGBT-scorefusion summary (fused): 471 layers, 5,166,340 parameters, 0 gradients, 12.58 GFLOPs
val: Scanning /root/.clearml/venvs-builds/3.10/task_repository/3d-net-clearml.git/workspace/dataset_rgbt/visible/val.cache... 1889 images, 907 backgrounds, 0 corrupt: 100% 1889/1889 [00:00
2026-05-13 21:11:47
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  29% 34/119 [00:04<00:05, 16.81it/s]
2026-05-13 21:11:53
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 119/119 [00:09<00:00, 12.85it/s]
2026-05-13 21:11:58
                   all       1889       1404       0.94      0.977      0.988      0.825
                person         85        191      0.997          1      0.995      0.813
                   car        625        798      0.831      0.985      0.976      0.865
            motorcycle          4          4      0.837          1      0.995      0.865
               bicycle         31         31      0.998          1      0.995       0.68
                   bus          7          7          1      0.969      0.995      0.926
                 truck        357        373      0.979      0.906      0.972      0.801
Speed: 0.3ms preprocess, 2.3ms inference, 0.0ms loss, 0.4ms postprocess per image


## 3D-Net Data Pool

### RGB
300 epochs completed in 15.100 hours.
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.1.0a0+32f93b1 CUDA:0 (NVIDIA RTX A5000, 24248MiB)
YOLO11n summary (fused): 238 layers, 2,583,322 parameters, 0 gradients, 6.32 GFLOPs
val: Scanning /root/.clearml/venvs-builds/3.10/task_repository/3d-net-clearml.git/workspace/dataset_rgb/labels/val.cache... 4989 images, 1111 backgrounds, 0 corrupt: 100% 4989/4989 [00:00
2026-05-19 22:30:05
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 312/312 [00:19<00:00, 15.76it/s]
2026-05-19 22:30:26
                   all       4989      42718      0.781      0.651       0.72      0.448
                person        843       5820      0.846      0.629      0.724      0.447
                   car       3040      30304      0.876      0.828      0.881      0.531
            motorcycle        659       1655      0.799      0.722       0.79      0.543
               bicycle        257        312      0.522      0.353      0.406      0.222
                   bus        465       1094      0.879      0.758      0.827      0.542
                 truck       1264       3533      0.764      0.616      0.691      0.405
Speed: 0.1ms preprocess, 1.1ms inference, 0.0ms loss, 0.5ms postprocess per image

### Thermal
300 epochs completed in 15.552 hours.
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.1.0a0+32f93b1 CUDA:0 (NVIDIA RTX A5000, 24248MiB)
YOLO11n summary (fused): 238 layers, 2,583,322 parameters, 0 gradients, 6.32 GFLOPs
2026-05-19 22:57:19
val: Scanning /root/.clearml/venvs-builds/3.10/task_repository/3d-net-clearml.git/workspace/dataset_thermal/labels/val.cache... 4989 images, 1111 backgrounds, 0 corrupt: 100% 4989/4989 [00:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  16% 49/312 [00:03<00:17, 15.41it/s]
2026-05-19 22:57:24
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 312/312 [00:19<00:00, 16.10it/s]
2026-05-19 22:57:40
                   all       4989      42718      0.776      0.604      0.655      0.392
                person        843       5820      0.726      0.424      0.513      0.221
                   car       3040      30304      0.904      0.865      0.905      0.611
            motorcycle        659       1655      0.756      0.545      0.604      0.308
               bicycle        257        312      0.554       0.26      0.264     0.0842
                   bus        465       1094      0.899      0.821      0.873      0.635
                 truck       1264       3533      0.818       0.71      0.768      0.492
Speed: 0.1ms preprocess, 1.0ms inference, 0.0ms loss, 0.5ms postprocess per image

### Early Fusion
300 epochs completed in 19.767 hours.
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.1.0a0+32f93b1 CUDA:0 (NVIDIA GeForce RTX 2080 Ti, 11012MiB)
YOLO11n-RGBT-earlyfusion summary (fused): 240 layers, 2,583,466 parameters, 0 gradients, 6.35 GFLOPs
val: Scanning /root/.clearml/venvs-builds/3.10/task_repository/3d-net-clearml.git/workspace/dataset_rgbt/visible/val.cache... 4989 images, 1111 backgrounds, 0 corrupt: 100% 4989/4989 [00:00
2026-05-16 15:36:16
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   0% 1/312 [00:00<01:33,  3.31it/s]
2026-05-16 15:36:22
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 312/312 [00:57<00:00,  5.43it/s]
                   all       4989      42718      0.785      0.694      0.754      0.497
                person        843       5820      0.796      0.635      0.718      0.428
                   car       3040      30304      0.906      0.895      0.935      0.661
            motorcycle        659       1655      0.779      0.683      0.766      0.517
               bicycle        257        312      0.548      0.411      0.434      0.216
                   bus        465       1094      0.889      0.804      0.881       0.64
                 truck       1264       3533      0.792      0.735       0.79      0.519
2026-05-16 15:37:20
Speed: 3.1ms preprocess, 1.8ms inference, 0.0ms loss, 0.6ms postprocess per image

### Mid Fusion
300 epochs completed in 26.083 hours.
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.1.0a0+32f93b1 CUDA:0 (NVIDIA GeForce RTX 2080 Ti, 11012MiB)
YOLO11n-RGBT-midfusion summary (fused): 324 layers, 3,785,858 parameters, 0 gradients, 9.31 GFLOPs
val: Scanning /root/.clearml/venvs-builds/3.10/task_repository/3d-net-clearml.git/workspace/dataset_rgbt/visible/val.cache... 4989 images, 1111 backgrounds, 0 corrupt: 100% 4989/4989 [00:00
2026-05-16 21:54:44
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   0% 0/312 [00:00<?, ?it/s]
2026-05-16 21:54:49
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   0% 1/312 [00:05<26:49,  5.18s/it]
2026-05-16 21:55:00
                   all       4989      42718      0.838      0.728      0.792      0.531
                person        843       5820      0.867      0.668      0.767      0.497
                   car       3040      30304      0.927      0.891      0.938      0.671
            motorcycle        659       1655      0.854      0.719      0.813      0.555
               bicycle        257        312      0.617        0.5      0.513      0.264
                   bus        465       1094      0.913      0.832      0.894      0.651
                 truck       1264       3533      0.852       0.76      0.827      0.551
Speed: 0.2ms preprocess, 3.9ms inference, 0.0ms loss, 1.0ms postprocess per image

### Mid P3 Fusion
300 epochs completed in 19.342 hours.
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.1.0a0+32f93b1 CUDA:0 (NVIDIA RTX A5000, 24248MiB)
YOLO11n-RGBT-midP3fusion summary (fused): 272 layers, 2,690,306 parameters, 0 gradients, 8.13 GFLOPs
val: Scanning /root/.clearml/venvs-builds/3.10/task_repository/3d-net-clearml.git/workspace/dataset_rgbt/visible/val.cache... 4989 images, 1111 backgrounds, 0 corrupt: 100% 4989/4989 [00:00
2026-05-16 14:38:13
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  13% 41/312 [00:04<00:19, 13.80it/s]
2026-05-16 14:38:19
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 312/312 [00:21<00:00, 14.43it/s]
                   all       4989      42718      0.831       0.72      0.783      0.532
                person        843       5820      0.869      0.666      0.766      0.497
                   car       3040      30304      0.922      0.892      0.936      0.675
            motorcycle        659       1655      0.874      0.721      0.817      0.558
               bicycle        257        312      0.641      0.458      0.488      0.265
                   bus        465       1094      0.901      0.824      0.883      0.646
                 truck       1264       3533      0.781      0.759      0.812      0.551
Speed: 0.1ms preprocess, 1.5ms inference, 0.0ms loss, 0.5ms postprocess per image

### Mid CMA3 Fusion
300 epochs completed in 24.396 hours.
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.1.0a0+32f93b1 CUDA:0 (NVIDIA RTX A5000, 24248MiB)
YOLO11n-RGBT-midCMA3fusion summary (fused): 355 layers, 3,732,241 parameters, 0 gradients, 9.18 GFLOPs
val: Scanning /root/.clearml/venvs-builds/3.10/task_repository/3d-net-clearml.git/workspace/dataset_rgbt/visible/val.cache... 4989 images, 1111 backgrounds, 0 corrupt: 100% 4989/4989 [00:00
2026-05-18 08:52:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   7% 22/312 [00:03<00:27, 10.37it/s]
2026-05-18 08:52:15
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 312/312 [00:26<00:00, 11.93it/s]
                   all       4989      42718      0.844      0.734      0.794      0.536
                person        843       5820      0.867      0.675      0.766      0.494
                   car       3040      30304      0.927      0.892      0.939      0.673
            motorcycle        659       1655      0.874      0.711       0.81       0.56
               bicycle        257        312      0.644      0.516      0.523      0.285
                   bus        465       1094      0.924      0.839      0.898      0.655
                 truck       1264       3533      0.829       0.77       0.83      0.552
Speed: 0.1ms preprocess, 1.9ms inference, 0.0ms loss, 0.5ms postprocess per image

### Mid CFT3 Fusion
300 epochs completed in 67.611 hours.
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.1.0a0+32f93b1 CUDA:0 (NVIDIA GeForce RTX 2080 Ti, 11012MiB)
YOLO11n-RGBT-midCFT3fusion summary: 705 layers, 13,088,770 parameters, 0 gradients, 978.96 GFLOPs
val: Scanning /root/.clearml/venvs-builds/3.10/task_repository/3d-net-clearml.git/workspace/dataset_rgbt/visible/val.cache... 4989 images, 1111 backgrounds, 0 corrupt: 100% 4989/4989 [00:00
2026-05-20 04:06:06
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   9% 28/312 [00:05<00:31,  8.99it/s]
2026-05-20 04:06:37
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 312/312 [00:33<00:00,  9.24it/s]
                   all       4989      42718      0.825      0.732      0.785      0.531
                person        843       5820      0.864      0.672      0.766      0.496
                   car       3040      30304      0.923      0.889      0.933      0.669
            motorcycle        659       1655       0.83      0.715      0.802      0.548
               bicycle        257        312      0.608      0.488      0.486      0.271
                   bus        465       1094      0.905      0.846      0.898      0.649
                 truck       1264       3533      0.818      0.783      0.826      0.555
Speed: 0.2ms preprocess, 3.4ms inference, 0.0ms loss, 0.5ms postprocess per image

### Late Fusion
300 epochs completed in 34.823 hours.
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.1.0a0+32f93b1 CUDA:0 (NVIDIA GeForce RTX 2080 Ti, 11012MiB)
2026-05-18 08:47:57
YOLO11n-RGBT-latefusion summary (fused): 420 layers, 5,124,786 parameters, 0 gradients, 12.21 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   9% 14/156 [00:04<00:28,  4.90it/s]
2026-05-18 08:48:02
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 156/156 [00:28<00:00,  5.41it/s]
                   all       4989      42718      0.851      0.739      0.805      0.549
                person        843       5820      0.849      0.678      0.763      0.503
                   car       3040      30304      0.922      0.894      0.938      0.673
            motorcycle        659       1655      0.837      0.727      0.817      0.575
               bicycle        257        312      0.763      0.535      0.599       0.33
                   bus        465       1094      0.909      0.841      0.894      0.655
                 truck       1264       3533      0.827      0.759      0.822      0.558
2026-05-18 08:48:28
Speed: 0.1ms preprocess, 1.4ms inference, 0.0ms loss, 1.1ms postprocess per image

### Score Fusion
300 epochs completed in 25.893 hours.
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.1.0a0+32f93b1 CUDA:0 (NVIDIA RTX A5000, 24248MiB)
YOLO11n-RGBT-scorefusion summary (fused): 471 layers, 5,166,340 parameters, 0 gradients, 12.58 GFLOPs
val: Scanning /root/.clearml/venvs-builds/3.10/task_repository/3d-net-clearml.git/workspace/dataset_rgbt/visible/val.cache... 4989 images, 1111 backgrounds, 0 corrupt: 100% 4989/4989 [00:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   0% 0/312 [00:00<?, ?it/s]
2026-05-17 22:11:08
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 312/312 [00:25<00:00, 12.45it/s]
                   all       4989      42718      0.859      0.724      0.797      0.542
                person        843       5820      0.896      0.659      0.755      0.495
                   car       3040      30304      0.927      0.886      0.931      0.666
            motorcycle        659       1655       0.89      0.731      0.837      0.588
               bicycle        257        312       0.67      0.538      0.545      0.306
                   bus        465       1094      0.924      0.808      0.892      0.656
                 truck       1264       3533      0.849      0.722       0.82      0.542
2026-05-17 22:11:35
Speed: 0.2ms preprocess, 2.0ms inference, 0.0ms loss, 0.5ms postprocess per image