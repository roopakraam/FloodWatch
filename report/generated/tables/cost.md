Table: What each encoder costs and what it delivers. Step time, prediction time and memory come from a controlled benchmark of each model alone on the GPU (batch of 8 crops of 256×256 pixels; one 512×512 chip); epochs and minutes are means of the three training runs. {#tbl:cost}

| Encoder | Parameters (millions) | Training step (ms) | Prediction per chip (ms) | Peak GPU memory, training (MB) | Epochs until stop | Minutes per run | Test IoU |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Scratch U-Net | 1.97 | 42 | 5.3 | 551 | 37 | 4.5 | 0.655 ± 0.010 |
| MobileNetV2 (default) | 5.42 | 56 | 9.0 | 618 | 35 | 6.2 | 0.667 ± 0.002 |
| MobileNetV2 + ImageNet | 5.42 | 58 | 6.5 | 612 | 51 | 9.0 | 0.662 ± 0.004 |
