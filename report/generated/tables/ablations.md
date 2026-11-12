Table: The three ablation studies: mean ± standard deviation over three seeds. Each configuration differs from the default in exactly one factor. {#tbl:ablations}

| Study | Configuration | Validation IoU | Test IoU | Test precision | Test recall | Bolivia IoU |
|:---|:---|---:|---:|---:|---:|---:|
| Reference | Default (VV+VH+ratio, BCE+Dice, MobileNetV2) | 0.639 ± 0.009 | 0.667 ± 0.002 | 0.836 ± 0.020 | 0.767 ± 0.018 | 0.637 ± 0.017 |
| E1 · input | VH only |  |  |  |  |  |
|  | VV+VH |  |  |  |  |  |
| E2 · loss | BCE loss |  |  |  |  |  |
|  | Dice loss |  |  |  |  |  |
| E3 · encoder | Scratch U-Net |  |  |  |  |  |
|  | MobileNetV2 + ImageNet |  |  |  |  |  |
