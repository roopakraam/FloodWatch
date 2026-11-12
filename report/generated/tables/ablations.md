Table: The three ablation studies: mean ± standard deviation over three seeds. Each configuration differs from the default in exactly one factor. {#tbl:ablations}

| Study | Configuration | Validation IoU | Test IoU | Test precision | Test recall | Bolivia IoU |
|:---|:---|---:|---:|---:|---:|---:|
| Reference | Default (VV+VH+ratio, BCE+Dice, MobileNetV2) | 0.639 ± 0.009 | 0.667 ± 0.002 | 0.836 ± 0.020 | 0.767 ± 0.018 | 0.637 ± 0.017 |
| E1 · input | VH only | 0.612 ± 0.008 | 0.634 ± 0.007 | 0.829 ± 0.015 | 0.730 ± 0.007 | 0.623 ± 0.018 |
|  | VV+VH | 0.650 ± 0.009 | 0.670 ± 0.006 | 0.854 ± 0.013 | 0.757 ± 0.018 | 0.632 ± 0.038 |
| E2 · loss | BCE loss | 0.616 ± 0.009 | 0.643 ± 0.011 | 0.853 ± 0.037 | 0.725 ± 0.016 | 0.559 ± 0.120 |
|  | Dice loss | 0.644 ± 0.007 | 0.660 ± 0.010 | 0.828 ± 0.011 | 0.765 ± 0.007 | 0.624 ± 0.019 |
| E3 · encoder | Scratch U-Net | 0.641 ± 0.009 | 0.655 ± 0.010 | 0.794 ± 0.001 | 0.790 ± 0.016 | 0.671 ± 0.071 |
|  | MobileNetV2 + ImageNet | 0.664 ± 0.007 | 0.662 ± 0.004 | 0.846 ± 0.002 | 0.752 ± 0.007 | 0.519 ± 0.019 |
