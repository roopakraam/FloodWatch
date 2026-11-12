Table: The default U-Net against the two threshold rules. U-Net values are the mean ± standard deviation of three seeds; the threshold rules are deterministic. IoU, precision, recall and F1 are pooled over all labelled pixels of the split. {#tbl:main}

| Split | Method | Water IoU | Precision | Recall | F1 | Mean per-chip IoU |
|:---|:---|---:|---:|---:|---:|---:|
| Test | Otsu threshold (R00) | 0.433 | 0.483 | 0.809 | 0.605 | 0.260 |
|  | Tuned threshold (R00b) | 0.524 | 0.747 | 0.637 | 0.688 | 0.279 |
|  | Default U-Net (R01–R03) | 0.667 ± 0.002 | 0.836 ± 0.020 | 0.767 ± 0.018 | 0.800 ± 0.002 | 0.398 ± 0.012 |
| Bolivia | Otsu threshold (R00) | 0.653 | 0.731 | 0.859 | 0.790 | 0.442 |
|  | Tuned threshold (R00b) | 0.531 | 0.870 | 0.577 | 0.694 | 0.322 |
|  | Default U-Net (R01–R03) | 0.637 ± 0.017 | 0.878 ± 0.020 | 0.699 ± 0.023 | 0.778 ± 0.013 | 0.382 ± 0.004 |
