Table: Published water IoU on the hand-labelled test chips of Sen1Floods11 and on Bolivia, next to this work. "Pooled": one IoU over all pixels of the split; "per chip": the mean of the chips' IoU values. Training data, inputs and model selection differ between papers (see text), so the comparison is indicative only. {#tbl:published}

| Method | Input | IoU | Test | Bolivia |
|:---|:---|:---|---:|---:|
| FCNN (ResNet-50), trained on the hand labels [@bonafilia2020sen1floods11] | Sentinel-1 | per chip | 0.313 | 0.352 |
| FCNN (ResNet-50), trained on Sentinel-2 weak labels [@bonafilia2020sen1floods11] | Sentinel-1 | per chip | 0.408 | 0.316 |
| Otsu threshold on VH [@bonafilia2020sen1floods11] | Sentinel-1 | per chip | 0.359 | 0.386 |
| FCNN of the dataset paper [@bonafilia2020sen1floods11], as listed in [@yadav2022attentive] | Sentinel-1 | not stated | 0.493 | – |
| U-Net [@konapala2021exploring], as listed in [@yadav2022attentive] | Sentinel-1 | not stated | 0.450 | – |
| Attentive U-Net (ResNet-50 encoder, ImageNet weights) [@yadav2022attentive] | Sentinel-1: VV, VH, ratio | not stated | 0.672 | 0.66 |
| Fusion network (ResNet-50 encoders) [@yadav2022attentive] | Sentinel-1, elevation, permanent water | not stated | 0.695 | 0.69 |
| DeepLabv3+, random initial weights [@patel2021evaluating] | Sentinel-1 | pooled | 0.635 | – |
| DeepLabv3+, self-supervised pre-training [@patel2021evaluating] | Sentinel-1 | pooled | 0.669 | – |
| Linear model on radar features [@iselborn2023importance] | Sentinel-1 | pooled | 0.497 | 0.655 |
| Gradient-boosted trees on radar features [@iselborn2023importance] | Sentinel-1 | pooled | 0.471 | 0.498 |
| BASNet [@bai2021enhancement] | Sentinel-1 + Sentinel-2 | pooled | 0.645 | 0.789 |
| Prithvi-100M, fine-tuned (100 M parameters) [@prithvi2023sen1floods11] | Sentinel-2, six optical bands | not stated | 0.805 | 0.780 |
| EfficientViT-B0 distilled from Prithvi-EO-2.0 (0.7 M parameters) [@schmalstieg2026distillation] | Sentinel-2 (optical) | not stated | 0.787 | – |
| This work: Otsu threshold on VH | Sentinel-1: VH | pooled | 0.433 | 0.653 |
| This work: Tuned VH threshold | Sentinel-1: VH | pooled | 0.524 | 0.531 |
| This work: **Default U-Net (MobileNetV2 encoder, 5.4 M parameters)** | Sentinel-1: VV, VH, ratio | pooled | 0.667 ± 0.002 | 0.637 ± 0.017 |
| This work: Otsu threshold on VH | Sentinel-1: VH | per chip | 0.260 | 0.442 |
| This work: Tuned VH threshold | Sentinel-1: VH | per chip | 0.279 | 0.322 |
| This work: **Default U-Net (MobileNetV2 encoder, 5.4 M parameters)** | Sentinel-1: VV, VH, ratio | per chip | 0.398 ± 0.012 | 0.382 ± 0.004 |
