Table: Published water IoU on the Sen1Floods11 hand-labelled test chips and on Bolivia, next to this work. Training data, inputs and the IoU definition differ between papers (see text), so the comparison is indicative only. {#tbl:published}

| Method | Input | Test IoU | Bolivia IoU |
|:---|:---|---:|---:|
| FCNN with ResNet-50 (dataset baseline) [@bonafilia2020sen1floods11] | Sentinel-1 | 0.493 | – |
| U-Net (Konapala et al.) [@konapala2021exploring] | Sentinel-1 | 0.450 | – |
| Attentive U-Net (ResNet-50 encoder from ImageNet) [@yadav2022attentive] | Sentinel-1: VV, VH, VV/VH ratio | 0.672 | 0.66 |
| Fusion network (ResNet-50 encoders) [@yadav2022attentive] | Sentinel-1 + elevation + permanent-water mask | 0.695 | 0.69 |
| Linear model on radar features [@iselborn2023importance] | Sentinel-1 | 0.497 | 0.655 |
| Gradient-boosted trees on radar features [@iselborn2023importance] | Sentinel-1 | 0.471 | 0.498 |
| DeepLabV3+ with self- / semi-supervised pre-training [@patel2021evaluating] | Sentinel-1 | 0.669 | 0.716 |
| BASNet [@bai2021enhancement] | Sentinel-1 + Sentinel-2 | 0.645 | 0.789 |
| Prithvi-100M fine-tuned (100 M parameters) [@prithvi2023sen1floods11] | Sentinel-2, six optical bands | 0.805 | 0.780 |
| EfficientViT-B0 distilled from Prithvi-EO-2.0 (0.7 M parameters) [@schmalstieg2026distillation] | optical (Sentinel-2) | 0.787 | – |
| This work: Otsu threshold on VH | Sentinel-1: VH | 0.433 | 0.653 |
| This work: Tuned VH threshold | Sentinel-1: VH | 0.524 | 0.531 |
| This work: **Default U-Net, MobileNetV2 encoder, 5.4 M parameters** | Sentinel-1: VV, VH, ratio | 0.667 ± 0.002 | 0.637 ± 0.017 |
