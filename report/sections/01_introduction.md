## Introduction {#sec:intro}

Floods are the most frequent kind of disaster. Of the 7,348 major disaster events recorded worldwide between 2000 and 2019, 3,254 were floods [@undrr2020human], and satellite records show that the share of the world's population living in places that flood has grown [@tellman2021satellite]. When a flood is under way, the first thing responders need is a map of where the water is. Satellites can provide one for a whole region within hours, if the water can be seen.

Seeing it is the difficulty. Floods arrive with heavy cloud, and an optical satellite cannot see through cloud. Synthetic aperture radar (SAR) can. A radar satellite sends its own microwave pulses and records what is scattered back, so it works at night and through cloud. The Sentinel-1 mission images most land every 6 to 12 days at 10 m resolution and its data are free [@torres2012gmes]. Calm water is smooth: it reflects the side-looking radar pulse away from the satellite, so open water appears dark in a radar image. This is the physical basis of every radar flood map.

"Dark means water" is, however, only a rule of thumb. Tarmac, sand, radar shadow and some crops are dark too, while water roughened by wind or broken by vegetation can be as bright as land. Operational systems started from thresholds on the backscatter [@otsu1979threshold; @bereczky2022sentinel]; over the last years convolutional networks, above all U-Nets [@ronneberger2015unet], have taken over because they can use the spatial context around a pixel. Sen1Floods11 [@bonafilia2020sen1floods11] made this research comparable by publishing Sentinel-1 chips of eleven flood events with water masks, 446 of them labelled by hand.

Much of the published work on this dataset uses large models: ResNet-50 encoders [@yadav2022attentive], or foundation models with hundreds of millions of parameters [@jakubik2023foundation; @schmalstieg2026distillation]. This project asks the opposite question. It studies what a small network can do when it is trained from 252 hand-labelled chips on a laptop with a 4 GB GPU, and which design choices are worth their cost.

**Research question.** How well can a lightweight U-Net segment flood water in single-date Sentinel-1 imagery under these constraints, and which design choices matter? Four sub-questions structure the work:

- **RQ1, input channels.** Does adding the VV band and the VV/VH ratio to the VH band help?
- **RQ2, loss.** Water is a small minority of the pixels. Which loss copes best with that: binary cross-entropy (BCE), Dice, or their sum?
- **RQ3, encoder.** Is a light MobileNetV2 encoder, with or without ImageNet weights, better than a small U-Net trained from nothing, and what does each cost in parameters, time and memory?
- **RQ4, generalisation.** Does the performance hold on a flood (Bolivia) that the model has never seen?

**Contributions.**

1. A complete and reproducible pipeline for Sentinel-1 flood segmentation that runs on a 4 GB laptop GPU: {{n_unet_runs}} training runs of about {{mean_train_minutes}} minutes each, with every number in this report generated from a saved result file.
2. A controlled ablation of input channels, loss function and encoder, each with three random seeds and a frozen protocol in which the test data were scored once per run, together with the cost of each encoder.
3. An error analysis that shows where and why the models fail, an honest comparison with two threshold baselines and with published results, and three findings that are easy to miss: a default setting of a standard encoder that silently disables validation, a threshold baseline whose strength depends on whether the flood is a familiar one, and model selection on validation data that picks the model which transfers worst.
