## Method {#sec:method}

### Threshold baselines (experiment E0)

The simplest flood mapper is one rule: a pixel is water if its VH backscatter is below a threshold. Two thresholds were used.

**Otsu (run R00, the planned baseline).** Otsu's method [@otsu1979threshold] was applied once to the histogram of all VH values of the training chips, without labels, and gave {{otsu_threshold}} dB. A single global threshold was preferred to one threshold per chip because Otsu always splits a histogram in two: on the {{train_chips_no_water}} training chips that contain no water, a per-chip threshold marks a median of {{per_chip_otsu_false_alarm}} of the chip as water.

**Tuned threshold (run R00b, an added reference).** A check on the training and validation chips showed that Otsu's cut is not the best one. The threshold with the highest IoU on the training labels is {{tuned_threshold}} dB. It was added as a second, stronger reference so that the networks are not compared only with a baseline that is easy to beat. It was fitted on training labels alone.

### U-Net with three encoders

A U-Net [@ronneberger2015unet] has an encoder that reduces the image step by step to a small, information-rich representation, a decoder that brings it back to full size, and skip connections that hand the encoder's fine detail to the decoder at each scale. All models here end in one sigmoid output per pixel, the probability of water, and accept any image size divisible by 32, so the same model is trained on 256×256 crops and applied to full 512×512 chips.

- **Scratch U-Net.** Five levels with 16, 32, 64, 128 and 256 filters. Each level is two 3×3 convolutions, each followed by batch normalisation [@ioffe2015batch] and ReLU. Max-pooling halves the resolution on the way down; on the way up the feature map is upsampled by two, concatenated with the skip connection and passed through the same pair of convolutions. {{params_scratch}} million parameters.
- **MobileNetV2 U-Net (the default).** The encoder is MobileNetV2 [@sandler2018mobilenetv2] up to its last block at 1/32 of the input size. Four intermediate feature maps, at 1/2, 1/4, 1/8 and 1/16 scale, serve as skip connections. The decoder has four blocks with 256, 128, 64 and 32 filters, built like the scratch decoder, and a final upsampling stage with 16 filters; MobileNetV2 halves the resolution in its very first layer, so there is no skip connection at full resolution. {{params_mobilenet}} million parameters, of which {{params_mobilenet_decoder}} million are in the decoder. The encoder weights are initialised randomly.
- **MobileNetV2 U-Net with ImageNet weights.** The same network with the encoder initialised from ImageNet classification weights [@deng2009imagenet]. This needs exactly three input channels, which the default input provides. The weights were learned on colour photographs; the network is given standardised radar channels instead and all layers are trained, so whether the weights still help is part of RQ3.

### Losses

Let $p_i$ be the predicted water probability of pixel $i$, $y_i \in \{0, 1\}$ its label, and $V$ the set of labelled pixels in a batch. Unlabelled pixels are left out of every sum.

$$\mathcal{L}_{\mathrm{BCE}} = -\frac{1}{|V|}\sum_{i \in V}\bigl[y_i \log p_i + (1 - y_i)\log(1 - p_i)\bigr]$$

$$\mathcal{L}_{\mathrm{Dice}} = 1 - \frac{2\sum_{i \in V} p_i y_i + \varepsilon}{\sum_{i \in V} p_i + \sum_{i \in V} y_i + \varepsilon}, \qquad \varepsilon = 10^{-6}$$

Binary cross-entropy judges every pixel separately, so with rare water it is dominated by land pixels. The Dice loss [@milletari2016vnet] measures the overlap between predicted and true water and ignores how many land pixels were easy. The default is their sum, $\mathcal{L}_{\mathrm{BCE}} + \mathcal{L}_{\mathrm{Dice}}$.

### Training

One epoch consists of {{cfg_crops_per_chip}} random {{cfg_crop_size}}×{{cfg_crop_size}} crops of every training chip, each randomly flipped and rotated by a multiple of 90°, with image and mask transformed identically: 1,008 crops, or 126 batches of {{cfg_batch_size}}. Models were trained with Adam [@kingma2015adam] at a learning rate of {{cfg_learning_rate}} for at most {{cfg_epochs}} epochs. After every epoch the model was scored on the full validation chips. The validation water IoU drove three mechanisms: the learning rate was halved after {{cfg_reduce_lr_patience}} epochs without improvement (down to {{cfg_min_learning_rate}}), training stopped after {{cfg_early_stopping_patience}} epochs without improvement, and the weights of the best epoch were kept. Training used mixed precision (16-bit arithmetic inside the network, 32-bit for the output and the loss).

These defaults were fixed after a five-epoch smoke test and a first full run that was inspected on validation data only (@sec:implementation), then tagged in the code repository before any other model was trained.

### Metrics

With TP, FP and FN the numbers of true-positive, false-positive and false-negative water pixels among the labelled pixels, and a decision threshold of {{cfg_threshold}} on the predicted probability:

$$\mathrm{IoU} = \frac{TP}{TP + FP + FN}, \qquad \mathrm{precision} = \frac{TP}{TP + FP}, \qquad \mathrm{recall} = \frac{TP}{TP + FN}, \qquad F_1 = \frac{2\,TP}{2\,TP + FP + FN}$$

The primary metric is the **pooled water IoU**: TP, FP and FN are summed over all chips of a split before dividing once. Precision, recall and F1 are pooled in the same way. As a secondary metric the **mean per-chip IoU** is the plain average of the chips' individual IoU values; chips with no water in the label and none predicted have an undefined IoU and are left out of that average. The two metrics answer different questions. The pooled IoU is dominated by chips with much water; the per-chip mean gives a chip with a small pond the same weight as a chip that is half flooded.

The same tested metric code scores the threshold baselines and every network.

### Experimental design and protocol

@tbl:design lists the runs. Each study changes one factor and keeps everything else at the default, which serves as the shared reference of all three studies. Every configuration was trained three times with seeds 0, 1 and 2; the seed fixes the initial weights and the sequence of crops.

Table: The experiments. Each study varies one factor; the default configuration (bold) is shared. {#tbl:design}

| Study | Question | Configurations | Runs |
|:---|:---|:---|:---|
| E0 | Is learning better than a threshold? | Otsu on VH; tuned VH threshold (added) | R00, R00b |
| E1 | RQ1, input channels | VH · VV+VH · **VV+VH+ratio** | R04–R09, R01–R03 |
| E2 | RQ2, loss | BCE · Dice · **BCE+Dice** | R10–R15, R01–R03 |
| E3 | RQ3, encoder | Scratch U-Net · **MobileNetV2** · MobileNetV2 + ImageNet | R16–R21, R01–R03 |

Three rules were followed throughout.

1. **Choices are made on validation data only.** Stopping, checkpoint selection and the choice of the "best" model use the {{n_val}} validation chips.
2. **Test and Bolivia are scored once per finished run.** No setting was changed after a test score had been seen. The first model was inspected on validation data, the defaults were frozen, and only then was it scored on test.
3. **Only the factor under study changes.** Splits, seeds and all other settings are identical across configurations; a check at the start of each experiment confirms that the training and scoring code is byte-identical to the frozen version.

**Reading differences.** Results are reported as mean ± standard deviation over the three seeds. A difference between two configurations is treated as real only if it is larger than the larger of their two standard deviations. For the main model, a second kind of uncertainty is also given: a 95% bootstrap interval obtained by resampling the test chips, which shows how much the score depends on which chips happen to be in the test set.

### Implementation {#sec:implementation}

The code is written in Python with TensorFlow {{tf_version}} and Keras {{keras_version}} and ran on a laptop with an {{gpu_name}} (4 GB) under WSL2 with 7.5 GB of RAM. Every run trains in its own process and writes its full configuration, its training history, the per-chip scores and the code version next to the model, so each number in this report can be traced to a file. Two findings from the smoke test shaped the set-up and are worth passing on.

**Compilation.** Compiling the training step with XLA, TensorFlow's just-in-time compiler, reduced the time per step from {{step_ms_plain}} ms to {{step_ms_xla}} ms on this machine. Without it the planned runs would not have fitted the time available.

**Batch-normalisation momentum.** The MobileNetV2 that ships with Keras uses a batch-normalisation momentum of 0.999, which means the running statistics used at validation and test time need thousands of updates to catch up with the weights. An epoch here has 126 updates. In the smoke test the training loss fell normally while the validation IoU stayed at exactly {{smoke_val_iou_bn999}} for all five epochs; with a momentum of 0.99 in every layer it reached {{smoke_val_iou_bn99}} (@fig:bn). Because checkpointing, learning-rate reduction and early stopping all depend on the validation score, the momentum was set to 0.99 for every model. Anyone who trains a Keras application model from scratch with few updates per epoch is likely to meet the same problem.

![The same five-epoch training run with two batch-normalisation momenta. Training behaves alike in both; the validation score only follows when the running statistics can keep up.](../figures/smoke_test_bn_momentum.png){#fig:bn width=88%}
