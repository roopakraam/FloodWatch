## Discussion {#sec:discussion}

**RQ1: use both polarisations, skip the ratio.** VV adds information that VH does not have, worth about {{gapabs_vh_vs_vvvh_test}} IoU on the test chips, and it improves precision and recall together. A hand-made ratio channel adds nothing, because the network can compute it. Its only use here is practical: it fills the third input channel that ImageNet weights require.

**RQ2: train with a Dice-based loss.** With water at about a tenth of the pixels, BCE alone under-predicts water, as expected, and it was the least stable loss on the unseen flood. Dice alone and BCE+Dice score the same; the combination gave the most consistent seeds, which is a reason to keep it as the default.

**RQ3: on familiar floods the encoder hardly matters; ImageNet weights are not worth it for radar.** A scratch U-Net with {{cost_scratch_params}} million parameters is within about a point of the MobileNetV2 U-Net with {{cost_default_params}} million. Pretraining on photographs raised the validation score, did nothing for the test score and made the model clearly worse on the unseen flood. The experiments do not show why. A plausible reading is that the pretrained encoder reaches a closer fit to the training floods (the validation chips come from the same floods) and that this closer fit is exactly what does not transfer. Testing that would need runs with a lower learning rate or a frozen encoder, which were outside the plan. The scratch U-Net is also the cheaper model to train and to run, so within the range studied here a larger or pretrained encoder buys nothing that can be measured on the test chips.

**RQ4: performance does not reliably hold on an unseen flood.** This is the most important result, and it is a negative one. On the test chips every network beats both thresholds by a wide margin. On Bolivia no configuration beats the Otsu threshold reliably; the default cannot be separated from it, the ImageNet models are far below it, and the scratch U-Net is above it in two seeds and below it in the third. The networks stay precise on Bolivia but miss water that is moderately dark, which a generous threshold catches. Two consequences follow. First, a simple unsupervised threshold is a serious competitor outside the training distribution and belongs in every such comparison. Second, validation data drawn from the training floods cannot be used to choose a model for new floods: here it selected the configuration that transferred worst.

**How much weight the numbers can bear.** Two sources of noise were measured. Between seeds, the test IoU moves by a few thousandths to about one point, the Bolivia IoU by two to twelve points. Between possible test sets, a bootstrap over chips moves the absolute IoU by about ten points in either direction, while paired differences between methods are much more stable. Differences of a point or less between configurations on the test chips, such as the default against VV+VH or against Dice, should therefore be treated as ties, and every statement about Bolivia rests on 15 chips from one flood.

**What limits radar-only mapping.** About a third of the best model's wrong test pixels are labelled water that is not dark in the radar image, and a further twentieth has no radar data at all. No change of architecture or loss can recover those. Part of what looks like model error on this benchmark is a mismatch between what the labels describe (water visible in optical and radar images together) and what a single radar image can show.

**Practical lessons.** Three things cost time or would have misled, and none of them is specific to floods. (1) A Keras application model trained from scratch with few updates per epoch needs a smaller batch-normalisation momentum; otherwise the validation score, and everything that depends on it, is meaningless for many epochs. (2) The baseline matters: Otsu is weak on the test chips and strong on Bolivia, a threshold tuned on labels is the reverse, and reporting only one of them would have told half the story in either direction. (3) Compiling the model made the difference between a plan that fits a laptop and one that does not.

## Limitations {#sec:limitations}

- **Little data, one held-out event.** 252 training chips, 90 test chips and 15 Bolivia chips. The generalisation result rests on a single unseen flood; another flood could behave differently.
- **Three seeds.** A standard deviation from three runs is itself uncertain, and the rule "gap larger than the standard deviation" is a convention, not a significance test. Several gaps that pass it have t-test p-values of 0.06 to 0.2.
- **One sensor, one date.** No pre-flood image, no elevation, no optical data. Each of these is known to help.
- **A single decision threshold of 0.5.** It was fixed in advance for every model. On validation, the first default run would have gained about {{thr_gain_r01}} IoU with a threshold of {{r01_best_thr}} (from {{r01_val_iou_thr05}} to {{r01_val_iou_best_thr}}); this was not used.
- **Defaults fixed after one look.** Learning-rate schedule, patience and batch size were set once, after inspecting one run on validation data. Other defaults might change close rankings. The early stop of run R18 at epoch {{R18_epochs}} shows that the noisy validation score can end a run too soon.
- **The cause of the ImageNet effect is not established.** It was observed in three seeds, not explained.
- **Labels and sensor disagree in places.** Some labelled water is invisible to radar, and some labelled pixels have no radar value. This lowers every radar-only score and blurs comparisons.
- **Timings come from one laptop.** They depend on the GPU, the driver, the compiler cache and the temperature of the machine, and are best read as ratios between models.
- **The published comparison is indicative.** Inputs, training data, model selection and the definition of IoU differ between papers, and some values were available only as quoted by later papers.

## Future work {#sec:future}

- **Select and validate by event.** Hold out whole floods during development (leave-one-event-out) so that model selection rewards transfer, and calibrate the decision threshold per event.
- **More data without more hand labels.** Use the 4,385 weakly labelled chips of Sen1Floods11, or self-supervised pre-training on unlabelled radar imagery, the one radar-only approach in @tbl:published that is clearly better on Bolivia.
- **More inputs.** Sentinel-2 when the sky is clear, a pre-flood radar image, elevation and a permanent-water mask.
- **Larger encoders** such as ResNet, now that the cost of the small ones is known, and ensembles of the three seeds.
- **Understand the ImageNet effect** with a lower learning rate for the encoder or a frozen encoder.

## Conclusion {#sec:conclusion}

A U-Net with a MobileNetV2 encoder and {{cost_default_params}} million parameters, trained in about {{cost_default_minutes}} minutes on a 4 GB laptop GPU from 252 hand-labelled chips, reaches a water IoU of {{default_test_iou_pm}} on the Sen1Floods11 test chips. That is {{default_minus_otsu_test_abs}} above an Otsu threshold, {{default_minus_tuned_test_abs}} above the best single threshold, and within a point of a published model with a much larger encoder. Of the design choices studied, two matter on familiar floods: using VV as well as VH, and a Dice-based loss instead of BCE alone. The ratio channel, the size of the encoder and ImageNet weights do not.

On a flood the models have never seen, the picture changes. No network beats the Otsu threshold reliably, the pretrained models are clearly worse than the same network trained from scratch, and the model that validation would select is the one that transfers worst. For flood mapping, where the next event is by definition a new one, this is the result that deserves most attention: how a model is validated matters as much as how it is built.
