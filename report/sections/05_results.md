## Results {#sec:results}

All {{n_unet_runs}} training runs finished; none failed or had to be repeated. A run took {{mean_train_minutes}} minutes on average (about {{mean_epochs_trained}} epochs of {{mean_sec_per_epoch}} seconds), and all runs together {{total_train_hours}} hours of GPU time. Every run is listed in @tbl:allruns in the appendix.

### The default U-Net against the threshold rules {#sec:main}

{{table:main}}

**On the test chips the network is far ahead** (@tbl:main). The default U-Net reaches a water IoU of {{default_test_iou_pm}}, against {{otsu_test_iou}} for the Otsu threshold and {{tuned_test_iou}} for the threshold tuned on the training labels. Compared with Otsu it gives up a little recall ({{default_test_recall}} against {{otsu_test_recall}}) and removes most of the false alarms (precision {{default_test_precision}} against {{otsu_test_precision}}). Compared with the tuned threshold it is better on both counts.

**Two kinds of uncertainty.** The three seeds agree closely on the test chips (standard deviation {{default_test_iou_std}}). The uncertainty that comes from the test set itself is much larger: resampling the 90 test chips gives a 95% interval of {{ci_R01_test}} for the IoU of run R01. Absolute IoU values on this dataset should therefore be read as "about 0.67", not to three decimals. Differences between methods on the same chips are much better determined. The lead of R01 over Otsu is {{diff_R01_R00_test}} (interval {{diffci_R01_R00_test}}) and over the tuned threshold {{diff_R01_R00b_test}} (interval {{diffci_R01_R00b_test}}), and the network wins in every resample.

**Per chip, every method looks much worse.** The mean per-chip IoU of the default U-Net is {{default_test_miou_pm}}, against {{otsu_test_miou}} and {{tuned_test_miou}} for the two thresholds. Most chips hold little water, and on such a chip a few hundred wrong pixels decide the score (@sec:errors). The pooled IoU is carried by the chips with large water bodies.

**Missing radar data costs about one point.** If the labelled pixels without a radar value are left out, the test IoU of the default U-Net rises from {{default_test_iou}} to {{default_test_iou_observed}}.

![Training of the three default runs. Grey: training crops; blue: validation chips; dots: the epoch whose weights were kept. Validation IoU swings strongly from epoch to epoch at the initial learning rate and settles as the rate is reduced. The kept epochs are peaks on a noisy plateau, which is why a single run cannot resolve differences of one or two IoU points.](../figures/report/training_curves.png){#fig:curves width=88%}

Training itself was unremarkable (@fig:curves). There is no sign of overfitting: training and validation loss fall together and end at a similar level, and early stopping ended the default runs after {{R02_epochs}} to {{R01_epochs}} epochs.

On Bolivia the picture is different: the default U-Net ({{default_bolivia_iou_pm}}) does not beat the Otsu threshold ({{otsu_bolivia_iou}}). @sec:rq4 returns to this.

### The three ablation studies

{{table:ablations}}

![The ablations on the test chips (left) and on Bolivia (right). Bars: mean of three seeds; whiskers: ± one standard deviation; dots: the individual seeds; vertical lines: the two threshold rules. Bars start at zero, so the differences between configurations look as small as they are.](../figures/report/ablations.png){#fig:ablations width=100%}

@tbl:ablations and @fig:ablations give the scores of all seven configurations and @tbl:gaps applies the rule fixed in advance: a difference counts only if it is larger than the spread between seeds.

{{table:gaps}}

#### Input channels (RQ1)

**Adding VV to VH helps; the ratio channel does not.** With VH alone the test IoU is {{vh_test_iou_pm}}; with VV and VH it is {{vvvh_test_iou_pm}}, a gain of {{gapabs_vh_vs_vvvh_test}}, about {{gapratio_vh_vs_vvvh_test}} times the spread between seeds. Precision and recall both improve ({{vh_test_precision}} to {{vvvh_test_precision}} and {{vh_test_recall}} to {{vvvh_test_recall}}). Adding the ratio as a third channel changes nothing measurable ({{default_test_iou_pm}}; gap {{gapabs_vvvh_vs_default_test}}, smaller than the spread). This is plausible: the ratio is VV − VH, a fixed linear combination of the other two channels, which the first convolution can form by itself. On Bolivia none of the three inputs is distinguishable from the others.

#### Loss (RQ2)

**BCE alone is the weakest loss; Dice and BCE+Dice cannot be told apart.** BCE reaches {{bce_test_iou_pm}} on the test chips, Dice {{dice_test_iou_pm}} and their sum {{default_test_iou_pm}}. The gaps between BCE and the other two are larger than the spread between seeds, although with three seeds a t-test stays inconclusive (p = {{welch_bce_vs_default_test}} against the default). The mechanism is the expected one. BCE has the highest precision ({{bce_test_precision}}) and the lowest recall ({{bce_test_recall}}): judged pixel by pixel, the cheapest way to be right is to under-predict the rare class. The Dice-based losses find more water (recall {{dice_test_recall}} and {{default_test_recall}}). BCE is also the least stable on Bolivia, where its three seeds give {{R10_bolivia_iou}}, {{R11_bolivia_iou}} and {{R12_bolivia_iou}}. Adding BCE to Dice does not raise the score, but the three seeds of the combined loss agree more closely on the test chips than those of Dice alone (standard deviation {{default_test_iou_std}} against {{dice_test_iou_std}}).

#### Encoder and cost (RQ3)

**On the test chips the three encoders are within about one IoU point of each other.** The scratch U-Net reaches {{scratch_test_iou_pm}}, MobileNetV2 {{default_test_iou_pm}} and MobileNetV2 with ImageNet weights {{imagenet_test_iou_pm}}. The gap between the scratch U-Net and the default ({{gapabs_scratch_vs_default_test}}) is only just larger than the spread between seeds.

**ImageNet weights help on validation, not on test, and they hurt on Bolivia.** Starting from ImageNet weights raises the validation IoU from {{default_val_iou_pm}} to {{imagenet_val_iou_pm}}. The test IoU does not move. On Bolivia the pretrained models fall to {{imagenet_bolivia_iou_pm}}, against {{default_bolivia_iou_pm}} for the same network trained from random weights: a gap of {{gapabs_default_vs_imagenet_bolivia}}, about {{gapratio_default_vs_imagenet_bolivia}} times the spread between seeds (p = {{welch_default_vs_imagenet_bolivia}}). The pretrained encoder fits the ten training floods better and carries over to the eleventh worse.

**The small scratch U-Net is the only model that keeps its level on Bolivia, but not reliably.** Its Bolivia IoU averages {{scratch_bolivia_iou}}, slightly above its test IoU, yet the three seeds give {{R16_bolivia_iou}}, {{R17_bolivia_iou}} and {{R18_bolivia_iou}}. The third stopped after {{R18_epochs}} epochs on an early validation peak (best epoch {{R18_best_epoch}}) and is the weakest of the three everywhere.

{{table:cost}}

![Cost of the three encoders, measured with each model alone on the GPU (median of three benchmark runs).](../figures/report/cost.png){#fig:cost width=100%}

**What the encoders cost** (@tbl:cost, @fig:cost). The scratch U-Net is the cheaper model on every count. It has {{cost_scratch_params}} million parameters against {{cost_default_params}} million, a faster training step ({{cost_scratch_step_ms}} ms against {{cost_default_step_ms}} ms), shorter runs ({{cost_scratch_minutes}} against {{cost_default_minutes}} minutes) and a faster prediction (about {{cost_scratch_infer_ms}} ms per chip). The two MobileNetV2 models are the same network and should cost the same; their training steps agree, while their measured prediction times ({{cost_default_infer_ms}} and {{cost_imagenet_infer_ms}} ms) differ only because that measurement is noisy, so prediction times are good to about a third. ImageNet weights cost nothing at run time, but the pretrained runs needed more epochs before early stopping ended them ({{cost_imagenet_epochs}} on average against {{cost_default_epochs}}). Memory is not a constraint for any of the models: about {{cost_scratch_gpu_train_mb}} to {{cost_default_gpu_train_mb}} MB of GPU memory in training, under {{cost_default_gpu_infer_mb_round}} MB at prediction, and {{cost_scratch_ram_gb}} to {{cost_default_ram_gb}} GB of host memory for a training process.

### An unseen flood: Bolivia (RQ4) {#sec:rq4}

**The lead over a simple threshold disappears on the held-out flood.** On Bolivia the Otsu threshold reaches {{otsu_bolivia_iou}}, far above its test score: the Bolivia chips hold proportionally more open water, and the threshold raises far fewer false alarms there (precision {{otsu_bolivia_precision}} against {{otsu_test_precision}}). The default U-Net reaches {{default_bolivia_iou_pm}}, {{default_bolivia_minus_test}} relative to its test score. A paired bootstrap over the 15 Bolivia chips shows that the two cannot be separated: for the three default runs the difference to Otsu is {{diff_R01_R00_bolivia}}, {{diff_R02_R00_bolivia}} and {{diff_R03_R00_bolivia}}, each interval includes zero, and the network wins in only {{diffwins_min_R00_bolivia}} to {{diffwins_max_R00_bolivia}} of the resamples. The mean per-chip IoU even favours Otsu ({{otsu_bolivia_miou}} against {{default_bolivia_miou_pm}}).

**What the networks lose on Bolivia is recall.** Precision stays high ({{default_bolivia_precision}} for the default), but recall drops from {{default_test_recall}} on the test chips to {{default_bolivia_recall}}. The tuned threshold shows the same pattern more strongly (recall {{tuned_bolivia_recall}}) and scores {{tuned_bolivia_iou}}, below Otsu. Bolivia's water is simply less dark: its median VH is {{water_median_vh_bolivia}} dB against {{water_median_vh_training}} dB in the training chips, and only {{water_darker_tuned_bolivia}} of it lies below the tuned threshold, although {{water_darker_otsu_bolivia}} lies below the more generous Otsu threshold (@fig:brightness). A rule or a model that has learned to be careful about moderately dark pixels, because in the training floods they were often land, will plausibly miss real water here.

![VH backscatter of the labelled water pixels in the training chips, the test chips and Bolivia, with the two thresholds. Much of Bolivia's water lies between the tuned and the Otsu threshold.](../figures/report/water_brightness.png){#fig:brightness width=70%}

**How well a model transfers depends on the configuration, and validation does not show it.** The change from test to Bolivia ranges from {{scratch_bolivia_minus_test}} (scratch U-Net) to {{imagenet_bolivia_minus_test}} (ImageNet weights), with the default at {{default_bolivia_minus_test}} and BCE at {{bce_bolivia_minus_test}}. The configuration with the best validation IoU, ImageNet weights, is the worst on Bolivia. Validation chips come from the same ten floods as the training chips, so they cannot reveal this.

### Where the best model fails {#sec:errors}

The error analysis uses the single model with the highest validation IoU, as the protocol demands: {{best_run}} ({{best_run_config}}, seed {{best_run_seed}}; test IoU {{R19_test_iou}}, Bolivia {{R19_bolivia_iou}}). Every wrong pixel was assigned to a category by fixed rules (@tbl:errors), and every chip with an IoU below 0.5 to a group (@tbl:failures). "Dark" means VH below the Otsu threshold.

{{table:errors}}

**Most errors on the test chips are missed water, and the largest part of it cannot be seen in the radar image.** {{err_test_unet_missed_all}} of the model's wrong test pixels are missed water. The largest single category, {{err_test_unet_miss_not_dark}} of all errors, is water that is not dark: in the radar image it looks like land. This is the price of labels that were drawn with the help of optical images. No radar-only model can recover such water, and it sets a ceiling on the achievable IoU. A further {{err_test_unet_miss_no_radar}} is water in places with no radar data at all. Genuine misses of dark water account for {{err_test_unet_miss_interior}}, thin rivers and small ponds for {{err_test_unet_miss_thin}}, and shorelines (missed and false together) for about a fifth.

**False alarms are the smaller problem, and they are what the network fixes.** For the Otsu threshold, {{err_test_otsu_alarm_dark_land}} of all wrong test pixels are dark land taken for water. For the U-Net this category shrinks to {{err_test_unet_alarm_dark_land}}, of a total that is itself almost three times smaller. The network has learned from context that a dark field is not a lake.

{{table:failures}}

**Chip by chip** (@tbl:failures, @fig:failures). {{n_failing}} chips have an IoU below 0.5. The largest group by far, {{n_failing_little_water}} chips, holds under 1% water: there, a few hundred missed or invented pixels are the whole score. Next come chips whose water is not dark in the radar image, and chips where dark land is taken for water. The model beats Otsu on {{chips_won_test}} of the {{chips_compared_test}} test chips on which both have a defined IoU, but on only {{chips_won_bolivia}} of the {{chips_compared_bolivia}} Bolivia chips.

![One example per failure group for the best-validation model: VH, the hand label, the Otsu threshold and the U-Net. Blue: water found; red: false alarm; yellow: missed water; grey: no data.](../figures/report/failure_examples.png){#fig:failures width=74%}

**On Bolivia the dominant error changes.** There, {{err_bolivia_unet_miss_interior}} of this model's wrong pixels are dark water inside larger water bodies that it simply did not mark, against {{err_test_unet_miss_interior}} on the test chips. This is not a limit of the sensor: the Otsu threshold finds that water. It is a failure to transfer.

{{bolivia_encoder_paragraph}}

![Pooled IoU per flood event for the two thresholds and the best-validation U-Net. The network improves on the thresholds in every training event, most where the thresholds drown in false alarms (USA, Somalia, Paraguay), and falls behind Otsu on the held-out flood.](../figures/report/per_event.png){#fig:events width=78%}

**By event** (@fig:events). Within the ten training floods the network is better than both thresholds everywhere, from {{event_pakistan_unet}} in Pakistan to {{event_nigeria_unet}} in Nigeria (in India only marginally). The gain is largest where a threshold produces mostly false alarms (USA: {{event_usa_otsu}} to {{event_usa_unet}}).

![Six test and Bolivia chips chosen by a fixed rule from the per-chip scores of the best-validation model: the two test chips with the lowest IoU among those with at least 2% water, three test chips at the 50th, 75th and 95th percentile, and the median Bolivia chip.](../figures/report/qualitative.png){#fig:qualitative width=74%}

@fig:qualitative shows the model's output next to Otsu on six chips chosen by rule.

### Comparison with published results {#sec:published}

{{table:published}}

@tbl:published places the default model next to published numbers for the same test chips. Three things can be said with some confidence.

- **A small network is in the same range as a much larger radar model.** The default U-Net ({{default_test_iou_pm}}) is within a point of the Attentive U-Net of Yadav et al. (0.672), which uses the same three inputs with a ResNet-50 encoder initialised from ImageNet and attention modules [@yadav2022attentive]. It is well above the radar-only classical learners of Iselborn et al. (about 0.47 to 0.50) [@iselborn2023importance] and the dataset's original baseline as quoted by Yadav et al. (0.493).
- **Radar alone stays well below optical input.** Models that read the Sentinel-2 bands of the same chips reach about 0.8 [@prithvi2023sen1floods11; @schmalstieg2026distillation]. The error analysis above gives one reason: a third of the errors of the radar model are water that is not visible in the radar image.
- **On Bolivia, simple and complex radar methods end up close together.** The linear radar model of Iselborn et al. is reported at 0.655, the Attentive U-Net at 0.66, the Otsu threshold here at {{otsu_bolivia_iou}} and the default U-Net at {{default_bolivia_iou}}. Only the model pre-trained on unlabelled radar imagery [@patel2021evaluating] is clearly higher (0.716).

The comparison is indicative only. The papers differ in training data (some use weak labels or unlabelled imagery), inputs, preprocessing and model selection; several report a single run where this work reports the mean of three; and not all state whether their IoU is pooled over pixels or averaged over chips, which on this dataset changes the number by up to a factor of two. The values for the dataset baseline and for three other models are quoted from the tables of later papers, as noted in `report/published_results.csv`.
