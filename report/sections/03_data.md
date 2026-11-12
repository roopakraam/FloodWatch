## Data {#sec:data}

### The hand-labelled subset of Sen1Floods11

All experiments use the {{n_chips_total}} hand-labelled chips of Sen1Floods11, version 1.1 [@bonafilia2020sen1floods11]. A chip is a 512×512-pixel tile of a Sentinel-1 scene at 10 m per pixel, about 5 km on a side, stored as a two-band image: the backscatter in decibels (dB) for the VV and the VH polarisation. Each chip comes with a mask of the same size in which every pixel is 1 (water), 0 (not water) or −1 (no data). The masks mark all surface water, permanent and flood water alike. For every event the dataset also holds a Sentinel-2 optical image taken within two days of the radar image, and the masks were drawn with the help of both [@bonafilia2020sen1floods11; @yadav2022attentive]. This matters later: water that is visible optically is not always dark in the radar image.

The chips come from eleven flood events between 2016 and 2019 (@tbl:events). The dataset's official split is used unchanged: {{n_train}} training, {{n_val}} validation and {{n_test}} test chips drawn from ten of the events, and the {{n_bolivia}} chips of the eleventh, Bolivia, held out completely. The test chips measure performance on new places within familiar floods; the Bolivia chips measure performance on a flood the model has never seen (RQ4). Before any modelling it was checked that all {{n_chips_total}} image and label files exist, that the four split lists cover every chip exactly once, and that no Bolivia chip appears in the other three.

{{table:events}}

### What the training data look like

All statistics that influenced a design decision were computed on the training chips only. @fig:samples shows six of them. Three properties of the data shape the rest of the project.

![Six training chips from six events: VV, VH and the hand label. Both radar bands use the same fixed dB range in every panel. Open water is dark, more cleanly so in VH than in VV. The masks contain thin rivers and ragged shorelines only a few pixels wide.](../figures/eda/sample_chips.png){#fig:samples width=100%}

**Water is rare.** Water covers {{train_water_share}} of the labelled training pixels. {{train_chips_no_water}} of the {{n_train}} training chips contain no water at all and {{train_chips_lt5}} contain less than 5%. A model that never predicts water would be about 90% accurate, so accuracy is not reported; and a loss that treats all pixels alike is dominated by the easy "not water" pixels, which is why the loss is one of the factors studied (RQ2).

**One number does not separate water from land.** @fig:hist shows the distribution of both bands for the two classes. Water is darker, but the distributions overlap: the shared area of the two normalised histograms is {{overlap_vh}} in VH and {{overlap_vv}} in VV. In VV the water distribution has a second hump that lies largely under the land distribution; the likely cause is water roughened by wind or mixed with vegetation, to which VV is sensitive. A threshold on VH is therefore a fair but limited baseline, and there is room for a model that looks at both bands and at the neighbourhood of a pixel (RQ1).

![Backscatter of the training pixels, water against not water, each class normalised to unit area. The classes overlap in both bands, less in VH than in VV.](../figures/eda/hist_vv_vh_water.png){#fig:hist width=92%}

**Part of the data is missing.** {{train_nodata_share}} of the training pixels have no label, and {{train_nan_share}} of the radar pixels have no value (NaN), nearly always where the label is missing too. Unlabelled pixels are excluded from the loss and from every metric. A small number of pixels have a label but no radar value; the largest such case is one test chip whose radar image is entirely empty although a fifth of it is labelled as water. No method can classify such pixels from the image. They are kept in the reported scores, following the usual convention that every labelled pixel counts, and @sec:results also gives the scores without them.

@tbl:splits summarises the four splits. The water share differs between them, and between events it differs far more, so a score pooled over all pixels and a score averaged over chips will not agree; both are reported.

{{table:splits}}

### Preprocessing

Every chip goes through the same function, in training, in evaluation and in the inference demo:

1. Both bands are clipped to [−35, 5] dB. This touches {{clipped_share}} of the training values, mostly very dark water in VH.
2. A third channel is added, the VV/VH ratio. On the logarithmic dB scale a ratio is a difference, so the channel is VV − VH, computed from the clipped values.
3. Each channel is standardised with the mean and standard deviation of the training chips (VV {{norm_mean_vv}} ± {{norm_std_vv}} dB, VH {{norm_mean_vh}} ± {{norm_std_vh}} dB, ratio {{norm_mean_ratio}} ± {{norm_std_ratio}} dB). The validation and test chips play no part in these statistics.
4. Missing radar values are set to 0, which after standardising is the average training pixel.

Depending on the configuration the model receives VH alone, VV and VH, or all three channels.
