Table: Chips on which the best U-Net fails (IoU below 0.5), grouped by a fixed rule: under 1% water, otherwise the category holding most of the chip's wrong pixels. {#tbl:failures}

| Failure group | Test chips | Bolivia chips | Mean IoU, U-Net | Mean IoU, Otsu, same chips | Example chip | Its IoU |
|:---|---:|---:|---:|---:|:---|---:|
| little or no water (under 1% of the chip) | 23 | 3 | 0.100 | 0.022 | Ghana 97059 | 0.000 |
| water that is not dark in the radar image | 14 | 2 | 0.226 | 0.196 | Sri-Lanka 450918 | 0.001 |
| dark land taken for water | 7 | 1 | 0.318 | 0.207 | Sri-Lanka 117737 | 0.097 |
| dark water missed | 4 | 2 | 0.157 | 0.351 | Pakistan 70625 | 0.000 |
| edges of water bodies | 5 | 0 | 0.444 | 0.212 | USA 778194 | 0.346 |
| no radar data | 1 | 1 | 0.007 | 0.015 | Paraguay 34417 | 0.000 |
