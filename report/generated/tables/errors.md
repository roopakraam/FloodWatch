Table: Where the wrong pixels are: share of all wrong pixels of a method on a split, by category. "Dark" means VH below the Otsu threshold. {#tbl:errors}

| Kind of error | Test: U-Net R19 | Test: Otsu | Bolivia: U-Net R19 | Bolivia: Otsu |
|:---|---:|---:|---:|---:|
| missed water: no radar data | 5% | 2% | 1% | 1% |
| missed water: not dark in the radar image | 35% | 16% | 20% | 29% |
| missed water: thin or small water body | 5% | 0% | 3% | 0% |
| missed water: shoreline | 4% | 0% | 5% | 0% |
| missed water: dark, inside a larger water body | 15% | 0% | 44% | 0% |
| false alarm: shoreline | 14% | 6% | 11% | 16% |
| false alarm: dark land | 18% | 76% | 12% | 53% |
| false alarm: other | 3% | 0% | 4% | 0% |
