Table: Every run. The same table, with more columns, is results/all_runs.csv. {#tbl:allruns}

| Run | Configuration | Seed | Epochs | Best epoch | Val IoU | Test IoU | Test precision | Test recall | Bolivia IoU | s / epoch |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| R00 | Otsu (E0) | – | – | – | 0.388 | 0.433 | 0.483 | 0.809 | 0.653 | – |
| R01 | Default | 0 | 39 | 24 | 0.628 | 0.668 | 0.852 | 0.756 | 0.655 | 8.6 |
| R02 | Default | 1 | 32 | 17 | 0.645 | 0.664 | 0.844 | 0.757 | 0.621 | 8.4 |
| R03 | Default | 2 | 34 | 19 | 0.643 | 0.667 | 0.813 | 0.788 | 0.635 | 8.6 |
| R00b | Tuned VH threshold | – | – | – | 0.492 | 0.524 | 0.747 | 0.637 | 0.531 | – |
