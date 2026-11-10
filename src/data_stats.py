"""Training-set statistics: class balance, backscatter histograms, data quality.

Run from the project root:  python -m src.data_stats
Writes results/data_stats.csv and figures/eda/hist_vv_vh_water.png.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import NO_DATA, NON_WATER, ROOT, WATER, list_split, load_chip

SPLIT = "train"
LOW_WATER_FRAC = 0.01  # "almost no water" = < 1% of valid pixels
LOW_WATER_EXTRA = (0.0, 0.05)  # extra thresholds reported for context
BINS = np.arange(-35, 6, 1.0)  # dB, 1 dB wide
BANDS = ("VV", "VH")
CLASSES = {"water": WATER, "non_water": NON_WATER}

OUT_CSV = ROOT / "results" / "data_stats.csv"
OUT_FIG = ROOT / "figures" / "eda" / "hist_vv_vh_water.png"


def main():
    pairs = list_split(SPLIT)
    hist = {(b, c): np.zeros(len(BINS) - 1, dtype=np.int64) for b in BANDS for c in CLASSES}
    rows = []

    for s1_path, label_path in pairs:
        image, label = load_chip(s1_path, label_path, nan_value=None)
        valid = label != NO_DATA
        is_water = label == WATER
        n_valid = int(valid.sum())
        n_water = int(is_water.sum())
        nan_per_band = np.isnan(image).sum(axis=(1, 2))

        rows.append({
            "section": "chip",
            "name": s1_path.name.replace("_S1Hand.tif", ""),
            "country": s1_path.name.split("_")[0],
            "n_valid": n_valid,
            "n_water": n_water,
            "water_frac": n_water / n_valid if n_valid else np.nan,
            "n_label_nodata": int((~valid).sum()),
            "n_nan_vv": int(nan_per_band[0]),
            "n_nan_vh": int(nan_per_band[1]),
        })

        for bi, band in enumerate(BANDS):
            ok = valid & ~np.isnan(image[bi])
            for cname, cval in CLASSES.items():
                vals = image[bi][ok & (label == cval)]
                hist[(band, cname)] += np.histogram(vals, bins=BINS)[0]

    chips = pd.DataFrame(rows)

    # --- overall + low-water summary rows (pooled, not mean of per-chip fractions)
    tot_valid, tot_water = chips["n_valid"].sum(), chips["n_water"].sum()
    overall = [
        {"section": "overall", "name": "n_chips", "value": len(chips)},
        {"section": "overall", "name": "water_share_of_valid_px", "value": tot_water / tot_valid},
        {"section": "overall", "name": "n_valid_px", "value": tot_valid},
        {"section": "overall", "name": "n_water_px", "value": tot_water},
        {"section": "overall", "name": "n_label_nodata_px", "value": chips["n_label_nodata"].sum()},
        {"section": "overall", "name": "n_nan_px_vv", "value": chips["n_nan_vv"].sum()},
        {"section": "overall", "name": "n_nan_px_vh", "value": chips["n_nan_vh"].sum()},
    ]
    for thr in sorted({LOW_WATER_FRAC, *LOW_WATER_EXTRA}):
        n = int((chips["water_frac"] <= thr).sum() if thr == 0 else (chips["water_frac"] < thr).sum())
        op = "==" if thr == 0 else "<"
        overall.append({"section": "low_water", "name": f"n_chips_water_frac{op}{thr}", "value": n})

    hist_rows = [
        {"section": "hist", "band": band, "class": cname,
         "bin_left": BINS[i], "bin_right": BINS[i + 1], "count": int(counts[i])}
        for (band, cname), counts in hist.items() for i in range(len(counts))
    ]

    out = pd.concat([chips, pd.DataFrame(overall), pd.DataFrame(hist_rows)], ignore_index=True)
    OUT_CSV.parent.mkdir(exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    # --- histogram figure (density-normalised: classes are heavily imbalanced)
    centers = (BINS[:-1] + BINS[1:]) / 2
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, band in zip(axes, BANDS):
        for cname, color in (("water", "royalblue"), ("non_water", "saddlebrown")):
            counts = hist[(band, cname)]
            ax.plot(centers, counts / counts.sum(), color=color, label=cname.replace("_", "-"))
        ax.set_title(f"{band} backscatter ({SPLIT} chips)")
        ax.set_xlabel("dB")
        ax.legend()
    axes[0].set_ylabel("fraction of pixels in class")
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=120)

    print(f"{len(chips)} chips | water share {tot_water / tot_valid:.2%}")
    print(pd.DataFrame(overall).to_string(index=False))
    print("wrote", OUT_CSV, "and", OUT_FIG)


if __name__ == "__main__":
    main()
