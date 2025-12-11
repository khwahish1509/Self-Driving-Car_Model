import os
import argparse
import pandas as pd
import numpy as np


# -----------------------------------------------------------
# Merge all driving_log.csv files from data/raw folders
# -----------------------------------------------------------
def merge_csvs(raw_dir, out_csv):
    rows = []
    for root, dirs, files in os.walk(raw_dir):
        for f in files:
            if f.endswith(".csv"):
                p = os.path.join(root, f)
                df = pd.read_csv(p, header=None)

                # Udacity CSV: center, left, right, steering, throttle, brake, speed
                for _, r in df.iterrows():
                    center_img_path = os.path.join(root, "IMG", os.path.basename(str(r[0]).strip()))
                    steering = float(r[3])

                    rows.append([center_img_path, steering])

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False, header=False)
    print(f"[OK] merged CSV saved to: {out_csv}")


# -----------------------------------------------------------
# Balance steering distribution
# -----------------------------------------------------------
def balance_csv(in_csv, out_csv, bins=31, max_per_bin=None):
    df = pd.read_csv(in_csv, header=None)
    angles = df[1].values

    hist, edges = np.histogram(angles, bins=bins)

    if max_per_bin is None:
        max_per_bin = int(np.mean(hist))

    keep_indices = []

    for i in range(bins):
        bin_left = edges[i]
        bin_right = edges[i+1]

        inds = np.where((angles >= bin_left) & (angles < bin_right))[0]

        if len(inds) > max_per_bin:
            inds = np.random.choice(inds, max_per_bin, replace=False)

        keep_indices.extend(list(inds))

    balanced = df.iloc[keep_indices]
    balanced.to_csv(out_csv, index=False, header=False)
    print(f"[OK] balanced CSV saved to: {out_csv}")


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge_csvs", default="")
    parser.add_argument("--out", default="data/processed/merged.csv")
    parser.add_argument("--balance", action="store_true")
    args = parser.parse_args()

    if args.merge_csvs:
        merge_csvs(args.merge_csvs, args.out)

    if args.balance:
        balanced_path = args.out.replace(".csv", "_balanced.csv")
        balance_csv(args.out, balanced_path)
