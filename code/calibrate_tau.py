"""Reliability threshold tau: maximum F1 for tumour overlap over the development candidates.

Every distinct development mean variance is tried as a threshold (HIGH if <= t); ties go to
the smallest value. The development predictions are in-sample (the network was trained on
these cases). Validation candidates are only scored with the frozen tau from parameters.yaml.

Usage: python code/calibrate_tau.py              # from the NIfTI outputs of candidates.py
       python code/calibrate_tau.py --from-csv   # from results/candidate_table_all.csv
"""
import argparse

import numpy as np
import pandas as pd

from candidates import candidate_rows
from common import REPO, load_params, load_split


def f1_max_threshold(mean_var, overlap):
    best_t, best_f1 = None, -1.0
    for t in np.unique(mean_var):                     # ascending, so ties keep the smallest t
        high = mean_var <= t
        tp, fp = np.sum(high & overlap), np.sum(high & ~overlap)
        fn = np.sum(~high & overlap)
        f1 = 2 * tp / (2 * tp + fp + fn)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1


def f1_curve(mean_var, overlap):
    """Development F1 at every distinct mean variance used as threshold."""
    ts = np.unique(mean_var)
    return ts, np.array([2 * np.sum((mean_var <= t) & overlap) / (np.sum(mean_var <= t) + np.sum(overlap))
                         for t in ts])


def confusion(df, tau):
    high, ov = df.mean_variance.values <= tau, df.gt_overlap.values == 1
    return dict(TP=int(np.sum(high & ov)), FP=int(np.sum(high & ~ov)),
                TN=int(np.sum(~high & ~ov)), FN=int(np.sum(~high & ov)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-csv", action="store_true")
    args = ap.parse_args()
    prm = load_params()
    train, val = load_split()

    if args.from_csv:
        df = pd.read_csv(REPO / prm["paths"]["results_dir"] / "candidate_table_all.csv")
    else:
        df = pd.DataFrame([r for split, cases in (("development", train), ("validation", val))
                           for c in cases for r in candidate_rows(prm, c, split)])
    dev, vld = df[df.split == "development"], df[df.split == "validation"]

    t, f1 = f1_max_threshold(dev.mean_variance.values, dev.gt_overlap.values == 1)
    tau = prm["stage2"]["tau"]
    print(f"Development: {len(dev)} candidates from {dev.case_id.nunique()} cases (in-sample)")
    print(f"Exact F1 optimum: t = {t:.10f} (development F1 {f1:.3f})")
    ts, curve = f1_curve(dev.mean_variance.values, dev.gt_overlap.values == 1)
    near = ts[curve >= f1 - 0.02]
    print(f"Development F1 within 0.02 of the maximum for t = {near.min():.4f} to {near.max():.4f}")
    c = confusion(dev, tau)
    print(f"Frozen tau applied in the paper: {tau} "
          f"(development F1 at tau: {2 * c['TP'] / (2 * c['TP'] + c['FP'] + c['FN']):.3f})")
    print(f"Development at tau: {c}; {c['TP']} of {c['TP'] + c['FP']} HIGH candidates overlap tumour "
          f"(precision {c['TP'] / (c['TP'] + c['FP']):.2f})")
    print(f"Validation at tau ({len(vld)} candidates): {confusion(vld, tau)}")


if __name__ == "__main__":
    main()
