"""Validation results of the paper (20 cases).

  Table 1   candidate-level reliability with exact (Clopper-Pearson) 95% intervals
  Overlap   robustness of the any-voxel overlap criterion
  Table 2   outcomes per reference lesion (hits, misses) and per candidate (false positives),
            with local Dice per reliability group (exact Mann-Whitney, HIGH vs LOW)
  Accepted  mean Dice in the cases with a HIGH candidate: raw, Stage 1, Stage 2 (needs the images)
  AUC       tumour overlap: -mean variance vs maximum probability (paired case-level bootstrap),
            saturation of maximum probability, the comparison at the flag's operating point and the
            risk-coverage curves over all thresholds (AURC)
  Size      variance within size halves and against candidate volume (development and validation)

Usage: python code/evaluate.py              # from the NIfTI outputs; also rewrites results/*.csv
       python code/evaluate.py --from-csv   # everything except "Accepted", from results/*.csv only
"""
import argparse

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import beta, mannwhitneyu
from sklearn.metrics import roc_auc_score

from candidates import candidate_rows
from common import REPO, case_file, gt_path, load_params, load_split, read


def exact_ci(k, n):
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return 100 * lo, 100 * hi


def table1(v):
    high, ov = v.confidence == "HIGH", v.gt_overlap == 1
    rows = [("HIGH coverage of candidates", high.sum(), len(v)),
            ("Overlap, HIGH candidates", (high & ov).sum(), high.sum()),
            ("Overlap, LOW candidates", (~high & ov).sum(), (~high).sum()),
            ("Overlapping labelled HIGH", (high & ov).sum(), ov.sum()),
            ("Non-overlapping labelled LOW", (~high & ~ov).sum(), (~ov).sum())]
    print("\nTable 1 (validation candidates, exact 95% CI)")
    for name, k, n in rows:
        lo, hi = exact_ci(int(k), int(n))
        print(f"  {name:30s} {k}/{n} ({100 * k / n:.1f}%)  CI {lo:.1f}-{hi:.1f}")


def overlap_robustness(v, min_purity=0.25):
    o = v[v.gt_overlap == 1]
    same = ((v.purity >= min_purity).astype(int) == v.gt_overlap).all()
    print(f"\nOverlap criterion: smallest overlap {o.overlap_voxels.min()} voxels; Table 1 "
          f"{'unchanged' if same else 'CHANGED'} if >= {min_purity:.0%} of a candidate's voxels "
          f"must lie in the tumour (lowest purity {o.purity.min():.2f})")


def case_rows(prm, case, raw_thr, lesion_conn):
    """(lesion, candidate) pairs in which any candidate voxel lies in the lesion, and one row per lesion."""
    s1 = np.asarray(read(case_file(prm, case, "stage1"))).astype(np.int32)
    s2 = read(case_file(prm, case, "stage2"))
    raw = read(case_file(prm, case, "mean_prob")) >= raw_thr
    gt = read(gt_path(prm, case)) > 0
    lesions, n = ndi.label(gt, structure=ndi.generate_binary_structure(3, lesion_conn))
    pairs, lesion_list = [], []
    for k in range(1, n + 1):
        L = lesions == k
        ids, cnt = np.unique(s1[L], return_counts=True)
        dice = {"HIGH": [], "LOW": []}
        for b, ov in zip(ids, cnt):
            if b == 0:
                continue
            B = (s1 == b) & raw                                    # candidate's p >= 0.5 voxels
            inter = int((B & L).sum())
            conf = "HIGH" if int(s2[s1 == b][0]) == 1 else "LOW"
            d = 2 * inter / (B.sum() + L.sum())
            dice[conf].append(d)
            pairs.append(dict(case_id=case, lesion_id=k, blob_id=int(b), confidence=conf,
                              lesion_voxels=int(L.sum()), blob_voxels_p05=int(B.sum()),
                              overlap_any_voxels=int(ov), overlap_p05_voxels=inter,
                              dice=d, coverage=inter / L.sum(),
                              precision=inter / B.sum() if B.sum() else np.nan))
        nh, nl = len(dice["HIGH"]), len(dice["LOW"])
        status = "both" if nh and nl else "HIGH_only" if nh else "LOW_only" if nl else "neither"
        raw_hit = bool((raw & L).any())                            # seen at p >= 0.5 but no candidate: gated out
        lesion_list.append(dict(case_id=case, lesion_id=k, lesion_voxels=int(L.sum()), status=status,
                                neither_type=("gated_out" if raw_hit else "undetected") if status == "neither" else "",
                                n_high_blobs=nh, n_low_blobs=nl,
                                best_high_dice=max(dice["HIGH"]) if nh else np.nan,
                                best_low_dice=max(dice["LOW"]) if nl else np.nan,
                                raw_p05_overlaps=raw_hit))
    return pairs, lesion_list


def table2(v, pairs, lesions, min_voxels):
    les = lesions[lesions.lesion_voxels >= min_voxels]
    p = pairs[pairs.lesion_voxels >= min_voxels]
    fp = v[v.gt_overlap == 0]
    print(f"\nTable 2 (outcomes; {len(les)} lesions >= {min_voxels} voxels, {len(v)} candidates)")
    for conf in ("HIGH", "LOW"):
        hit = les[les.status == f"{conf}_only"]
        d = p[p.confidence == conf].dice
        print(f"  {conf + ' on tumour':24s}{len(hit)} lesions in {hit.case_id.nunique()} cases; "
              f"local Dice median {d.median():.2f} ({d.min():.2f}-{d.max():.2f})")
    for conf in ("HIGH", "LOW"):
        f = fp[fp.confidence == conf]
        print(f"  {conf + ' false positive':24s}{len(f)} candidates in {f.case_id.nunique()} cases")
    for kind, name in (("undetected", "no candidate"), ("gated_out", "removed by gate")):
        m = les[(les.status == "neither") & (les.neither_type == kind)]
        print(f"  {'Missed, ' + name:24s}{len(m)} lesions {m.case_id.tolist()}")
    if (les.status == "both").any():
        print(f"  Lesions with both HIGH and LOW candidates: {(les.status == 'both').sum()}")
    h, l = p[p.confidence == "HIGH"], p[p.confidence == "LOW"]
    pv = mannwhitneyu(h.dice, l.dice, alternative="two-sided", method="exact").pvalue
    print(f"  local Dice HIGH vs LOW: exact two-sided Mann-Whitney p = {pv:.3f} (exploratory); "
          f"{len(p)} pairs from {p.case_id.nunique()} cases")
    print(f"  median lesion coverage HIGH {h.coverage.median():.2f}, LOW {l.coverage.median():.2f}")
    print(f"  false positives with maximum probability >= 0.90: {(fp.max_prob >= 0.90).sum()} of {len(fp)}")


def accepted_dice(prm, val, cand, raw_thr):
    rows = []
    for case in val:
        high_ids = cand[(cand.case_id == case) & (cand.confidence == "HIGH")].blob_id.tolist()
        if not high_ids:
            continue
        s1 = np.asarray(read(case_file(prm, case, "stage1"))).astype(np.int32)
        raw = read(case_file(prm, case, "mean_prob")) >= raw_thr
        gt = read(gt_path(prm, case)) > 0
        for stage, pred in (("Raw", raw), ("Stage 1", raw & (s1 > 0)), ("Stage 2", raw & np.isin(s1, high_ids))):
            tp = int((pred & gt).sum())
            rows.append(dict(case_id=case, stage=stage, dice=2 * tp / (pred.sum() + gt.sum()), recall=tp / gt.sum()))
    df = pd.DataFrame(rows)
    print(f"\nAccepted cases ({df.case_id.nunique()} validation cases with a HIGH candidate)")
    for stage, s in df.groupby("stage", sort=False):
        print(f"  {stage:8s} mean Dice {s.dice.mean():.3f}  mean recall {s.recall.mean():.3f}")


def auc(d, score):
    return roc_auc_score(d.gt_overlap, score(d))


VAR, MAXP, VOL = (lambda d: -d.mean_variance), (lambda d: d.max_prob), (lambda d: d.log_volume)


def case_bootstrap(d, stat, n_boot, seed):
    rng = np.random.default_rng(seed)
    cases = d.case_id.unique()
    groups = {c: d[d.case_id == c] for c in cases}
    out = []
    for _ in range(n_boot):                                      # resample cases, keep their candidates
        b = pd.concat([groups[c] for c in rng.choice(cases, len(cases))], ignore_index=True)
        if b.gt_overlap.nunique() == 2:
            out.append(stat(b))
    return np.mean(out), *np.percentile(out, [2.5, 97.5]), len(out)


def risk_coverage(v, score):
    """False-positive fraction among the k most reliable candidates (highest score first), k = 1..n."""
    fp = (v.gt_overlap.values == 0)[np.argsort(-score(v).values, kind="stable")]
    return fp, np.cumsum(fp) / np.arange(1, len(fp) + 1)


def auc_comparison(v, n_boot, seed):
    m, lo, hi, n = case_bootstrap(v, lambda b: auc(b, VAR) - auc(b, MAXP), n_boot, seed)
    print(f"\nAUC for tumour overlap: variance {auc(v, VAR):.3f}, maximum probability {auc(v, MAXP):.3f}")
    print(f"  paired case-level bootstrap of the difference ({n} resamples): "
          f"mean {m:+.3f}, 95% CI {lo:+.3f} to {hi:+.3f}")
    fp = v.gt_overlap == 0
    print(f"  maximum probability >= 0.99: {(v.max_prob >= 0.99).sum()} of {len(v)} candidates, "
          f"{(fp & (v.max_prob >= 0.99)).sum()} of {fp.sum()} false positives")
    k = int((v.confidence == "HIGH").sum())
    by_var = int((v.nsmallest(k, "mean_variance").gt_overlap == 0).sum())
    by_max = int((v.nlargest(k, "max_prob").gt_overlap == 0).sum())
    print(f"  at the operating point ({k} accepted candidates): false positives "
          f"{by_var} by variance, {by_max} by maximum probability")
    (fp_var, r_var), (fp_max, r_max) = risk_coverage(v, VAR), risk_coverage(v, MAXP)
    perfect = np.cumsum(np.sort(fp_var)) / np.arange(1, len(fp_var) + 1)
    first = lambda f: int(np.argmax(f)) if f.any() else len(f)     # candidates accepted before the first FP
    print(f"  risk-coverage over all thresholds: AURC {r_var.mean():.3f} for variance, {r_max.mean():.3f} for "
          f"maximum probability (perfect ranking {perfect.mean():.3f}); accepted before the first false "
          f"positive: {first(fp_var)} by variance, {first(fp_max)} by maximum probability")


def size_check(cand, n_boot, seed):
    print("\nVariance and candidate size (log volume)")
    for split in ("development", "validation"):
        d = cand[cand.split == split]
        small = d.voxel_count <= d.voxel_count.median()
        m, lo, hi, _ = case_bootstrap(d, lambda b: auc(b, VAR) - auc(b, VOL), n_boot, seed)
        print(f"  {split}: AUC variance {auc(d, VAR):.3f}, volume {auc(d, VOL):.3f}; variance within the "
              f"smaller and larger halves {auc(d[small], VAR):.3f} (n={small.sum()}) and "
              f"{auc(d[~small], VAR):.3f} (n={(~small).sum()}); variance minus volume "
              f"{m:+.3f} (95% CI {lo:+.3f} to {hi:+.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-csv", action="store_true")
    args = ap.parse_args()
    prm = load_params()
    ev = prm["evaluation"]
    train, val = load_split()
    out = REPO / prm["paths"]["results_dir"]

    if args.from_csv:
        cand = pd.read_csv(out / "candidate_table_all.csv")
        pairs = pd.read_csv(out / "pair_table.csv")
        lesions = pd.read_csv(out / "lesion_table.csv")
    else:
        cand = pd.DataFrame([r for split, cases in (("development", train), ("validation", val))
                             for c in cases for r in candidate_rows(prm, c, split)])
        per_case = [case_rows(prm, c, ev["raw_threshold"], ev["lesion_connectivity"]) for c in val]
        pairs = pd.DataFrame([r for p, _ in per_case for r in p])
        lesions = pd.DataFrame([r for _, les in per_case for r in les])
        out.mkdir(parents=True, exist_ok=True)
        cand.to_csv(out / "candidate_table_all.csv", index=False)
        pairs.to_csv(out / "pair_table.csv", index=False)
        lesions.to_csv(out / "lesion_table.csv", index=False)

    v = cand[cand.split == "validation"]
    print(f"Validation: {len(v)} candidates in {v.case_id.nunique()} of {len(val)} cases")
    table1(v)
    overlap_robustness(v)
    table2(v, pairs, lesions, ev["min_lesion_voxels"])
    if not args.from_csv:
        accepted_dice(prm, val, cand, ev["raw_threshold"])
    auc_comparison(v, ev["n_bootstrap"], ev["seed"])
    size_check(cand, ev["n_bootstrap"], ev["seed"])


if __name__ == "__main__":
    main()
