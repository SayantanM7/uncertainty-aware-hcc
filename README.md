# Knowing When Not to Trust: Uncertainty-Based Abstention for Hepatocellular Carcinoma Candidates in CT

Code and results for a candidate-level reliability flag for automatically generated
hepatocellular carcinoma (HCC) candidates in contrast-enhanced CT, based on Monte Carlo (MC) Dropout.

**Authors:** Sayantan Mandal (Independent Researcher, Kolkata, India and ex-student of VNIT Nagpur, Maharashtra, India) and
Ashish Tiwari (VNIT Nagpur, India).
The accompanying paper is currently under review.

> Research prototype. HIGH/LOW denotes **model reliability, not HCC probability**.
> The framework does not diagnose HCC or select treatment.

## Overview

Aggregate segmentation accuracy does not show whether an individual predicted tumour can be trusted.
This framework asks, for each automatically generated candidate, whether it is reliable enough for
downstream interpretation, and abstains when it is not.

1. **Candidate generation (Stage 1).** An nnU-Net with Dropout3d produces mean probability and
   variance maps from 30 stochastic passes; candidates are connected components of the mean map.
2. **Reliability (Stage 2).** Each candidate is labelled **HIGH** or **LOW** by its mean variance.
   LOW candidates are an **abstention**: they stay visible for review and are not treated as non-tumour.
3. **Spatial concern (HIGH only).** An expert-anchored Spatial Concern Score (SCS) combines local
   uncertainty with portal-vein proximity. LOW candidates receive no score.
4. **Interpretable prototype.** A Tkinter interface reports, for a queried voxel, the reliability
   label, local variance, portal-vein distance and, for HIGH candidates, the SCS with an explanation.

Pipeline: LiTS pretraining → MC-Dropout fine-tuning → 30-pass inference →
Stage 1 candidates → Stage 2 HIGH/LOW → evaluation → SCS → prototype.

## Method

| Step | Setting |
|---|---|
| Data | HCC-TACE-Seg, 100 cases, binary tumour label; nnU-Net fold 0: 80 development, 20 validation cases |
| Network | nnU-Net v2, 3D full resolution, 128³ patches, 2.5 × 0.78 × 0.78 mm |
| Training | Pretrained on LiTS (binary tumour label, no dropout); fine-tuned on the 80 development cases with Dropout3d (p = 0.2), configuration `3d_fullres_mcdropout`; checkpoint epoch 295 (nnU-Net's best checkpoint by smoothed validation pseudo-Dice) |
| Inference | 30 passes with all 22 Dropout3d layers active, mirroring and Gaussian blending; mean probability p̄ and population variance U |
| Stage 1 | p̄ ≥ 0.10, 6-connected components, keep a component if max p̄ ≥ 0.80 |
| Stage 2 | HIGH if the candidate's mean variance ≤ τ = 0.01929155, otherwise LOW; τ maximised F1 for tumour overlap over the 132 in-sample development candidates and was frozen before validation |
| SCS (HIGH only) | U_N = U / (U + 0.05); P = 1 for D ≤ 5 mm, else exp(−(D − 5)/5); R = 1 + 6.5P + 4U_N − 2PU_N, and R = 10 for D ≤ 5 mm |

D is the 3D distance (mm) from the queried voxel to the expert portal-vein annotation provided with
HCC-TACE-Seg (not predicted by the network). An interventional radiologist rated eight scenarios
(HIGH/LOW confidence × sharp/fuzzy uncertainty × far/near portal vein); the SCS coefficients reproduce
the four HIGH-candidate ratings, and proximity dominates uncertainty over the whole surface
(∂R/∂P ≥ 4.5 > ∂R/∂U_N), as the expert ranked. LOW candidates receive no score, because ablation would
not be planned on a low-confidence lesion. The score is an interpretable prioritisation measure,
**not a clinically validated risk or safety score**.

## Data preparation (summary)

HCC-TACE-Seg CT series and their DICOM SEG annotations were converted to NIfTI (stored CT values,
voxel spacing from the DICOM headers). 100 of the 105 patients were used; HCC_006, 041 and 089 were
excluded for CT/SEG slice-count mismatch and HCC_048 and 065 for incomplete DICOM metadata. Tumour
and portal-vein masks come from the SEG annotations. For pretraining, LiTS labels were binarised
(tumour = 1, liver merged into background; 118 cases). Conversion scripts are available on request.

## Results (20 validation cases, 31 candidates, 22 lesions)

| Outcome | Count | Local Dice, median (range) |
|---|---|---|
| HIGH on tumour | 14 lesions (14 cases) | 0.82 (0.55–0.92) |
| LOW on tumour | 4 lesions (4 cases) | 0.62 (0.22–0.71) |
| HIGH false positive | **0 candidates** | — |
| LOW false positive | 13 candidates (8 cases) | — |
| Missed, no candidate | 3 lesions | — |
| Missed, removed by the 0.80 gate | 1 lesion | — |

| Measure | Result |
|---|---|
| HIGH candidates overlapping tumour | 14 / 14 (100%; exact 95% CI 76.8–100%) |
| Non-overlapping candidates labelled LOW | 13 / 13; 11 of them had maximum probability ≥ 0.90 |
| Local Dice, HIGH vs LOW | exact two-sided Mann–Whitney p = 0.008 (exploratory; one lesion–candidate pair per case) |
| Mean Dice in the 14 cases with a HIGH candidate | 0.738 (raw) → 0.746 (Stage 1) → 0.801 (Stage 2) |
| Candidate-overlap AUC | variance 0.876 vs maximum probability 0.846 (paired case-level bootstrap of the difference: +0.029, 95% CI −0.064 to 0.118) |
| At the flag's operating point (14 accepted candidates) | variance accepts 0 false positives; maximum probability would accept 2 |

A candidate overlaps tumour if any voxel intersects the reference; every overlap shares at least
3,864 voxels, and the results are unchanged if at least 25% of a candidate's voxels must lie within
the tumour. Five cases had only LOW candidates and were deferred for review; HCC_060 produced no
candidate. Because HIGH and LOW candidates carry different reliability by design, Dice is reported
per group rather than pooled. Variance is treated as a complementary reliability signal to maximum
probability. Results are preliminary (small internal cohort); external validation is future work.

`results/lesion_table.csv` lists every validation reference component; components below 100 voxels
(three in total) are excluded from the 22 lesions, as in the paper.

## Repository structure

```
uncertainty-aware-hcc/
├── README.md
├── LICENSE
├── requirements.txt
├── config/
│   ├── split.json            # nnU-Net fold-0 split (80 development / 20 validation)
│   ├── nnunet_plans.json     # plans incl. 3d_fullres_mcdropout (Dropout3d, p = 0.2)
│   └── parameters.yaml       # all thresholds and SCS constants used in the paper
├── training/
│   ├── README.md             # exact LiTS pretraining and HCC fine-tuning commands
│   └── prepare_fine_tuning.py
├── code/
│   ├── common.py             # parameters, split, paths, NIfTI I/O
│   ├── mc_dropout.py         # 30 passes -> mean_prob.nii.gz, var_map.nii.gz
│   ├── candidates.py         # Stage 1 and Stage 2
│   ├── calibrate_tau.py      # F1-maximising threshold on development candidates
│   ├── evaluate.py           # Table 1, local Dice, Mann–Whitney, AUCs, paired bootstrap
│   └── scs.py                # portal-vein distance map, U_N, P(D), SCS
├── prototype/
│   └── gui.py                # Tkinter interface
└── results/
    ├── candidate_table_all.csv   # all 163 candidates (132 development, 31 validation)
    ├── pair_table.csv            # validation lesion–candidate pairs (local Dice)
    └── lesion_table.csv          # all validation reference components; the paper counts the 22 with ≥ 100 voxels
```

## Installation

```bash
git clone https://github.com/SayantanM7/uncertainty-aware-hcc.git
cd uncertainty-aware-hcc
pip install -r requirements.txt
```

Edit the paths at the top of `config/parameters.yaml`; all other settings are those used in the paper.

## Quick check without images

```bash
cd code
python evaluate.py --from-csv
python calibrate_tau.py --from-csv
```

These recompute Table 1, the local Dice comparison, the AUCs with the paired bootstrap and τ from
the CSV files in `results/`; no images or GPU are needed.

## Full pipeline

1. **Data.** Download HCC-TACE-Seg from The Cancer Imaging Archive
   (doi:10.7937/TCIA.5FNA-0924) and LiTS. Images and masks are not redistributed here.
   Convert them to nnU-Net format as described in *Data preparation* (images
   `imagesTr/<case>_0000.nii.gz`, binary tumour labels `labelsTr/<case>.nii.gz`).
2. **Training.** Follow `training/README.md` (LiTS pretraining, plan alignment, fine-tuning with
   `3d_fullres_mcdropout`).
3. **Inference.** `python code/mc_dropout.py` writes `mean_prob.nii.gz` and `var_map.nii.gz` per case.
4. **Candidates and reliability.** `python code/candidates.py` writes Stage-1 and Stage-2 label maps;
   `python code/calibrate_tau.py` reproduces τ from the development candidates.
5. **Evaluation.** `python code/evaluate.py` reproduces the validation results above and rewrites
   `candidate_table_all.csv` and `pair_table.csv` in `results/`.
6. **Spatial concern.** Place each case's portal-vein annotation in its work folder as
   `portal_vein_mask.nii.gz` (same geometry as the CT); `python code/scs.py` writes the distance maps.
7. **Prototype.** `python prototype/gui.py`; enter a case ID and a voxel index (i, j, k) read from 3D Slicer.

## Reproducibility notes

- The trained weights (epoch-295 `checkpoint_best.pth`) are not included in this repository;
  `training/README.md` gives the exact commands that produced them.
- MC-Dropout inference is stochastic and no seed was fixed, so repeated runs may differ slightly
  in the variance maps; the values reported in the paper are in `results/`.
- The paired bootstrap uses 5000 case-level resamples with seed 0 (`config/parameters.yaml`).
- τ is stored as applied in the paper (0.01929155); `calibrate_tau.py` prints both the exact
  F1 optimum and the applied value.
- Inference was run with nnU-Net v2 (nnunetv2 2.8.1) and PyTorch 2.7.1 (CUDA 12.8).

## Citation

If you use this code, please cite the accompanying paper (citation details will be added after review).

## Acknowledgements

We thank an interventional radiologist for the expert ratings used to anchor the spatial concern
score, and a medical oncologist for clinical guidance.

## License

This project is released under the [MIT License](LICENSE).
The datasets are subject to their own licences (HCC-TACE-Seg: TCIA; LiTS: challenge terms).
