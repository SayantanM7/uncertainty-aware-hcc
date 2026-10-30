# Training

The model is nnU-Net v2 (3D full resolution), pretrained on LiTS and fine-tuned on HCC-TACE-Seg
with Dropout3d. Inference for the paper was run with nnunetv2 2.8.1.

```bash
export nnUNet_raw=/path/to/nnUNet_raw              # folder that contains Dataset00X_* folders
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
export nnUNet_compile=False
```

## 1. LiTS pretraining (Dataset002, binary tumour label, no dropout)

LiTS labels were binarised (tumour = 1, liver merged into background; 118 cases with tumour), as
described under *Data preparation* in the main README.

```bash
nnUNetv2_plan_and_preprocess -d 2 --verify_dataset_integrity
nnUNetv2_train 2 3d_fullres 0
```

Training was stopped manually; the best checkpoint (epoch 131, smoothed pseudo-Dice 0.678 on the
LiTS fold-0 validation split) was used for fine-tuning.

## 2. HCC-TACE-Seg plans aligned to the LiTS network (Dataset003)

```bash
nnUNetv2_plan_and_preprocess -d 3 --verify_dataset_integrity
cp config/nnunet_plans.json $nnUNet_preprocessed/Dataset003_HCCTACE/nnUNetPlans.json
cp config/split.json $nnUNet_preprocessed/Dataset003_HCCTACE/splits_final.json
nnUNetv2_preprocess -d 3 -c 3d_fullres
```

`config/nnunet_plans.json` is the plans file used in the paper. It contains the aligned `3d_fullres`
configuration (LiTS architecture and 128³ patch size, so the pretrained weights load), the
`3d_fullres_mcdropout` configuration (the same with `dropout_op = torch.nn.Dropout3d`, p = 0.2), and
the dataset fingerprint, so normalisation matches the paper. `3d_fullres_mcdropout` shares the
`3d_fullres` preprocessed data, so no extra preprocessing is needed. `config/split.json` is the fold-0
split used in the paper (80 development / 20 validation cases).

The plans were produced from nnU-Net's own HCC plans with `prepare_fine_tuning.py`, which copies the
LiTS `architecture` and `patch_size` into the HCC `3d_fullres` configuration:

```bash
python training/prepare_fine_tuning.py \
    $nnUNet_preprocessed/Dataset002_*/nnUNetPlans.json \
    $nnUNet_preprocessed/Dataset003_HCCTACE/nnUNetPlans.json
```

followed by adding `3d_fullres_mcdropout` as a copy of `3d_fullres` with Dropout3d (p = 0.2).

## 3. Fine-tuning with MC Dropout (fold 0: 80 development cases)

```bash
nnUNetv2_train 3 3d_fullres_mcdropout 0 \
    -pretrained_weights $nnUNet_results/Dataset002_*/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth
```

Fine-tuning was stopped manually after 320 epochs; `checkpoint_best.pth` (epoch 295, smoothed
pseudo-Dice 0.717) is the model used in the paper. The trained weights are not included in this
repository.
