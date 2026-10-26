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

```bash
nnUNetv2_plan_and_preprocess -d 2 --verify_dataset_integrity
nnUNetv2_train 2 3d_fullres 0
```

Training was stopped manually; the best checkpoint (epoch 131, smoothed pseudo-Dice 0.678 on the
LiTS fold-0 validation split) was used for fine-tuning.

## 2. HCC-TACE-Seg plans aligned to the LiTS network (Dataset003)

```bash
nnUNetv2_plan_and_preprocess -d 3 --verify_dataset_integrity
python training/prepare_fine_tuning.py \
    $nnUNet_preprocessed/Dataset002_*/nnUNetPlans.json \
    $nnUNet_preprocessed/Dataset003_HCCTACE/nnUNetPlans.json
nnUNetv2_preprocess -d 3 -c 3d_fullres
```

`prepare_fine_tuning.py` copies the LiTS architecture and patch size (128³) into the HCC plans so
that the pretrained weights load. The configuration `3d_fullres_mcdropout` is the aligned
`3d_fullres` configuration with `dropout_op = torch.nn.Dropout3d` (p = 0.2); it is defined in
`config/nnunet_plans.json`, which can be copied to
`$nnUNet_preprocessed/Dataset003_HCCTACE/nnUNetPlans.json`. It shares the `3d_fullres`
preprocessed data, so no extra preprocessing is needed.

## 3. Fine-tuning with MC Dropout (fold 0: 80 development cases)

```bash
nnUNetv2_train 3 3d_fullres_mcdropout 0 \
    -pretrained_weights $nnUNet_results/Dataset002_*/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth
```

The fold-0 split is in `config/split.json` (nnU-Net's `splits_final.json`). Fine-tuning was stopped
manually after 320 epochs; `checkpoint_best.pth` (epoch 295, smoothed pseudo-Dice 0.717) is the model used in the paper.
