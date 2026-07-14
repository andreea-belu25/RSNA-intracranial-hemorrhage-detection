# Intracranial Hemorrhage Detection from CT Scans

A multi-label classification system that detects five types of intracranial hemorrhage from head CT images, built on the RSNA Intracranial Hemorrhage Detection dataset.

The final model, **EfficientNet-B7** fine-tuned with MONAI, reaches **75.28% validation accuracy**, **0.86 precision**, **0.69 recall**, and **0.76 F1-score** across all six labels.

---

## Hemorrhage Types

| Label | Type | Location |
|-------|------|----------|
| EDH | Epidural | Between skull and dura mater |
| IPH | Intraparenchymal | Inside brain tissue |
| IVH | Intraventricular | Inside the ventricles |
| SAH | Subarachnoid | Surface of the brain, in the sulci |
| SDH | Subdural | Between dura mater and brain surface |
| Any | Any of the above | — |

Each image can have **multiple labels simultaneously** (multi-label classification), so a prediction is considered correct only when all six labels match the ground truth.

## Notebooks

### `prerequisites.ipynb`

Covers the full data pipeline before training:

1. **Dataset class**: Custom `RSNAHemorrhageDataset` (PyTorch `Dataset`) with lazy loading and CSV label parsing for all six categories.
2. **Train/validation split**: 80/20 split using `train_test_split` with `random_state=42` for reproducibility.
3. **Class distribution analysis**: Bar charts across train, validation and test sets revealing class imbalance.
4. **Visual analysis**: Side-by-side CT samples for each hemorrhage type, with anatomical observations on hyperdense patterns, location and symmetry.
5. **Data consistency checks**: Verification of channel count, image dimensions and pixel value ranges.
6. **Preprocessing operations**:
   - *Contrast*: CLAHE, Histogram Equalization, Random Adjust Contrast, Gaussian Sharpen
   - *Noise reduction*: Gaussian Smooth, Random Gaussian Smooth, Median Filter
   - *Augmentation*: Random Flip, Random Rotate, Random Zoom
   - *Structural*: Sobel Edge Detection
   - *Standardization*: Resize (128×128), Scale Intensity ([0, 1])

### `training.ipynb`

End-to-end training and evaluation pipeline using **MONAI** and **PyTorch**:

1. **Data loading** — Dictionary-based MONAI transforms with differentiated pipelines:
   - *Training*: `LoadImaged` → `EnsureChannelFirstd` → `ScaleIntensityd` → `Resized(128×128)` → `MedianSmoothd` → `AdjustContrastd` → `RandFliped` → `RandRotated(±15°)` → `RandZoomd(0.9–1.1)` → `ToTensord`
   - *Validation/Test*: Same base transforms without augmentation
2. **Model** — `EfficientNet-B7` pretrained on ImageNet, modified for 1-channel grayscale input and 6-class multi-label output.
3. **Training** — `BCEWithLogitsLoss`, Adam optimizer (lr=1e-4), batch size 32, 15 epochs with best-model checkpointing on validation accuracy.
4. **Evaluation** — Precision, recall, F1-score, overall confusion matrix, and per-class confusion matrices with individual accuracy scores.
   
---

## Results

### Final Model Performance

| Metric | Value |
|--------|-------|
| Test Loss | 0.5390 |
| Test Accuracy (exact match) | 37.88% |
| Precision | 0.8597 |
| Recall | 0.6881 |
| F1-Score | 0.7585 |

### Per-Class Accuracy

| Class | Accuracy |
|-------|----------|
| Epidural | 95.91% |
| Intraventricular | 86.20% |
| Intraparenchymal | 82.19% |
| Subdural | 76.21% |
| Subarachnoid | 73.82% |
