# Live/Dead Cell Classification - Project Status

## Current State: COMPLETE (Dual Pipeline Support)

Supports two datasets/modes:
- **caco2**: Caco2 cells with AO/PI fluorescence staining
- **ipsc**: iPSC cells with morphology-based labels

---

## Performance

### Caco2 Pipeline
| Metric | Dead | Live |
|--------|------|------|
| Precision | 0.93 | 0.87 |
| Recall | 0.89 | 0.92 |
| F1-Score | 0.91 | 0.89 |

- **Training Accuracy:** ~90%
- **Pipeline Accuracy:** 90.1%

### iPSC Pipeline
| Metric | Dead | Live |
|--------|------|------|
| Precision | 1.00 | 1.00 |
| Recall | 1.00 | 1.00 |
| F1-Score | 1.00 | 1.00 |

- **Training Accuracy:** 100%
- **Dataset:** 2,826 images (1,365 live + 1,461 dead)

---

## Files

| File | Purpose | Dataset |
|------|---------|---------|
| `convert_ipsc_dataset.py` | Convert ethz_iPSC TIFFs to crops | ipsc |
| `generate_masks.py` | Generate masks from fluorescence | caco2 |
| `extract_crops.py` | Extract training crops | caco2 |
| `train_classifier.py` | Train model (--dataset flag) | both |
| `run_pipeline.py` | End-to-end inference (--dataset flag) | both |
| `live_dead_gui.py` | GUI application | caco2 |
| `USAGE.md` | Quick start guide | both |
| `scripts_explained.md` | Detailed documentation | both |

---

## Quick Start

### Caco2 Pipeline
```bash
# Run GUI
python scripts/live_dead_gui.py

# Run CLI
python scripts/run_pipeline.py --dataset caco2 \
    --input_dir "data/caco2_dataset/Set_1_Cam_AO_PI_19_02_16" \
    --output_dir "data/pipeline_output" \
    --visualize

# Retrain model
python scripts/train_classifier.py --dataset caco2
```

### iPSC Pipeline
```bash
# Step 1: Convert dataset (first time only)
python scripts/convert_ipsc_dataset.py

# Step 2: Train classifier
python scripts/train_classifier.py --dataset ipsc

# Step 3: Run inference
python scripts/run_pipeline.py --dataset ipsc \
    --input_dir "data/test_images" \
    --output_dir "data/ipsc_output" \
    --visualize
```

---

## Input Formats

### Caco2 Mode
```
folder/
  1.0.jpg  ← Brightfield
  1.1.jpg  ← Green (AO = live)
  1.2.jpg  ← Red (PI = dead)
```

### iPSC Mode
```
folder/
  image1.jpg  ← Brightfield only
  image2.png
```

---

## Output

- `classification_results.csv` — Predictions with coordinates
- `*_annotated.png` — Images with bounding boxes (green=live, red=dead)

---

## Model Files

| File | Dataset | Input Size |
|------|---------|------------|
| `live_dead_classifier_caco2.pth` | caco2 | 64x64 |
| `live_dead_classifier_ipsc.pth` | ipsc | 100x100 |
| `live_dead_classifier.pth` | legacy | 64x64 |
