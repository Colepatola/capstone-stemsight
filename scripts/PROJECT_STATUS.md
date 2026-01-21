# Live/Dead Cell Classification - Project Status

## Current State: COMPLETE

### Performance
- **Training Accuracy:** 90.1%
- **Pipeline Accuracy:** 93.9%

| Metric | Dead | Live |
|--------|------|------|
| Precision | 0.93 | 0.87 |
| Recall | 0.89 | 0.92 |
| F1-Score | 0.91 | 0.89 |

---

## Files

| File | Purpose |
|------|---------|
| `train_classifier.py` | Train model (pretrained ResNet-18, 25 epochs) |
| `run_pipeline.py` | End-to-end inference pipeline |
| `live_dead_gui.py` | GUI application |
| `generate_masks.py` | Generate masks from fluorescence |
| `extract_crops.py` | Extract training crops |
| `USAGE.md` | Quick start guide |

---

## Quick Start

```bash
# Run GUI
python scripts/live_dead_gui.py

# Run CLI
python scripts/run_pipeline.py \
    --input_dir "data/caco2_dataset/Set_1_Cam_AO_PI_19_02_16" \
    --output_dir "data/pipeline_output" \
    --visualize

# Retrain model
python scripts/train_classifier.py
```

---

## Input Format

```
folder/
  1.0.jpg  ← Brightfield
  1.1.jpg  ← Green (AO = live)
  1.2.jpg  ← Red (PI = dead)
```

## Output

- `classification_results.csv` — Predictions with coordinates
- `*_annotated.png` — Images with bounding boxes (green=live, red=dead)
