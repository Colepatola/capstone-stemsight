# Live/Dead Cell Classifier - Usage Guide

## Quick Start

### Run the GUI
```bash
cd /Users/colepatola/Desktop/ASU/ASU-Year-4/CSE-485/cellpose_plus
python scripts/live_dead_gui.py
```

### Run from Command Line
```bash
python scripts/run_pipeline.py \
    --input_dir "data/caco2_dataset/Set_1_Cam_AO_PI_19_02_16" \
    --output_dir "data/pipeline_output" \
    --visualize
```

---

## GUI Instructions

1. **Model** — Auto-loads `live_dead_classifier.pth` from project root
2. **Input** — Click Browse, select folder with image sets
3. **Output** — Click Browse, select where to save results
4. **Run** — Click "Run Classification"
5. **View** — Browse annotated images with Previous/Next buttons

---

## Input Format

Images must follow this naming convention in the same folder:
```
1.0.jpg   ← Brightfield
1.1.jpg   ← Green fluorescence (AO = live)
1.2.jpg   ← Red fluorescence (PI = dead)
```

Each number (1, 2, 3...) is a separate image set.

---

## Output

- `classification_results.csv` — All cell predictions
- `*_annotated.png` — Images with bounding boxes (green=live, red=dead)

---

## Retrain the Model

```bash
python scripts/train_classifier.py
```

Outputs: `live_dead_classifier.pth`, `confusion_matrix.png`
