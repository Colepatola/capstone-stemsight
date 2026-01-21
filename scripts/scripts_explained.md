# Scripts Overview

## Training Pipeline

### `generate_masks.py`
Segments fluorescence images to identify live/dead cells.

- **Input:** `.1.jpg` (green/AO), `.2.jpg` (red/PI) from `data/caco2_dataset/`
- **Process:** Intensity thresholding → noise removal → connected components
- **Output:** `data/masks/*.1.png` (live), `data/masks/*.2.png` (dead)

### `extract_crops.py`
Extracts 32×32 brightfield patches around detected cells.

- **Input:** `.0.jpg` (brightfield) + masks from above
- **Process:** Find centroids in masks → crop brightfield at each location
- **Output:** `data/cell_crops/live/*.png`, `data/cell_crops/dead/*.png`

### `train_classifier.py`
Trains ResNet-18 CNN to classify live vs dead cells.

- **Input:** Crops from `data/cell_crops/`
- **Model:** ResNet-18 with pretrained ImageNet weights
- **Training:** 25 epochs, LR scheduler, class-weighted loss, data augmentation
- **Output:** `live_dead_classifier.pth`, `confusion_matrix.png`
- **Accuracy:** 90.1% validation

---

## Inference Pipeline

### `run_pipeline.py`
End-to-end classification from raw images.

- **Input:** Directory with image sets (`.0.jpg`, `.1.jpg`, `.2.jpg`)
- **Process:** Generate masks → detect cells → extract crops → classify
- **Output:** `classification_results.csv`, `*_annotated.png`
- **Accuracy:** 93.9% on test set

```bash
python scripts/run_pipeline.py \
    --input_dir "data/caco2_dataset/Set_1_Cam_AO_PI_19_02_16" \
    --output_dir "data/pipeline_output" \
    --visualize
```

### `live_dead_gui.py`
PyQt GUI for interactive classification.

- **Features:** Load model, select directories, run pipeline, preview results
- **Launch:** `python scripts/live_dead_gui.py`

---

## Reproducibility

```bash
# From cellpose_plus root directory

# Step 1 – Generate masks from fluorescence
python scripts/generate_masks.py

# Step 2 – Extract brightfield crops
python scripts/extract_crops.py

# Step 3 – Train classifier
python scripts/train_classifier.py

# Step 4 – Run inference pipeline
python scripts/run_pipeline.py --input_dir <path> --output_dir <path> --visualize

# Or use the GUI
python scripts/live_dead_gui.py
```

---

## File Summary

| Script | Purpose |
|--------|---------|
| `generate_masks.py` | Fluorescence → binary masks |
| `extract_crops.py` | Masks + brightfield → 32×32 crops |
| `train_classifier.py` | Train ResNet-18 classifier |
| `run_pipeline.py` | End-to-end CLI pipeline |
| `live_dead_gui.py` | Interactive GUI |
| `PROJECT_STATUS.md` | Current project status |
| `USAGE.md` | Quick start guide |
