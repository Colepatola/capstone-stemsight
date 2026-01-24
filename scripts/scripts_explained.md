# Scripts Overview

## Supported Datasets

The pipeline supports two datasets/modes:

| Dataset | Description | Input Format |
|---------|-------------|--------------|
| `caco2` | Caco2 cells with AO/PI fluorescence staining | `.0/.1/.2` image triplets |
| `ipsc` | iPSC cells with morphology-based labels | Brightfield-only |

---

## Data Preparation

### `convert_ipsc_dataset.py` (NEW)
Converts ethz_iPSC multi-channel TIFFs to training-ready format.

- **Input:** `data/ethz_iPSC/ETHResearchCollection/iPSC_QCData/`
  - `Cell/` folder (healthy cells) → "live"
  - `DyingCell/` folder (apoptotic cells) → "dead"
- **Process:** Extract brightfield channel (channel 1) → normalize uint16 → save as PNG
- **Output:** `data/ipsc_cell_crops/live/*.png`, `data/ipsc_cell_crops/dead/*.png`

```bash
python scripts/convert_ipsc_dataset.py
```

### `generate_masks.py` (caco2 only)
Segments fluorescence images to identify live/dead cells.

- **Input:** `.1.jpg` (green/AO), `.2.jpg` (red/PI) from `data/caco2_dataset/`
- **Process:** Intensity thresholding → noise removal → connected components
- **Output:** `data/masks/*.1.png` (live), `data/masks/*.2.png` (dead)

### `extract_crops.py` (caco2 only)
Extracts 32x32 brightfield patches around detected cells.

- **Input:** `.0.jpg` (brightfield) + masks from above
- **Process:** Find centroids in masks → crop brightfield at each location
- **Output:** `data/cell_crops/live/*.png`, `data/cell_crops/dead/*.png`

---

## Training Pipeline

### `train_classifier.py` (MODIFIED)
Trains ResNet-18 CNN to classify live vs dead cells.

**New:** Supports `--dataset` flag to switch between datasets.

- **Input:** Crops from `data/cell_crops/` (caco2) or `data/ipsc_cell_crops/` (ipsc)
- **Model:** ResNet-18 with pretrained ImageNet weights
- **Training:** 25 epochs, LR scheduler, class-weighted loss, data augmentation
- **Output:** `live_dead_classifier_{dataset}.pth`, `confusion_matrix_{dataset}.png`

```bash
# Train on caco2 (default, original behavior)
python scripts/train_classifier.py --dataset caco2

# Train on iPSC
python scripts/train_classifier.py --dataset ipsc
```

**Additional options:**
- `--epochs N` — Number of training epochs (default: 25)
- `--batch_size N` — Batch size (default: 32)
- `--lr N` — Learning rate (default: 0.001)

---

## Inference Pipeline

### `run_pipeline.py` (MODIFIED)
End-to-end classification from raw images.

**New:** Supports `--dataset` flag to switch between modes.

**Caco2 mode:**
- **Input:** Directory with image sets (`.0.jpg`, `.1.jpg`, `.2.jpg`)
- **Process:** Generate masks → detect cells → extract crops → classify
- **Output:** `classification_results.csv`, `*_annotated.png`

**iPSC mode:**
- **Input:** Directory with brightfield images
- **Process:** Segment cells (adaptive threshold or Cellpose) → extract crops → classify
- **Output:** `classification_results.csv`, `*_annotated.png`

```bash
# Caco2 mode (default)
python scripts/run_pipeline.py --dataset caco2 \
    --input_dir "data/caco2_dataset/Set_1_Cam_AO_PI_19_02_16" \
    --output_dir "data/pipeline_output" \
    --visualize

# iPSC mode
python scripts/run_pipeline.py --dataset ipsc \
    --input_dir "data/test_brightfield/" \
    --output_dir "data/ipsc_output" \
    --visualize --cellpose
```

**Additional options:**
- `--model PATH` — Custom model path
- `--visualize` — Generate annotated images
- `--cellpose` — Use Cellpose for segmentation (iPSC mode)
- `--threshold N` — Fluorescence threshold (caco2 mode)

### `live_dead_gui.py`
PyQt GUI for interactive classification.

- **Features:** Load model, select directories, run pipeline, preview results
- **Launch:** `python scripts/live_dead_gui.py`
- **Note:** Currently uses caco2 mode only

---

## Full Workflow

### Caco2 Pipeline
```bash
cd /Users/colepatola/Desktop/ASU/ASU-Year-4/CSE-485/cellpose_plus

# Step 1 – Generate masks from fluorescence
python scripts/generate_masks.py

# Step 2 – Extract brightfield crops
python scripts/extract_crops.py

# Step 3 – Train classifier
python scripts/train_classifier.py --dataset caco2

# Step 4 – Run inference
python scripts/run_pipeline.py --dataset caco2 --input_dir <path> --output_dir <path> --visualize
```

### iPSC Pipeline
```bash
cd /Users/colepatola/Desktop/ASU/ASU-Year-4/CSE-485/cellpose_plus

# Step 1 – Convert iPSC dataset (one-time)
python scripts/convert_ipsc_dataset.py

# Step 2 – Train classifier on iPSC data
python scripts/train_classifier.py --dataset ipsc

# Step 3 – Run inference
python scripts/run_pipeline.py --dataset ipsc --input_dir <path> --output_dir <path> --visualize
```

---

## File Summary

| Script | Purpose | Dataset |
|--------|---------|---------|
| `convert_ipsc_dataset.py` | Convert ethz_iPSC TIFFs to crops | ipsc |
| `generate_masks.py` | Fluorescence → binary masks | caco2 |
| `extract_crops.py` | Masks + brightfield → 32x32 crops | caco2 |
| `train_classifier.py` | Train ResNet-18 classifier | both |
| `run_pipeline.py` | End-to-end CLI pipeline | both |
| `live_dead_gui.py` | Interactive GUI | caco2 |

---

## Model Files

| File | Dataset | Input Size | Description |
|------|---------|------------|-------------|
| `live_dead_classifier_caco2.pth` | caco2 | 64x64 | Trained on fluorescence-labeled caco2 crops |
| `live_dead_classifier_ipsc.pth` | ipsc | 100x100 | Trained on morphology-labeled iPSC crops |
| `live_dead_classifier.pth` | legacy | 64x64 | Original model (same as caco2) |
