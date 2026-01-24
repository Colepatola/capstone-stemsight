# Live/Dead Cell Classification Pipeline

## Overview
This pipeline classifies cells as live or dead using brightfield microscopy images. It supports two datasets with different approaches:

| Mode | Dataset | Training Labels | Inference Input |
|------|---------|-----------------|-----------------|
| **caco2** | Caco2 cells | Fluorescence staining (AO/PI) | Brightfield + fluorescence |
| **ipsc** | iPSC cells | Pre-labeled morphology data | Brightfield only |

---

## Caco2 Pipeline

### How It Works

**Step 1: Generate Ground Truth Masks**
- Fluorescence images provide labels: green (AO) = live, red (PI) = dead
- Threshold these images to create binary masks
- Each mask shows where live or dead cells are located

**Step 2: Extract Training Data**
- For each cell in the masks, crop a 32x32 patch from the brightfield image
- Save crops to `data/cell_crops/live/` and `data/cell_crops/dead/`
- The brightfield crops look similar to human eyes, but the CNN learns subtle differences

**Step 3: Train the Classifier**
- ResNet-18 CNN with pretrained ImageNet weights
- Input: 32x32 brightfield crop → Output: live or dead
- Training uses data augmentation (flips, rotations, color jitter)
- Class weights handle the live/dead imbalance
- Result: ~90% validation accuracy

**Step 4: Run Inference**
- New images go through the same pipeline:
  - Threshold fluorescence → find cells → crop brightfield → classify
- Output: CSV with predictions + annotated images

### Input Format (caco2)
```
folder/
  1.0.jpg  ← Brightfield
  1.1.jpg  ← Green fluorescence (AO = live)
  1.2.jpg  ← Red fluorescence (PI = dead)
```

### Quick Start (caco2)
```bash
# Train from scratch
python scripts/generate_masks.py
python scripts/extract_crops.py
python scripts/train_classifier.py --dataset caco2

# Run inference
python scripts/run_pipeline.py --dataset caco2 \
    --input_dir data/caco2_dataset/Set_1_Cam_AO_PI_19_02_16 \
    --output_dir data/output --visualize
```

---

## iPSC Pipeline

### How It Works

**Step 1: Convert Dataset**
- ethz_iPSC dataset contains pre-cropped, pre-labeled cell images
- Multi-channel TIFFs with brightfield in channel 1
- Labels from folder names: `Cell/` = live, `DyingCell/` = dead

**Step 2: Extract Brightfield Channel**
- Read 5-channel TIFFs from `data/ethz_iPSC/ETHResearchCollection/iPSC_QCData/`
- Extract channel 1 (brightfield)
- Normalize uint16 → uint8
- Save to `data/ipsc_cell_crops/live/` and `data/ipsc_cell_crops/dead/`

**Step 3: Train the Classifier**
- Same ResNet-18 architecture as caco2
- Input: 100x100 brightfield crop → Output: live or dead
- Result: ~100% validation accuracy (morphologically distinct classes)

**Step 4: Run Inference**
- Accept any brightfield images
- Segment cells using adaptive thresholding or Cellpose
- Classify each detected cell

### Input Format (ipsc)
```
folder/
  image1.jpg  ← Any brightfield image
  image2.png
  sample.tif
```

### Quick Start (ipsc)
```bash
# Convert dataset (first time only)
python scripts/convert_ipsc_dataset.py

# Train classifier
python scripts/train_classifier.py --dataset ipsc

# Run inference
python scripts/run_pipeline.py --dataset ipsc \
    --input_dir data/test_images \
    --output_dir data/output --visualize --cellpose
```

---

## Why This Works

### Caco2 (Fluorescence-Based Labels)
The fluorescence staining tells us ground truth (live vs dead), but requires special dyes and imaging. The CNN learns to recognize live/dead cells from brightfield alone by picking up on subtle morphological differences - cell membrane integrity, internal structure, contrast patterns. Once trained, we can classify cells without needing fluorescence.

### iPSC (Morphology-Based Labels)
The ethz_iPSC dataset was manually labeled by experts based on cell morphology. Healthy cells (`Cell/`) and dying cells (`DyingCell/`) have distinct visual characteristics that the CNN learns to recognize.

---

## GUI Usage

The GUI supports both modes via a dropdown selector:

```bash
python scripts/live_dead_gui.py
```

1. Select **Dataset Mode** (caco2 or ipsc)
2. Model auto-loads based on selection
3. Browse for input/output directories
4. Click **Run Classification**
5. View results with image navigation

---

## Output

Both pipelines produce:
- `classification_results.csv` — Cell ID, coordinates, prediction, confidence
- `*_annotated.png` — Images with colored bounding boxes (green=live, red=dead)

---

## Performance

| Pipeline | Training Accuracy | Inference Accuracy |
|----------|-------------------|-------------------|
| caco2 | ~90% | ~90% |
| ipsc | ~100% | ~100% |

---

## File Summary

| Script | Purpose | Dataset |
|--------|---------|---------|
| `convert_ipsc_dataset.py` | Convert ethz_iPSC TIFFs to crops | ipsc |
| `generate_masks.py` | Fluorescence → binary masks | caco2 |
| `extract_crops.py` | Masks + brightfield → crops | caco2 |
| `train_classifier.py` | Train ResNet-18 classifier | both |
| `run_pipeline.py` | End-to-end inference | both |
| `live_dead_gui.py` | Interactive GUI | both |
