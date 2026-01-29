# Live/Dead Cell Classifier - Usage Guide

## Overview

This is a standalone tool that classifies cells as **live** or **dead** from microscopy images.

### How It Works

```
Microscopy Image → [Segmentation] → Cell Crops → [CNN Classifier] → Live/Dead Labels
```

1. **Segmentation**: Locates each cell using thresholding (or optionally Cellpose)
2. **Crop Extraction**: Each cell is cropped from the original image
3. **Classification**: A trained ResNet-18 CNN examines each crop and predicts live or dead
4. **Output**: Annotated images with colored boxes (green=live, red=dead) + CSV results

### Segmentation Options

The pipeline uses **adaptive thresholding** by default, which works well for most cases.

For iPSC mode, you can optionally enable **Cellpose segmentation** for better accuracy:
- **GUI**: Check "Use Cellpose for segmentation"
- **CLI**: Add `--cellpose` flag

---

## GUI (Recommended)

The GUI supports both caco2 and iPSC datasets.

### How to Run
```bash
cd scripts
python live_dead_gui.py
```

### GUI Steps
1. **Dataset Mode** — Select `caco2` or `ipsc` from dropdown
2. **Model** — Auto-loads the correct model for selected mode
3. **Input** — Click Browse, select your input folder
4. **Output** — Click Browse, select where to save results
5. **Run** — Click "Run Classification"
6. **View** — Browse annotated images with Previous/Next buttons

---

## Testing the GUI

### Test Caco2 Mode
1. Set **Dataset Mode** → `caco2`
2. **Input** → `data/caco2_dataset/Set_1_Cam_AO_PI_19_02_16`
3. **Output** → `data/test_gui_caco2` (or any folder)
4. Click **Run Classification**
5. Should see ~90% accuracy on ~1,895 cells

### Test iPSC Mode (Quick Test - Recommended)
1. Set **Dataset Mode** → `ipsc`
2. **Input** → `data/test_ipsc_small/` (40 images: 20 live + 20 dead)
3. **Output** → `data/test_gui_ipsc` (or any folder)
4. Click **Run Classification**
5. Should see ~50% live, ~50% dead (balanced test set)
6. Runs in seconds instead of minutes

### Test iPSC Mode (Full Dataset)
1. Set **Dataset Mode** → `ipsc`
2. **Input** → `data/ipsc_cell_crops/live/` (1,365 images)
3. **Output** → `data/test_gui_ipsc` (or any folder)
4. Click **Run Classification**
5. Most predictions should be "live" (validating the model works)
6. Note: Takes longer due to large dataset size

---

## Command Line Usage

### Caco2 Pipeline
```bash
python scripts/run_pipeline.py --dataset caco2 \
    --input_dir "data/caco2_dataset/Set_1_Cam_AO_PI_19_02_16" \
    --output_dir "data/pipeline_output" \
    --visualize
```

### iPSC Pipeline
```bash
python scripts/run_pipeline.py --dataset ipsc \
    --input_dir "data/ipsc_cell_crops/live" \
    --output_dir "data/ipsc_output" \
    --visualize
```

---

## Input Formats

### Caco2 Mode
Requires three images per sample:
```
folder/
  1.0.jpg   ← Brightfield
  1.1.jpg   ← Green fluorescence (AO = live)
  1.2.jpg   ← Red fluorescence (PI = dead)
```

### iPSC Mode
Any brightfield images:
```
folder/
  image1.jpg
  image2.png
  sample.tif
```

---

## Output

Both modes produce:
- `classification_results.csv` — Cell predictions with coordinates
- `*_annotated.png` — Images with bounding boxes (green=live, red=dead)

---

## Training

### Train Caco2 Classifier
```bash
python scripts/train_classifier.py --dataset caco2
```
Output: `live_dead_classifier_caco2.pth`

### Train iPSC Classifier
```bash
# First convert dataset (one time only)
python scripts/convert_ipsc_dataset.py

# Then train
python scripts/train_classifier.py --dataset ipsc
```
Output: `live_dead_classifier_ipsc.pth`

---

## Command Reference

### run_pipeline.py
```
--dataset {caco2,ipsc}   Dataset mode (default: caco2)
--input_dir PATH         Input image directory
--output_dir PATH        Output directory for results
--model PATH             Custom model path (optional)
--visualize              Generate annotated images
--cellpose               Use Cellpose segmentation (ipsc mode)
--threshold N            Fluorescence threshold (caco2 mode)
```

### train_classifier.py
```
--dataset {caco2,ipsc}   Dataset to train on (default: caco2)
--epochs N               Training epochs (default: 25)
--batch_size N           Batch size (default: 32)
--lr N                   Learning rate (default: 0.001)
```

### convert_ipsc_dataset.py
```bash
python scripts/convert_ipsc_dataset.py
```
No arguments. Reads from `data/ethz_iPSC/`, outputs to `data/ipsc_cell_crops/`.

---

## Available Test Data

| Folder | Mode | Description |
|--------|------|-------------|
| `data/caco2_dataset/Set_1_Cam_AO_PI_19_02_16` | caco2 | 34 image sets with .0/.1/.2 files |
| `data/test_ipsc_small/` | ipsc | 40 images (20 live + 20 dead) - quick testing |
| `data/ipsc_cell_crops/live/` | ipsc | 1,365 live cell images |
| `data/ipsc_cell_crops/dead/` | ipsc | 1,461 dead cell images |
