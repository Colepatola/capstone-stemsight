# Live/Dead Cell Classifier - Usage Guide

## Quick Start

### Caco2 Dataset (Default)
```bash
cd /Users/colepatola/Desktop/ASU/ASU-Year-4/CSE-485/cellpose_plus

# Run inference
python scripts/run_pipeline.py --dataset caco2 \
    --input_dir "data/caco2_dataset/Set_1_Cam_AO_PI_19_02_16" \
    --output_dir "data/pipeline_output" \
    --visualize
```

### iPSC Dataset
```bash
cd /Users/colepatola/Desktop/ASU/ASU-Year-4/CSE-485/cellpose_plus

# Step 1: Convert dataset (first time only)
python scripts/convert_ipsc_dataset.py

# Step 2: Train classifier
python scripts/train_classifier.py --dataset ipsc

# Step 3: Run inference
python scripts/run_pipeline.py --dataset ipsc \
    --input_dir "data/test_images/" \
    --output_dir "data/ipsc_output" \
    --visualize
```

---

## GUI (Caco2 only)

```bash
python scripts/live_dead_gui.py
```

1. **Model** — Auto-loads `live_dead_classifier.pth` from project root
2. **Input** — Click Browse, select folder with image sets
3. **Output** — Click Browse, select where to save results
4. **Run** — Click "Run Classification"
5. **View** — Browse annotated images with Previous/Next buttons

---

## Input Formats

### Caco2 Mode
Images must follow this naming convention in the same folder:
```
1.0.jpg   ← Brightfield
1.1.jpg   ← Green fluorescence (AO = live)
1.2.jpg   ← Red fluorescence (PI = dead)
```
Each number (1, 2, 3...) is a separate image set.

### iPSC Mode
Any brightfield images in the input directory:
```
image1.jpg
image2.png
sample_001.tif
```

---

## Output

- `classification_results.csv` — All cell predictions
- `*_annotated.png` — Images with bounding boxes (green=live, red=dead)

---

## Training

### Train on Caco2
```bash
python scripts/train_classifier.py --dataset caco2
```
Outputs: `live_dead_classifier_caco2.pth`, `confusion_matrix_caco2.png`

### Train on iPSC
```bash
# First convert the dataset
python scripts/convert_ipsc_dataset.py

# Then train
python scripts/train_classifier.py --dataset ipsc
```
Outputs: `live_dead_classifier_ipsc.pth`, `confusion_matrix_ipsc.png`

---

## Command Reference

### run_pipeline.py
```bash
python scripts/run_pipeline.py \
    --dataset {caco2,ipsc}     # Dataset mode (default: caco2)
    --input_dir PATH           # Input image directory
    --output_dir PATH          # Output directory for results
    --model PATH               # Custom model path (optional)
    --visualize                # Generate annotated images
    --cellpose                 # Use Cellpose segmentation (ipsc mode)
    --threshold N              # Fluorescence threshold (caco2 mode)
```

### train_classifier.py
```bash
python scripts/train_classifier.py \
    --dataset {caco2,ipsc}     # Dataset to train on (default: caco2)
    --epochs N                 # Training epochs (default: 25)
    --batch_size N             # Batch size (default: 32)
    --lr N                     # Learning rate (default: 0.001)
```

### convert_ipsc_dataset.py
```bash
python scripts/convert_ipsc_dataset.py
```
No arguments needed. Reads from `data/ethz_iPSC/`, outputs to `data/ipsc_cell_crops/`.

---

## Testing Both Pipelines

### Test Caco2 Pipeline
```bash
# Verify existing model works
python scripts/run_pipeline.py --dataset caco2 \
    --input_dir "data/caco2_dataset/Set_1_Cam_AO_PI_19_02_16" \
    --output_dir "data/test_caco2" \
    --visualize

# Check results
ls data/test_caco2/
cat data/test_caco2/classification_results.csv
```

### Test iPSC Pipeline
```bash
# Convert and train (if not done)
python scripts/convert_ipsc_dataset.py
python scripts/train_classifier.py --dataset ipsc

# Verify counts
ls data/ipsc_cell_crops/live/ | wc -l   # Should be ~1365
ls data/ipsc_cell_crops/dead/ | wc -l   # Should be ~1461

# Check model was created
ls -la live_dead_classifier_ipsc.pth
```
