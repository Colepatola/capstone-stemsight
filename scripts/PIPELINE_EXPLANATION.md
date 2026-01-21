## Live/Dead Cell Classification Pipeline

### Overview
This pipeline classifies cells as live or dead using only brightfield images. We train a CNN on labeled data from fluorescence staining, then apply it to new brightfield images where fluorescence isn't available.

### How It Works

**Step 1: Generate Ground Truth Masks**
- Fluorescence images give us labels: green (AO) = live, red (PI) = dead
- We threshold these images to create binary masks
- Each mask shows where live or dead cells are located

**Step 2: Extract Training Data**
- For each cell in the masks, we crop a 32x32 patch from the brightfield image
- These crops are saved to `live/` and `dead/` folders
- The brightfield crops look similar to human eyes, but the CNN learns subtle differences

**Step 3: Train the Classifier**
- ResNet-18 CNN with pretrained ImageNet weights
- Input: 32x32 brightfield crop → Output: live or dead
- Training uses data augmentation (flips, rotations, color jitter)
- Class weights handle the live/dead imbalance
- Result: 90% validation accuracy

**Step 4: Run Inference**
- New images go through the same pipeline: threshold fluorescence → find cells → crop brightfield → classify
- Or use the GUI to select folders and run everything automatically
- Output: CSV with predictions + annotated images with colored boxes

### Why This Works
The fluorescence staining tells us ground truth (live vs dead), but requires special dyes and imaging. The CNN learns to recognize live/dead cells from brightfield alone by picking up on subtle morphological differences - cell membrane integrity, internal structure, contrast patterns. Once trained, we can classify cells without needing fluorescence.

### Quick Start
```bash
# Train from scratch
python scripts/generate_masks.py
python scripts/extract_crops.py
python scripts/train_classifier.py

# Run on new images
python scripts/run_pipeline.py --input_dir <path> --output_dir <path> --visualize

# Or use the GUI
python scripts/live_dead_gui.py
```
