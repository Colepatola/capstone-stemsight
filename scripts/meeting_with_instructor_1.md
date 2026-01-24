# Meeting Notes - Live/Dead Cell Classification

## What is Cellpose Plus?
Cellpose Plus is an extension of Cellpose (a cell segmentation tool). Our team is adding features for cell analysis - my focus is live/dead cell classification.

## What I Built
A pipeline that classifies cells as live or dead using brightfield microscopy images.

**How it works:**
- Train a CNN (ResNet-18) on labeled cell crops
- Model learns to distinguish live vs dead from brightfield alone
- Outputs predictions + annotated images with colored bounding boxes

**Results:**
- Caco2 pipeline: ~90% accuracy
- iPSC pipeline: ~100% accuracy

## Datasets

| Dataset | Type | Labels | Status |
|---------|------|--------|--------|
| Caco2 | Fluorescence-labeled | AO/PI staining (green=live, red=dead) | Complete |
| ethz_iPSC | Morphology-labeled | Pre-labeled by experts (Cell vs DyingCell) | Complete |

## Files Created
- `convert_ipsc_dataset.py` - Converts iPSC TIFFs to training data
- `train_classifier.py` - Trains the CNN (supports both datasets)
- `run_pipeline.py` - End-to-end inference
- `live_dead_gui.py` - GUI for running classification

## Next Steps
- Integration with teammate's pipeline
- Testing on new iPSC data from sponsor

## Connection to Team
My classifier can plug into the main Cellpose Plus pipeline - after cells are segmented, my model classifies each one as live or dead.
