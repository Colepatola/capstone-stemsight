# Meeting Notes - Live/Dead Cell Classification

## What is Cellpose Plus?
Cellpose Plus is an extension of Cellpose, which is a deep learning tool for cell segmentation. Our team is building additional analysis features on top of it. My role is developing the live/dead cell classification component.

## Problem I'm Solving
Researchers need to know if cells are alive or dead in microscopy images. Traditional methods require fluorescent dyes (AO/PI staining) which can be expensive and time-consuming. My pipeline learns to classify cells using just brightfield images - no special staining needed after training.

## What I Built

### The Pipeline
1. **Data Preparation** - Extract labeled cell crops from training data
2. **Training** - ResNet-18 CNN learns live vs dead patterns from brightfield
3. **Inference** - New images go through segmentation → crop extraction → classification
4. **Output** - CSV with predictions + annotated images (green boxes = live, red = dead)

### Key Features
- Supports two datasets via `--dataset` flag (caco2 or ipsc)
- GUI application for easy use (dropdown to switch modes)
- Command-line interface for batch processing
- Generates visual output with bounding boxes

## Datasets I Worked With

### Caco2 Dataset
- **Source**: Fluorescence microscopy images with AO/PI staining
- **Labels**: Green channel (AO) marks live cells, red channel (PI) marks dead cells
- **Process**: Generated masks from fluorescence → extracted 32x32 crops from brightfield
- **Result**: ~90% accuracy

### ethz_iPSC Dataset
- **Source**: Pre-cropped cell images from ETH Zurich research collection
- **Labels**: Manually labeled by experts (Cell folder = live, DyingCell folder = dead)
- **Process**: Extracted brightfield channel from 5-channel TIFFs → 100x100 crops
- **Result**: ~100% accuracy (morphologically distinct classes)

## Files I Created/Modified

| File | What It Does |
|------|--------------|
| `convert_ipsc_dataset.py` | Extracts brightfield from iPSC TIFFs, organizes into live/dead folders |
| `train_classifier.py` | Trains the CNN - added `--dataset` flag for caco2/ipsc |
| `run_pipeline.py` | End-to-end inference - added `--dataset` flag for both modes |
| `live_dead_gui.py` | GUI with dataset mode selector dropdown |

## Evidence of Work
- Both pipelines trained and tested
- Caco2: 90.1% accuracy on 1,895 cells
- iPSC: 100% accuracy on 2,826 cells
- GUI working with both dataset modes
- All code pushed to `cole-cellposeplus` branch on team repo

## Connection to Team
My classifier integrates with the main Cellpose Plus pipeline:
1. Cellpose segments cells in an image
2. My model takes each segmented cell
3. Classifies as live or dead
4. Returns results for downstream analysis

A teammate asked me to extract the image processing components so they can plug it into their pipeline. I've structured the code to make this possible.

## Next Steps
- Integrate with teammate's pipeline
- Test on new iPSC data from sponsor (Defined Biosciences)
- Potentially add more cell state classifications beyond live/dead
