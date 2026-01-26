# Meeting Notes - Live/Dead Cell Classification

## What is Cellpose?
Cellpose is a deep learning model that finds and outlines individual cells in microscopy images. Think of it like drawing a boundary around each cell automatically. This process is called **segmentation**.

- Input: A microscopy image with many cells
- Output: A mask showing where each individual cell is located

## What is Cellpose Plus?
Cellpose Plus is our team's extension that adds **analysis features** on top of Cellpose's segmentation. Instead of just finding cells, we want to answer questions about them:

- Are they alive or dead?
- What type of cell is it?
- How healthy does it look?

**My contribution:** I built the **live/dead classification** component.

## How Cellpose Integrates With My Work
The pipeline works in stages:

```
Raw Image → [Cellpose Segmentation] → Individual Cell Crops → [My Classifier] → Live/Dead Labels
```

1. **Cellpose** finds each cell in the image and creates a mask
2. **My code** extracts a small crop of each cell from the original image
3. **My CNN classifier** looks at each crop and predicts: live or dead
4. **Output** shows the original image with colored boxes (green=live, red=dead)

In the GUI, there's a checkbox "Use Cellpose for segmentation" that enables Cellpose. Without it, I use simpler image processing (adaptive thresholding) as a fallback.

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
My classifier is one module in the larger Cellpose Plus system:

```
[Cellpose Core]     →  Segments cells (finds boundaries)
[My Classifier]     →  Determines if cells are live or dead
[Other Teammates]   →  Additional analysis features
```

The workflow:
1. **Cellpose** (or fallback segmentation) finds cell locations
2. **My classifier** takes each cell crop and predicts live/dead
3. **Results** feed into the team's larger analysis pipeline

A teammate asked me to extract the image processing components (masks, crops, centroids) so they can plug it into their work. The code is modular to support this.

## Next Steps
- Integrate with teammate's pipeline
- Test on new iPSC data from sponsor (Defined Biosciences)
- Potentially add more cell state classifications beyond live/dead
