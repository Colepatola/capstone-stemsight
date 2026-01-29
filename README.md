# Live/Dead Cell Classifier

A deep learning tool that classifies cells as **live** or **dead** from microscopy images.

## Overview

This classifier uses a ResNet-18 CNN to analyze cell images and predict whether each cell is alive or dead. It supports two dataset modes:

- **caco2**: Requires fluorescence-labeled images (brightfield + green/red channels)
- **ipsc**: Brightfield-only images with morphology-based classification

## Installation

```bash
pip install torch torchvision pillow opencv-python pandas numpy scipy
pip install pyqt5  # For GUI
```

## Quick Start

### GUI (Recommended)
```bash
cd scripts
python live_dead_gui.py
```

### Command Line
```bash
python scripts/run_pipeline.py --dataset ipsc \
    --input_dir "data/test_ipsc_small" \
    --output_dir "output" \
    --visualize
```

## Files

- `scripts/` - Main classifier code
  - `live_dead_gui.py` - GUI application
  - `run_pipeline.py` - Command line pipeline
  - `train_classifier.py` - Model training
  - `USAGE.md` - Detailed usage guide
- `data/test_ipsc_small/` - Sample test images (20 live + 20 dead)
- `live_dead_classifier.pth` - Trained model weights

## Output

- `classification_results.csv` - Predictions with cell coordinates
- `*_annotated.png` - Images with colored boxes (green=live, red=dead)

## Documentation

See `scripts/USAGE.md` for detailed usage instructions.

## License

GPL v3 - See LICENSE file
