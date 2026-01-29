# Live/Dead Cell Classifier

This tool looks at microscope images of cells and figures out which ones are alive and which ones are dead.

## What It Does

Scientists often need to know if cells are alive or dead when doing experiments. The usual way is to use special dyes that glow different colors, but those cost money and take time. This classifier learns what live and dead cells look like, then can classify new cells without needing dyes.

The basic flow is:
1. Give it a microscope image with cells
2. It finds each cell in the image
3. Cuts out a small picture of each cell
4. A neural network looks at each one and guesses live or dead
5. You get back the image with colored boxes (green = live, red = dead) plus a CSV with all the predictions

## Two Modes

The classifier works with two different types of cell data:

**Caco2 mode** - For images that have fluorescence channels. The green channel shows live cells, red shows dead. Works really well, getting about 90% accuracy.

**iPSC mode** - For brightfield only images. Uses cell shape and texture to make predictions. Still being improved.

## Quick Start

The easiest way is to use the GUI:
```bash
cd scripts
python live_dead_gui.py
```

Or run from command line:
```bash
python scripts/run_pipeline.py --dataset ipsc \
    --input_dir "data/test_ipsc_small" \
    --output_dir "output" \
    --visualize
```

## Whats in This Repo

```
scripts/
  live_dead_gui.py      - GUI app with buttons and dropdowns
  run_pipeline.py       - Command line version
  train_classifier.py   - Train new models
  USAGE.md              - Detailed instructions

data/test_ipsc_small/   - Small test set (40 images) for quick testing

live_dead_classifier.pth - Trained model file
```

## Installation

You need Python with these packages:
```bash
pip install torch torchvision pillow opencv-python pandas numpy scipy
pip install pyqt5  # for the GUI
```

## Output

When you run the classifier you get:
- `classification_results.csv` - All the predictions with cell coordinates
- `*_annotated.png` - Images with boxes drawn around cells showing live (green) or dead (red)

## More Info

Check out `scripts/USAGE.md` for detailed usage instructions and examples.
