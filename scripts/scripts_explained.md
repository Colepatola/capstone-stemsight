# Scripts Explained

This file explains what each script does and how to use them.

## The Two Datasets

The pipeline works with two different types of cell data:

Caco2 (colon cancer cells)
- Has fluorescence images with special dyes
- Green dye marks live cells, red dye marks dead cells
- Images come in sets of three: .0 (brightfield), .1 (green), .2 (red)
- This one works well

iPSC (stem cells)
- Brightfield images only, no dyes
- Cells are already cut out and sorted into folders
- Still working on getting this to work as well as caco2

---

## Data Preparation Scripts

### convert_ipsc_dataset.py

This script converts the iPSC dataset into a format the classifier can use.

What it does:
- Reads the multi-channel TIFF files from the ethz_iPSC folder
- Pulls out just the brightfield channel (channel 1)
- Saves them as PNG files sorted into live and dead folders

Input: data/ethz_iPSC/ETHResearchCollection/iPSC_QCData/
- Cell folder = live cells
- DyingCell folder = dead cells

Output: data/ipsc_cell_crops/live/ and data/ipsc_cell_crops/dead/

How to run:
```
python scripts/convert_ipsc_dataset.py
```

You only need to run this once to set up the data.

### generate_masks.py (caco2 only)

This script looks at the fluorescence images and figures out where the cells are.

What it does:
- Reads the green (.1) and red (.2) images
- Finds bright spots which are the cells
- Creates mask images showing where live and dead cells are

Input: The .1 and .2 images from data/caco2_dataset/
Output: Mask images in data/masks/

### extract_crops.py (caco2 only)

This script cuts out small pictures of each cell from the brightfield images.

What it does:
- Uses the masks to find where cells are located
- Cuts out a 32x32 pixel square around each cell from the brightfield image
- Saves them sorted by live or dead

Input: The .0 brightfield images plus the masks
Output: Small cell images in data/cell_crops/live/ and data/cell_crops/dead/

---

## Training Script

### train_classifier.py

This script trains the neural network to tell live cells from dead cells.

What it does:
- Loads all the cell crop images
- Trains a ResNet-18 model to recognize the difference between live and dead
- Saves the trained model so you can use it later

How to run:
```
python scripts/train_classifier.py --dataset caco2
python scripts/train_classifier.py --dataset ipsc
```

The --dataset flag tells it which dataset to train on.

Other options:
- --epochs N = how many times to go through the training data (default 25)
- --batch_size N = how many images to process at once (default 32)
- --lr N = learning rate (default 0.001)

Output:
- live_dead_classifier_caco2.pth or live_dead_classifier_ipsc.pth (the trained model)
- confusion_matrix_caco2.png or confusion_matrix_ipsc.png (shows how accurate it is)

---

## Running the Pipeline

### run_pipeline.py

This is the main script that runs everything end to end. Give it a folder of images and it will classify all the cells.

For caco2 mode:
- Needs image sets with .0, .1, and .2 files
- Uses the fluorescence to find cells
- Classifies each cell as live or dead

For iPSC mode:
- Just needs brightfield images
- Uses image processing or Cellpose to find cells
- Classifies each cell as live or dead

How to run:
```
python scripts/run_pipeline.py --dataset caco2 --input_dir path/to/images --output_dir path/to/output --visualize

python scripts/run_pipeline.py --dataset ipsc --input_dir path/to/images --output_dir path/to/output --visualize
```

Options:
- --dataset = which mode to use (caco2 or ipsc)
- --input_dir = folder with your images
- --output_dir = where to save results
- --visualize = create images with boxes drawn around cells
- --cellpose = use Cellpose for finding cells (ipsc mode only)
- --model = use a different model file
- --threshold = adjust the brightness cutoff for finding cells (caco2 mode)

Output:
- classification_results.csv = list of all cells and their predictions
- annotated images = pictures with green boxes for live and red boxes for dead

### live_dead_gui.py

This is the GUI version so you dont have to type commands.

What it does:
- Lets you pick input and output folders by clicking
- Has a dropdown to switch between caco2 and ipsc mode
- Shows the results with a viewer so you can flip through the images

How to run:
```
python scripts/live_dead_gui.py
```

---

## Full Workflow

### If you want to use caco2 data:

1. Generate masks from the fluorescence images
```
python scripts/generate_masks.py
```

2. Extract cell crops from brightfield
```
python scripts/extract_crops.py
```

3. Train the classifier
```
python scripts/train_classifier.py --dataset caco2
```

4. Run on new images
```
python scripts/run_pipeline.py --dataset caco2 --input_dir your/folder --output_dir output/folder --visualize
```

### If you want to use iPSC data:

1. Convert the dataset (only need to do this once)
```
python scripts/convert_ipsc_dataset.py
```

2. Train the classifier
```
python scripts/train_classifier.py --dataset ipsc
```

3. Run on new images
```
python scripts/run_pipeline.py --dataset ipsc --input_dir your/folder --output_dir output/folder --visualize
```

---

## Quick Reference

Scripts for data prep:
- convert_ipsc_dataset.py = converts iPSC tiff files to usable format
- generate_masks.py = finds cells in caco2 fluorescence images
- extract_crops.py = cuts out individual cells from caco2 images

Scripts for training:
- train_classifier.py = trains the neural network

Scripts for running:
- run_pipeline.py = command line tool to classify cells
- live_dead_gui.py = same thing but with a graphical interface

Model files:
- live_dead_classifier_caco2.pth = trained model for caco2 data (works well)
- live_dead_classifier_ipsc.pth = trained model for ipsc data (still improving)
