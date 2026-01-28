# Meeting Notes - Live/Dead Cell Classification

## What is Cellpose?

Cellpose is a tool that finds cells in microscope images. It draws outlines around each cell so you can work with them one at a time. This is called segmentation.

You give it an image with a bunch of cells, and it gives you back a map showing where each cell is.

## What is Cellpose Plus?

Cellpose Plus is what our team is building on top of Cellpose. Cellpose finds the cells, but we want to do more than that. We want to answer questions like:

- Is this cell alive or dead?
- What kind of cell is it?
- Is it healthy?

My part of the project is the live/dead classification piece.

## How My Work Fits With Cellpose

Here is how the whole thing works:

1. Start with a microscope image that has lots of cells
2. Cellpose finds each cell and marks where it is
3. My code cuts out a small picture of each cell
4. My classifier looks at each cell picture and guesses if its live or dead
5. The output is the original image with boxes drawn around each cell (green = live, red = dead)

In the GUI I made, theres a checkbox to turn on Cellpose. If you dont use it, the code uses a simpler method to find cells instead.

## The Problem Im Solving

Scientists need to know if cells are alive or dead when they look at microscope images. The normal way to do this uses special dyes that glow different colors for live and dead cells. But those dyes cost money and take time to use.

My code learns what live and dead cells look like from training data. Then it can classify new cells without needing the dyes.

## What I Built

The pipeline has a few main parts:

1. Data prep - get labeled cell images ready for training
2. Training - teach a neural network to tell live from dead
3. Running on new images - find cells, cut them out, classify each one
4. Output - a CSV file with all the predictions plus images with colored boxes showing which cells are live or dead

Main features:
- Works with two different datasets (caco2 and ipsc)
- Has a GUI so you can just click buttons instead of typing commands
- Also works from command line if you want to process lots of images
- Makes pictures with boxes around cells so you can see the results

## The Datasets Im Working With

### Caco2 Dataset (working well)

This dataset has microscope images of colon cancer cells with special dyes. Green dye shows live cells, red dye shows dead cells. I used those colors to label the cells, then trained my model on just the regular brightfield images (no colors).

This one is working pretty well. Getting about 90% accuracy which is solid for this type of task.

### iPSC Dataset (still figuring this out)

This one is trickier. The iPSC data came from ETH Zurich and the cells were already cut out and labeled by researchers. Live cells are in a folder called Cell, dead ones are in DyingCell.

The images had 5 channels so I had to pull out just the brightfield channel to use.

Im still working on getting this to work as well as the caco2 dataset. The iPSC cells look different from caco2 cells so I might need to tweak some things. Right now Im testing different approaches to see what works best.

## Current Status

Working:
- Caco2 pipeline is solid, 90% accuracy
- GUI works and can switch between dataset modes
- Training and inference scripts are set up

Still in progress:
- iPSC classification needs more work
- Trying to figure out the best way to handle the different cell types
- May need to adjust the model or training process for iPSC

## Files I Made or Changed

- convert_ipsc_dataset.py - pulls out the brightfield channel from the iPSC images and sorts them into live/dead folders
- train_classifier.py - trains the neural network, I added a flag so you can pick which dataset to train on
- run_pipeline.py - runs the whole thing end to end, also has the dataset flag
- live_dead_gui.py - the GUI with a dropdown to switch between datasets

## How This Connects to the Team

The team is building Cellpose Plus together. Different people work on different parts:

- Cellpose does the cell finding
- My code does live/dead classification
- Other teammates are adding other features

One teammate asked me to pull out some of the image processing pieces (the masks and cell locations) so they can use them in their part of the project. The code is set up so thats easy to do.

## Whats Next

- Keep working on the iPSC classification to get better results
- Help teammate plug my code into their pipeline
- Test on new iPSC images from the sponsor (Defined Biosciences) when we get them
- Maybe add more categories besides just live and dead once the basics are solid
