import os
import cv2
import numpy as np
from tqdm import tqdm
from skimage.measure import regionprops, label
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='skimage')

IMAGE_DIR = 'data/caco2_dataset/Set_1_Cam_AO_PI_19_02_16'
MASK_DIR = 'data/masks'
CROP_DIR = 'data/cell_crops'
os.makedirs(os.path.join(CROP_DIR, 'live'), exist_ok=True)
os.makedirs(os.path.join(CROP_DIR, 'dead'), exist_ok=True)

live_shown = False
dead_shown = False

for filename in tqdm(sorted(os.listdir(IMAGE_DIR))):
    if not filename.endswith('.0.jpg'):
        continue

    base = filename.split('.')[0]

    brightfield = cv2.imread(os.path.join(IMAGE_DIR, filename), cv2.IMREAD_GRAYSCALE)
    green_mask = cv2.imread(os.path.join(MASK_DIR, f"{base}.1.png"), cv2.IMREAD_GRAYSCALE)
    red_mask = cv2.imread(os.path.join(MASK_DIR, f"{base}.2.png"), cv2.IMREAD_GRAYSCALE)

    green_label = label(green_mask)
    red_label = label(red_mask)

    used_coords = set()

    # Extract live (green)
    for region in regionprops(green_label):
        y, x = region.centroid
        x, y = int(x), int(y)
        if (x, y) in used_coords:
            continue
        crop = brightfield[y-16:y+16, x-16:x+16]
        if crop.shape == (32, 32):
            imsave(os.path.join(CROP_DIR, 'live', f"{base}_live_{x}.png"), crop)
            used_coords.add((x, y))
            if not live_shown:
                plt.figure(figsize=(2, 2))
                plt.title(f"First Live Cell: {base}_live_{x}.png")
                plt.imshow(crop, cmap='gray')
                plt.axis('off')
                plt.show()
                live_shown = True

    # Extract dead (red)
    for region in regionprops(red_label):
        y, x = region.centroid
        x, y = int(x), int(y)
        if (x, y) in used_coords:
            continue
        crop = brightfield[y-16:y+16, x-16:x+16]
        if crop.shape == (32, 32):
            imsave(os.path.join(CROP_DIR, 'dead', f"{base}_dead_{x}.png"), crop)
            used_coords.add((x, y))
            if not dead_shown:
                plt.figure(figsize=(2, 2))
                plt.title(f"First Dead Cell: {base}_dead_{x}.png")
                plt.imshow(crop, cmap='gray')
                plt.axis('off')
                plt.show()
                dead_shown = True
