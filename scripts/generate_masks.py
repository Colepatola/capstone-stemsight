import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage import morphology
from tqdm import tqdm

INPUT_DIR = 'data/caco2_dataset/Set_1_Cam_AO_PI_19_02_16'
MASK_DIR = 'data/masks'

# Step 1: Clear old masks
if os.path.exists(MASK_DIR):
    shutil.rmtree(MASK_DIR)
os.makedirs(MASK_DIR, exist_ok=True)

# Step 2: Generate masks
for filename in tqdm(sorted(os.listdir(INPUT_DIR))):
    if filename.endswith('.1.jpg') or filename.endswith('.2.jpg'):
        img_path = os.path.join(INPUT_DIR, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        _, thresh = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
        cleaned = morphology.remove_small_objects(thresh > 0, min_size=30)
        labeled = label(cleaned)

        out_name = filename.replace('.jpg', '.png')
        out_path = os.path.join(MASK_DIR, out_name)
        cv2.imwrite(out_path, labeled.astype(np.uint8))

        # Step 3: Show visual preview only for 1.1.jpg and 1.2.jpg
        if filename.startswith('1.1') or filename.startswith('1.2'):
            plt.figure(figsize=(12, 4))
            plt.suptitle(f'Preview for {filename}')
            plt.subplot(1, 3, 1)
            plt.imshow(img, cmap='gray')
            plt.title('Original')
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(thresh, cmap='gray')
            plt.title('Thresholded')
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(labeled, cmap='nipy_spectral')
            plt.title('Labeled Mask')
            plt.axis('off')

            plt.tight_layout()
            plt.show()
