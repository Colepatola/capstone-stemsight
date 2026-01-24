#!/usr/bin/env python3
"""
Convert ethz_iPSC Dataset for Live/Dead Classification

This script extracts the brightfield channel from multi-channel TIFF files
in the ethz_iPSC dataset and organizes them for training with the live/dead classifier.

Input:
    data/ethz_iPSC/ETHResearchCollection/iPSC_QCData/
    ├── Cell/           (healthy cells → "live")
    └── DyingCell/      (apoptotic cells → "dead")

Output:
    data/ipsc_cell_crops/
    ├── live/           (brightfield PNGs from Cell/)
    └── dead/           (brightfield PNGs from DyingCell/)

Usage:
    python scripts/convert_ipsc_dataset.py
"""

import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import tifffile

# === CONFIG ===
INPUT_DIR = Path("data/ethz_iPSC/ETHResearchCollection/iPSC_QCData")
OUTPUT_DIR = Path("data/ipsc_cell_crops")

# Channel mapping (from Data_Description.txt)
# Channel order: 647-Brightfield-DAPI-488-PE
BRIGHTFIELD_CHANNEL = 1  # Index 1 = Brightfield

# Class mapping: source folder → output label
CLASS_MAPPING = {
    "Cell": "live",
    "DyingCell": "dead",
    # Excluded: "Debris" (not cells), "MitoticCell" (different morphology)
}


def normalize_uint16_to_uint8(img):
    """
    Normalize uint16 image to uint8 for CNN input.

    Uses percentile-based normalization to handle outliers.
    """
    img = img.astype(np.float32)

    # Use percentile normalization (robust to outliers)
    p_low, p_high = np.percentile(img, [1, 99])

    if p_high - p_low > 0:
        img = (img - p_low) / (p_high - p_low)
    else:
        img = img - img.min()
        if img.max() > 0:
            img = img / img.max()

    # Clip and convert to uint8
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def process_tiff(tiff_path, output_path):
    """
    Extract brightfield channel from multi-channel TIFF and save as PNG.

    Args:
        tiff_path: Path to input TIFF file
        output_path: Path to output PNG file
    """
    # Read multi-channel TIFF
    img = tifffile.imread(tiff_path)

    # Expected shape: (height, width, channels) or (channels, height, width)
    if len(img.shape) == 3:
        if img.shape[0] < img.shape[2]:
            # Shape is (channels, height, width) - need to transpose
            img = np.transpose(img, (1, 2, 0))

        # Extract brightfield channel
        if img.shape[2] > BRIGHTFIELD_CHANNEL:
            brightfield = img[:, :, BRIGHTFIELD_CHANNEL]
        else:
            print(f"Warning: {tiff_path} has fewer channels than expected")
            return False
    else:
        print(f"Warning: {tiff_path} has unexpected shape {img.shape}")
        return False

    # Normalize to uint8
    brightfield_uint8 = normalize_uint16_to_uint8(brightfield)

    # Save as PNG
    Image.fromarray(brightfield_uint8).save(output_path)
    return True


def main():
    print("=" * 60)
    print("iPSC Dataset Conversion")
    print("=" * 60)

    # Check input directory
    if not INPUT_DIR.exists():
        print(f"Error: Input directory not found: {INPUT_DIR}")
        print("Make sure the ethz_iPSC dataset is extracted in data/ethz_iPSC/")
        return

    # Create output directories
    for label in CLASS_MAPPING.values():
        (OUTPUT_DIR / label).mkdir(parents=True, exist_ok=True)

    # Process each class
    total_converted = 0
    stats = {}

    for source_class, target_label in CLASS_MAPPING.items():
        source_dir = INPUT_DIR / source_class
        target_dir = OUTPUT_DIR / target_label

        if not source_dir.exists():
            print(f"Warning: Source directory not found: {source_dir}")
            continue

        # Find all TIFF files
        tiff_files = list(source_dir.glob("*.tiff")) + list(source_dir.glob("*.tif"))
        print(f"\nProcessing {source_class}/ → {target_label}/ ({len(tiff_files)} files)")

        converted = 0
        for tiff_path in tqdm(tiff_files, desc=f"  {source_class}"):
            # Create output filename
            output_name = tiff_path.stem + ".png"
            output_path = target_dir / output_name

            if process_tiff(tiff_path, output_path):
                converted += 1

        stats[target_label] = converted
        total_converted += converted

    # Print summary
    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"\nImages converted:")
    for label, count in stats.items():
        print(f"  {label}/: {count} images")
    print(f"  Total: {total_converted} images")

    print(f"\nNext step: Train the classifier with:")
    print(f"  python scripts/train_classifier.py --dataset ipsc")


if __name__ == "__main__":
    main()
