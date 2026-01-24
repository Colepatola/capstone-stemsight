#!/usr/bin/env python3
"""
End-to-end Live/Dead Cell Classification Pipeline

Supports two modes:
  - caco2: Requires brightfield + fluorescence images (.0/.1/.2 naming)
  - ipsc:  Brightfield-only with Cellpose segmentation

Input (caco2 mode):
  Directory with images following naming convention:
    - *.0.jpg (or .png) = brightfield
    - *.1.jpg (or .png) = green fluorescence (AO stain = live)
    - *.2.jpg (or .png) = red fluorescence (PI stain = dead)

Input (ipsc mode):
  Directory with brightfield images (any .jpg/.png/.tif files)

Output:
  - CSV with cell locations and predictions
  - Annotated images with colored bounding boxes (optional)

Usage:
  python run_pipeline.py --dataset caco2 --input_dir <path> --output_dir <path> [--visualize]
  python run_pipeline.py --dataset ipsc --input_dir <path> --output_dir <path> [--visualize]
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from scipy import ndimage


# === DATASET CONFIGURATION ===
DATASET_CONFIG = {
    'caco2': {
        'model_path': 'live_dead_classifier_caco2.pth',
        'crop_size': 32,
        'resize': 64,
        'description': 'Caco2 mode (requires .0/.1/.2 image triplets)'
    },
    'ipsc': {
        'model_path': 'live_dead_classifier_ipsc.pth',
        'crop_size': 50,  # Larger crop for iPSC cells
        'resize': 100,
        'description': 'iPSC mode (brightfield-only with Cellpose segmentation)'
    }
}


class LiveDeadPipeline:
    """End-to-end pipeline for live/dead cell classification."""

    def __init__(self, model_path, device=None, crop_size=32, resize=64):
        """
        Initialize the pipeline.

        Args:
            model_path: Path to trained classifier .pth file
            device: torch device (auto-detected if None)
            crop_size: Size of cell crops to extract
            resize: Size to resize crops for model input
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.crop_size = crop_size
        self.resize = resize
        # Must match training transforms (ImageNet normalization for pretrained weights)
        self.transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.classes = ['dead', 'live']  # Must match training order (alphabetical)

    def _load_model(self, model_path):
        """Load the trained ResNet-18 classifier."""
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(512, 2)
        model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        model.to(self.device)
        model.eval()
        return model

    def generate_mask(self, fluorescence_img, threshold=30, min_area=50):
        """
        Generate binary mask from fluorescence image using thresholding.

        Args:
            fluorescence_img: Grayscale or RGB fluorescence image (numpy array)
            threshold: Intensity threshold for binarization
            min_area: Minimum cell area in pixels

        Returns:
            Binary mask (numpy array)
        """
        # Convert to grayscale if needed
        if len(fluorescence_img.shape) == 3:
            gray = cv2.cvtColor(fluorescence_img, cv2.COLOR_RGB2GRAY)
        else:
            gray = fluorescence_img

        # Threshold
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Remove small objects
        labeled, num_features = ndimage.label(binary)
        for i in range(1, num_features + 1):
            if np.sum(labeled == i) < min_area:
                binary[labeled == i] = 0

        return binary

    def find_cell_centroids(self, mask):
        """
        Find centroids of cells in a binary mask.

        Args:
            mask: Binary mask (numpy array)

        Returns:
            List of (x, y) centroid coordinates
        """
        labeled, num_features = ndimage.label(mask)
        centroids = ndimage.center_of_mass(mask, labeled, range(1, num_features + 1))
        # Convert from (row, col) to (x, y)
        return [(int(c[1]), int(c[0])) for c in centroids]

    def extract_crop(self, image, centroid):
        """
        Extract a crop around a centroid.

        Args:
            image: Source image (numpy array, should be grayscale)
            centroid: (x, y) center coordinate

        Returns:
            Cropped image (PIL Image) or None if out of bounds
        """
        x, y = centroid
        half = self.crop_size // 2
        h, w = image.shape[:2]

        # Check bounds
        if x - half < 0 or x + half > w or y - half < 0 or y + half > h:
            return None

        crop = image[y - half:y + half, x - half:x + half]

        # Ensure grayscale, then convert to RGB (to match training data)
        if len(crop.shape) == 3:
            crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        # Convert grayscale to RGB (replicates across channels like ImageFolder does)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(crop_rgb)

    def classify_crop(self, crop):
        """
        Classify a single cell crop.

        Args:
            crop: PIL Image of cell crop

        Returns:
            (prediction, confidence) tuple
        """
        input_tensor = self.transform(crop).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            pred_idx = probs.argmax(dim=1).item()
            confidence = probs[0, pred_idx].item()

        return self.classes[pred_idx], confidence

    def process_image_set(self, brightfield_path, green_path, red_path):
        """
        Process a set of images (brightfield + fluorescence) - CACO2 MODE.

        Args:
            brightfield_path: Path to brightfield image
            green_path: Path to green fluorescence image (live cells)
            red_path: Path to red fluorescence image (dead cells)

        Returns:
            List of dicts with cell info: {cell_id, x, y, prediction, confidence, ground_truth}
        """
        # Load images (brightfield as grayscale to match training data)
        brightfield = cv2.imread(str(brightfield_path), cv2.IMREAD_GRAYSCALE)
        green = np.array(Image.open(green_path))
        red = np.array(Image.open(red_path))

        # Generate masks
        live_mask = self.generate_mask(green)
        dead_mask = self.generate_mask(red)

        # Find centroids with ground truth labels
        live_centroids = [(c, 'live') for c in self.find_cell_centroids(live_mask)]
        dead_centroids = [(c, 'dead') for c in self.find_cell_centroids(dead_mask)]
        all_cells = live_centroids + dead_centroids

        results = []
        for cell_id, (centroid, ground_truth) in enumerate(all_cells, 1):
            crop = self.extract_crop(brightfield, centroid)
            if crop is None:
                continue

            prediction, confidence = self.classify_crop(crop)
            results.append({
                'cell_id': cell_id,
                'x': centroid[0],
                'y': centroid[1],
                'prediction': prediction,
                'confidence': round(confidence, 3),
                'ground_truth': ground_truth
            })

        return results

    def process_brightfield_only(self, brightfield_path, use_cellpose=False):
        """
        Process brightfield image only (no ground truth) - IPSC MODE.

        Args:
            brightfield_path: Path to brightfield image
            use_cellpose: Whether to use Cellpose for segmentation

        Returns:
            List of dicts with cell info
        """
        brightfield = np.array(Image.open(brightfield_path))

        # Convert to grayscale if needed
        if len(brightfield.shape) == 3:
            gray = cv2.cvtColor(brightfield, cv2.COLOR_RGB2GRAY)
        else:
            gray = brightfield

        if use_cellpose:
            # Use Cellpose for better cell segmentation
            try:
                from cellpose import models as cp_models
                cellpose_model = cp_models.Cellpose(model_type='cyto2', gpu=torch.cuda.is_available())
                masks, _, _, _ = cellpose_model.eval(gray, diameter=None, channels=[0, 0])

                # Find centroids from Cellpose masks
                centroids = []
                for i in range(1, masks.max() + 1):
                    ys, xs = np.where(masks == i)
                    if len(xs) > 0:
                        centroids.append((int(xs.mean()), int(ys.mean())))
            except ImportError:
                print("Warning: Cellpose not installed. Using adaptive thresholding.")
                mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 11, 2)
                centroids = self.find_cell_centroids(mask)
        else:
            # Simple adaptive thresholding as fallback
            mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)
            centroids = self.find_cell_centroids(mask)

        results = []
        for cell_id, centroid in enumerate(centroids, 1):
            crop = self.extract_crop(gray, centroid)
            if crop is None:
                continue

            prediction, confidence = self.classify_crop(crop)
            results.append({
                'cell_id': cell_id,
                'x': centroid[0],
                'y': centroid[1],
                'prediction': prediction,
                'confidence': round(confidence, 3),
                'ground_truth': None
            })

        return results

    def visualize_results(self, image_path, results, output_path):
        """
        Create annotated image with colored bounding boxes.

        Args:
            image_path: Path to source image
            results: List of cell result dicts
            output_path: Path to save annotated image
        """
        image = cv2.imread(str(image_path))
        half = self.crop_size // 2

        for cell in results:
            x, y = cell['x'], cell['y']
            color = (0, 255, 0) if cell['prediction'] == 'live' else (0, 0, 255)  # Green=live, Red=dead

            # Draw bounding box
            cv2.rectangle(image, (x - half, y - half), (x + half, y + half), color, 2)

            # Draw label
            label = f"{cell['prediction'][0].upper()} {cell['confidence']:.2f}"
            cv2.putText(image, label, (x - half, y - half - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cv2.imwrite(str(output_path), image)


def find_image_sets_caco2(input_dir):
    """
    Find sets of images following the .0/.1/.2 naming convention (CACO2 MODE).

    Returns:
        List of (base_name, brightfield_path, green_path, red_path) tuples
    """
    input_dir = Path(input_dir)
    image_sets = {}

    for ext in ['*.jpg', '*.png', '*.tif', '*.tiff']:
        for img_path in input_dir.glob(ext):
            name = img_path.stem
            # Check if name ends with .0, .1, or .2
            if '.' in name:
                base, channel = name.rsplit('.', 1)
                if channel in ['0', '1', '2']:
                    if base not in image_sets:
                        image_sets[base] = {}
                    image_sets[base][channel] = img_path

    # Filter to complete sets
    complete_sets = []
    for base, channels in image_sets.items():
        if '0' in channels and '1' in channels and '2' in channels:
            complete_sets.append((base, channels['0'], channels['1'], channels['2']))

    return complete_sets


def find_brightfield_images(input_dir):
    """
    Find all brightfield images in directory (IPSC MODE).

    Returns:
        List of (image_name, image_path) tuples
    """
    input_dir = Path(input_dir)
    images = []

    for ext in ['*.jpg', '*.png', '*.tif', '*.tiff']:
        for img_path in input_dir.glob(ext):
            # Skip files with .0/.1/.2 naming (those are caco2 format)
            name = img_path.stem
            if '.' in name:
                _, suffix = name.rsplit('.', 1)
                if suffix in ['0', '1', '2']:
                    continue
            images.append((img_path.stem, img_path))

    return images


def main():
    parser = argparse.ArgumentParser(description='Live/Dead Cell Classification Pipeline')
    parser.add_argument('--dataset', choices=['caco2', 'ipsc'], default='caco2',
                        help='Dataset/mode to use (default: caco2)')
    parser.add_argument('--input_dir', required=True, help='Directory containing input images')
    parser.add_argument('--output_dir', required=True, help='Directory for output files')
    parser.add_argument('--model', default=None, help='Path to trained model (auto-selected if not specified)')
    parser.add_argument('--visualize', action='store_true', help='Generate annotated images')
    parser.add_argument('--threshold', type=int, default=30, help='Fluorescence threshold (caco2 mode)')
    parser.add_argument('--cellpose', action='store_true', help='Use Cellpose for segmentation (ipsc mode)')
    args = parser.parse_args()

    # Get dataset config
    config = DATASET_CONFIG[args.dataset]

    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find model
    model_path = Path(args.model) if args.model else Path(config['model_path'])
    if not model_path.exists():
        # Try relative to script directory
        script_dir = Path(__file__).parent.parent
        model_path = script_dir / config['model_path']
    if not model_path.exists():
        # Try legacy name (without dataset suffix)
        legacy_path = Path("live_dead_classifier.pth")
        if legacy_path.exists():
            model_path = legacy_path
            print(f"Using legacy model: {model_path}")
        else:
            print(f"Error: Model not found. Expected: {config['model_path']}")
            print(f"Run 'python scripts/train_classifier.py --dataset {args.dataset}' first.")
            sys.exit(1)

    print("=" * 60)
    print(f"Live/Dead Cell Classification Pipeline")
    print("=" * 60)
    print(f"Mode:        {args.dataset} ({config['description']})")
    print(f"Model:       {model_path}")
    print(f"Input:       {input_dir}")
    print(f"Output:      {output_dir}")
    print("=" * 60)

    # Initialize pipeline
    pipeline = LiveDeadPipeline(
        str(model_path),
        crop_size=config['crop_size'],
        resize=config['resize']
    )

    all_results = []

    if args.dataset == 'caco2':
        # CACO2 MODE: Requires .0/.1/.2 image triplets
        image_sets = find_image_sets_caco2(input_dir)
        print(f"Found {len(image_sets)} complete image sets", flush=True)

        if not image_sets:
            print("No complete image sets found. Expected naming: <name>.0.jpg, <name>.1.jpg, <name>.2.jpg")
            sys.exit(1)

        for base_name, bf_path, green_path, red_path in image_sets:
            print(f"Processing {base_name}...", flush=True)

            results = pipeline.process_image_set(bf_path, green_path, red_path)

            for r in results:
                r['image_name'] = base_name
            all_results.extend(results)

            # Visualize if requested
            if args.visualize and results:
                vis_path = output_dir / f"{base_name}_annotated.png"
                pipeline.visualize_results(bf_path, results, vis_path)

    else:
        # IPSC MODE: Brightfield-only images
        images = find_brightfield_images(input_dir)
        print(f"Found {len(images)} brightfield images", flush=True)

        if not images:
            print("No brightfield images found in input directory.")
            sys.exit(1)

        for image_name, image_path in images:
            print(f"Processing {image_name}...", flush=True)

            results = pipeline.process_brightfield_only(image_path, use_cellpose=args.cellpose)

            for r in results:
                r['image_name'] = image_name
            all_results.extend(results)

            # Visualize if requested
            if args.visualize and results:
                vis_path = output_dir / f"{image_name}_annotated.png"
                pipeline.visualize_results(image_path, results, vis_path)

    # Save results
    df = pd.DataFrame(all_results)
    csv_path = output_dir / 'classification_results.csv'
    df.to_csv(csv_path, index=False)

    # Print summary
    print(f"\n{'='*50}")
    print(f"Results saved to {csv_path}")
    print(f"Total cells processed: {len(all_results)}")

    if 'ground_truth' in df.columns and df['ground_truth'].notna().any():
        correct = (df['prediction'] == df['ground_truth']).sum()
        total = len(df[df['ground_truth'].notna()])
        print(f"Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")

        print(f"\nPrediction breakdown:")
        print(df.groupby(['ground_truth', 'prediction']).size().unstack(fill_value=0))
    else:
        print(f"\nPredictions: {df['prediction'].value_counts().to_dict()}")

    if args.visualize:
        print(f"\nAnnotated images saved to {output_dir}")


if __name__ == '__main__':
    main()
