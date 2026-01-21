#!/usr/bin/env python3
"""
End-to-end Live/Dead Cell Classification Pipeline

Input: Directory with images following naming convention:
  - *.0.jpg (or .png) = brightfield
  - *.1.jpg (or .png) = green fluorescence (AO stain = live)
  - *.2.jpg (or .png) = red fluorescence (PI stain = dead)

Output:
  - CSV with cell locations and predictions
  - Annotated images with colored bounding boxes (optional)

Usage:
  python run_pipeline.py --input_dir <path> --output_dir <path> [--visualize]
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


class LiveDeadPipeline:
    """End-to-end pipeline for live/dead cell classification."""

    def __init__(self, model_path, device=None):
        """
        Initialize the pipeline.

        Args:
            model_path: Path to trained live_dead_classifier.pth
            device: torch device (auto-detected if None)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        # Must match training transforms (ImageNet normalization for pretrained weights)
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.crop_size = 32
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
        Process a set of images (brightfield + fluorescence).

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

    def process_brightfield_only(self, brightfield_path, mask_path=None):
        """
        Process brightfield image only (no ground truth).
        Uses provided mask or requires cells to be pre-segmented.

        Args:
            brightfield_path: Path to brightfield image
            mask_path: Optional path to pre-computed cell mask

        Returns:
            List of dicts with cell info
        """
        brightfield = np.array(Image.open(brightfield_path))

        if mask_path:
            mask = np.array(Image.open(mask_path).convert('L'))
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        else:
            # Simple edge-based detection as fallback
            gray = cv2.cvtColor(brightfield, cv2.COLOR_RGB2GRAY) if len(brightfield.shape) == 3 else brightfield
            mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        centroids = self.find_cell_centroids(mask)

        results = []
        for cell_id, centroid in enumerate(centroids, 1):
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


def find_image_sets(input_dir):
    """
    Find sets of images following the naming convention.

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


def main():
    parser = argparse.ArgumentParser(description='Live/Dead Cell Classification Pipeline')
    parser.add_argument('--input_dir', required=True, help='Directory containing input images')
    parser.add_argument('--output_dir', required=True, help='Directory for output files')
    parser.add_argument('--model', default='live_dead_classifier.pth', help='Path to trained model')
    parser.add_argument('--visualize', action='store_true', help='Generate annotated images')
    parser.add_argument('--threshold', type=int, default=30, help='Fluorescence threshold')
    args = parser.parse_args()

    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find model
    model_path = Path(args.model)
    if not model_path.exists():
        # Try relative to script directory
        script_dir = Path(__file__).parent.parent
        model_path = script_dir / args.model
    if not model_path.exists():
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)

    print(f"Loading model from {model_path}...", flush=True)
    pipeline = LiveDeadPipeline(str(model_path))

    # Find image sets
    image_sets = find_image_sets(input_dir)
    print(f"Found {len(image_sets)} complete image sets", flush=True)

    if not image_sets:
        print("No complete image sets found. Expected naming: <name>.0.jpg, <name>.1.jpg, <name>.2.jpg")
        sys.exit(1)

    # Process each set
    all_results = []
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
        total = len(df)
        print(f"Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")

        print(f"\nPrediction breakdown:")
        print(df.groupby(['ground_truth', 'prediction']).size().unstack(fill_value=0))
    else:
        print(f"\nPredictions: {df['prediction'].value_counts().to_dict()}")

    if args.visualize:
        print(f"\nAnnotated images saved to {output_dir}")


if __name__ == '__main__':
    main()
