#!/usr/bin/env python3
"""
Live/Dead Cell Classifier Training

Trains a ResNet-18 classifier on cell crops to distinguish live vs dead cells.

Supports two datasets:
  - caco2: Original caco2 dataset (32x32 crops from fluorescence-labeled data)
  - ipsc:  ethz_iPSC dataset (100x100 crops, morphology-based labels)

Usage:
    python scripts/train_classifier.py --dataset caco2   # default
    python scripts/train_classifier.py --dataset ipsc
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision import transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# === ARGUMENT PARSING ===
parser = argparse.ArgumentParser(description='Train Live/Dead Cell Classifier')
parser.add_argument('--dataset', choices=['caco2', 'ipsc'], default='caco2',
                    help='Dataset to train on (default: caco2)')
parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
args = parser.parse_args()

# === DATASET CONFIG ===
DATASET_CONFIG = {
    'caco2': {
        'data_dir': 'data/cell_crops',
        'crop_size': 32,
        'resize': 64,
        'class_weights': [1.0, 1.24],  # Adjust for class imbalance
        'description': 'Caco2 fluorescence-labeled cell crops'
    },
    'ipsc': {
        'data_dir': 'data/ipsc_cell_crops',
        'crop_size': 100,
        'resize': 100,  # No resize needed, already good size
        'class_weights': [1.0, 1.07],  # ~1365 live vs ~1461 dead
        'description': 'iPSC morphology-based cell crops'
    }
}

config = DATASET_CONFIG[args.dataset]

# === CONFIG ===
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr
USE_PRETRAINED = True
DATA_DIR = config['data_dir']
RESIZE = config['resize']
MODEL_OUTPUT = f"live_dead_classifier_{args.dataset}.pth"

# ImageNet normalization (required for pretrained weights)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

print("=" * 60)
print(f"Training Live/Dead Classifier")
print("=" * 60)
print(f"Dataset:     {args.dataset} ({config['description']})")
print(f"Data dir:    {DATA_DIR}")
print(f"Image size:  {RESIZE}x{RESIZE}")
print(f"Epochs:      {EPOCHS}")
print(f"Batch size:  {BATCH_SIZE}")
print(f"Output:      {MODEL_OUTPUT}")
print("=" * 60)

# Data transforms with augmentation
transform_train = transforms.Compose([
    transforms.Resize((RESIZE, RESIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

transform_val = transforms.Compose([
    transforms.Resize((RESIZE, RESIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Load dataset
print("\nLoading dataset...", flush=True)
if not os.path.exists(DATA_DIR):
    print(f"Error: Data directory not found: {DATA_DIR}")
    if args.dataset == 'ipsc':
        print("Run 'python scripts/convert_ipsc_dataset.py' first to create iPSC crops.")
    sys.exit(1)

full_dataset = ImageFolder(DATA_DIR, transform=transform_train)
print(f"Found {len(full_dataset)} images in {len(full_dataset.classes)} classes: {full_dataset.classes}", flush=True)

# Split
train_len = int(0.8 * len(full_dataset))
val_len = len(full_dataset) - train_len
train_ds, val_ds = random_split(full_dataset, [train_len, val_len])

# Apply validation transform to val set
val_ds.dataset.transform = transform_val

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)

# Setup model with pretrained weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

if USE_PRETRAINED:
    print("Loading pretrained ImageNet weights...", flush=True)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Freeze early layers (optional - can unfreeze for fine-tuning)
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze last layers for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True
else:
    model = models.resnet18(weights=None)

# Replace final layer
model.fc = nn.Linear(512, 2)
model.to(device)

# Class weights for imbalance
class_weights = torch.tensor(config['class_weights']).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Optimizer (only train unfrozen params)
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(trainable_params, lr=LEARNING_RATE)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# Training loop with validation
best_val_acc = 0
print(f"\nStarting training for {EPOCHS} epochs...\n", flush=True)

for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += (out.argmax(1) == y).sum().item()
        train_total += y.size(0)

    train_acc = 100 * train_correct / train_total

    # Validate
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            val_loss += loss.item()
            val_correct += (out.argmax(1) == y).sum().item()
            val_total += y.size(0)

    val_acc = 100 * val_correct / val_total
    avg_val_loss = val_loss / len(val_loader)

    # Update scheduler
    scheduler.step(avg_val_loss)

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_OUTPUT)

    print(f"Epoch {epoch+1:2d}/{EPOCHS} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}% | Val Loss: {avg_val_loss:.4f}", flush=True)

print(f"\nBest validation accuracy: {best_val_acc:.1f}%", flush=True)

# Load best model for final evaluation
model.load_state_dict(torch.load(MODEL_OUTPUT, weights_only=True))
model.eval()

# Final evaluation
all_preds, all_labels = [], []
with torch.no_grad():
    for x, y in val_loader:
        out = model(x.to(device))
        preds = out.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())

# Report
print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)
print(f"Dataset: {args.dataset}")
print(f"Classes: {full_dataset.classes}")
print(classification_report(all_labels, all_preds, target_names=full_dataset.classes))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=full_dataset.classes, yticklabels=full_dataset.classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix - {args.dataset} (Val Acc: {best_val_acc:.1f}%)")
plt.tight_layout()
confusion_matrix_file = f"confusion_matrix_{args.dataset}.png"
plt.savefig(confusion_matrix_file, dpi=150)
print(f"\nConfusion matrix saved to {confusion_matrix_file}", flush=True)
print(f"Model saved to {MODEL_OUTPUT}", flush=True)
