#!/usr/bin/env python3
"""
Live/Dead Cell Classification GUI

A standalone GUI for running the live/dead cell classification pipeline.
Supports both caco2 and iPSC datasets.

Usage:
    python live_dead_gui.py
"""

import sys
import os
from pathlib import Path

from qtpy.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QPushButton, QLabel, QLineEdit, QFileDialog,
    QProgressBar, QTextEdit, QGroupBox, QCheckBox, QSpinBox,
    QMessageBox, QSplitter, QScrollArea, QComboBox
)
from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtGui import QPixmap, QFont, QImage

import numpy as np
from PIL import Image

# Import pipeline components
from run_pipeline import (
    LiveDeadPipeline,
    find_image_sets_caco2,
    find_brightfield_images,
    DATASET_CONFIG
)


class PipelineWorker(QThread):
    """Worker thread for running pipeline without blocking GUI."""

    progress = Signal(str)  # Status message
    progress_value = Signal(int)  # Progress bar value
    finished = Signal(object)  # Results dataframe
    error = Signal(str)  # Error message

    def __init__(self, pipeline, input_dir, output_dir, dataset_mode, visualize=True, use_cellpose=False):
        super().__init__()
        self.pipeline = pipeline
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.dataset_mode = dataset_mode
        self.visualize = visualize
        self.use_cellpose = use_cellpose

    def run(self):
        try:
            import pandas as pd

            all_results = []

            if self.dataset_mode == 'caco2':
                # CACO2 MODE: requires .0/.1/.2 image triplets
                image_sets = find_image_sets_caco2(self.input_dir)
                if not image_sets:
                    self.error.emit("No complete image sets found. Expected: <name>.0.jpg, <name>.1.jpg, <name>.2.jpg")
                    return

                self.progress.emit(f"Found {len(image_sets)} image sets (caco2 mode)")

                for i, (base_name, bf_path, green_path, red_path) in enumerate(image_sets):
                    self.progress.emit(f"Processing {base_name}...")
                    self.progress_value.emit(int(100 * i / len(image_sets)))

                    results = self.pipeline.process_image_set(bf_path, green_path, red_path)

                    for r in results:
                        r['image_name'] = base_name
                    all_results.extend(results)

                    # Visualize
                    if self.visualize and results:
                        vis_path = Path(self.output_dir) / f"{base_name}_annotated.png"
                        self.pipeline.visualize_results(bf_path, results, vis_path)

            else:
                # IPSC MODE: brightfield-only images
                images = find_brightfield_images(self.input_dir)
                if not images:
                    self.error.emit("No brightfield images found in input directory.")
                    return

                self.progress.emit(f"Found {len(images)} brightfield images (ipsc mode)")

                for i, (image_name, image_path) in enumerate(images):
                    self.progress.emit(f"Processing {image_name}...")
                    self.progress_value.emit(int(100 * i / len(images)))

                    results = self.pipeline.process_brightfield_only(image_path, use_cellpose=self.use_cellpose)

                    for r in results:
                        r['image_name'] = image_name
                    all_results.extend(results)

                    # Visualize
                    if self.visualize and results:
                        vis_path = Path(self.output_dir) / f"{image_name}_annotated.png"
                        self.pipeline.visualize_results(image_path, results, vis_path)

            # Save CSV
            df = pd.DataFrame(all_results)
            csv_path = Path(self.output_dir) / 'classification_results.csv'
            df.to_csv(csv_path, index=False)

            self.progress_value.emit(100)
            self.progress.emit(f"Complete! Processed {len(all_results)} cells")
            self.finished.emit(df)

        except Exception as e:
            self.error.emit(str(e))


class LiveDeadGUI(QMainWindow):
    """Main GUI window for live/dead cell classification."""

    def __init__(self):
        super().__init__()
        self.pipeline = None
        self.results_df = None
        self.current_annotated_image = None

        self.init_ui()
        self.load_default_model()

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Live/Dead Cell Classifier")
        self.setMinimumSize(900, 750)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Title
        title = QLabel("Live/Dead Cell Classification")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        # Splitter for left/right panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel - Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Dataset selection section
        dataset_group = QGroupBox("Dataset Mode")
        dataset_layout = QGridLayout(dataset_group)

        self.dataset_combo = QComboBox()
        self.dataset_combo.addItem("caco2 - Fluorescence labeled (.0/.1/.2 files)", "caco2")
        self.dataset_combo.addItem("ipsc - Brightfield only (morphology-based)", "ipsc")
        self.dataset_combo.currentIndexChanged.connect(self.on_dataset_changed)
        dataset_layout.addWidget(QLabel("Mode:"), 0, 0)
        dataset_layout.addWidget(self.dataset_combo, 0, 1)

        self.dataset_info = QLabel("Requires: <name>.0.jpg, <name>.1.jpg, <name>.2.jpg")
        self.dataset_info.setStyleSheet("color: #666; font-style: italic;")
        dataset_layout.addWidget(self.dataset_info, 1, 0, 1, 2)

        left_layout.addWidget(dataset_group)

        # Model section
        model_group = QGroupBox("Model")
        model_layout = QGridLayout(model_group)

        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Path to classifier model (.pth)")
        model_layout.addWidget(QLabel("Model:"), 0, 0)
        model_layout.addWidget(self.model_path_edit, 0, 1)

        model_browse_btn = QPushButton("Browse")
        model_browse_btn.clicked.connect(self.browse_model)
        model_layout.addWidget(model_browse_btn, 0, 2)

        self.model_status = QLabel("No model loaded")
        self.model_status.setStyleSheet("color: red;")
        model_layout.addWidget(self.model_status, 1, 0, 1, 3)

        left_layout.addWidget(model_group)

        # Input/Output section
        io_group = QGroupBox("Input / Output")
        io_layout = QGridLayout(io_group)

        self.input_dir_edit = QLineEdit()
        self.input_dir_edit.setPlaceholderText("Directory with image sets")
        io_layout.addWidget(QLabel("Input:"), 0, 0)
        io_layout.addWidget(self.input_dir_edit, 0, 1)

        input_browse_btn = QPushButton("Browse")
        input_browse_btn.clicked.connect(self.browse_input)
        io_layout.addWidget(input_browse_btn, 0, 2)

        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Output directory for results")
        io_layout.addWidget(QLabel("Output:"), 1, 0)
        io_layout.addWidget(self.output_dir_edit, 1, 1)

        output_browse_btn = QPushButton("Browse")
        output_browse_btn.clicked.connect(self.browse_output)
        io_layout.addWidget(output_browse_btn, 1, 2)

        left_layout.addWidget(io_group)

        # Options section
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout(options_group)

        self.visualize_check = QCheckBox("Generate annotated images")
        self.visualize_check.setChecked(True)
        options_layout.addWidget(self.visualize_check)

        # Caco2-specific options
        self.caco2_options = QWidget()
        caco2_layout = QHBoxLayout(self.caco2_options)
        caco2_layout.setContentsMargins(0, 0, 0, 0)
        caco2_layout.addWidget(QLabel("Fluorescence threshold:"))
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(1, 255)
        self.threshold_spin.setValue(30)
        caco2_layout.addWidget(self.threshold_spin)
        caco2_layout.addStretch()
        options_layout.addWidget(self.caco2_options)

        # iPSC-specific options
        self.ipsc_options = QWidget()
        ipsc_layout = QVBoxLayout(self.ipsc_options)
        ipsc_layout.setContentsMargins(0, 0, 0, 0)
        self.cellpose_check = QCheckBox("Use Cellpose for segmentation (better but slower)")
        self.cellpose_check.setChecked(False)
        ipsc_layout.addWidget(self.cellpose_check)
        self.ipsc_options.setVisible(False)  # Hidden by default
        options_layout.addWidget(self.ipsc_options)

        left_layout.addWidget(options_group)

        # Run button
        self.run_btn = QPushButton("Run Classification")
        self.run_btn.setFont(QFont("Arial", 12, QFont.Bold))
        self.run_btn.setMinimumHeight(50)
        self.run_btn.clicked.connect(self.run_pipeline)
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        left_layout.addWidget(self.run_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        left_layout.addWidget(self.progress_bar)

        # Log output
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)
        left_layout.addWidget(log_group)

        left_layout.addStretch()
        splitter.addWidget(left_panel)

        # Right panel - Results
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Results summary
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)

        self.results_label = QLabel("No results yet")
        self.results_label.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(self.results_label)

        # Image preview
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 400)
        self.image_label.setStyleSheet("border: 1px solid #ccc; background: #f0f0f0;")

        scroll = QScrollArea()
        scroll.setWidget(self.image_label)
        scroll.setWidgetResizable(True)
        results_layout.addWidget(scroll)

        # Image navigation
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("< Previous")
        self.prev_btn.clicked.connect(self.show_prev_image)
        self.prev_btn.setEnabled(False)
        nav_layout.addWidget(self.prev_btn)

        self.image_name_label = QLabel("")
        self.image_name_label.setAlignment(Qt.AlignCenter)
        nav_layout.addWidget(self.image_name_label)

        self.next_btn = QPushButton("Next >")
        self.next_btn.clicked.connect(self.show_next_image)
        self.next_btn.setEnabled(False)
        nav_layout.addWidget(self.next_btn)

        results_layout.addLayout(nav_layout)

        # Open output folder button
        self.open_folder_btn = QPushButton("Open Output Folder")
        self.open_folder_btn.clicked.connect(self.open_output_folder)
        self.open_folder_btn.setEnabled(False)
        results_layout.addWidget(self.open_folder_btn)

        right_layout.addWidget(results_group)
        splitter.addWidget(right_panel)

        # Set splitter sizes
        splitter.setSizes([400, 500])

    def on_dataset_changed(self, index):
        """Handle dataset mode change."""
        dataset = self.dataset_combo.currentData()

        if dataset == 'caco2':
            self.dataset_info.setText("Requires: <name>.0.jpg, <name>.1.jpg, <name>.2.jpg")
            self.caco2_options.setVisible(True)
            self.ipsc_options.setVisible(False)
        else:
            self.dataset_info.setText("Accepts: Any brightfield images (.jpg, .png, .tif)")
            self.caco2_options.setVisible(False)
            self.ipsc_options.setVisible(True)

        # Reload model for new dataset
        self.load_default_model()

    def log(self, message):
        """Add message to log."""
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def load_default_model(self):
        """Try to load the default model for current dataset."""
        dataset = self.dataset_combo.currentData()
        config = DATASET_CONFIG[dataset]

        # Look for model in common locations
        script_dir = Path(__file__).parent
        possible_paths = [
            script_dir.parent / config['model_path'],
            script_dir / config['model_path'],
            Path(config['model_path']),
            # Legacy fallback
            script_dir.parent / "live_dead_classifier.pth",
            Path("live_dead_classifier.pth"),
        ]

        for path in possible_paths:
            if path.exists():
                self.model_path_edit.setText(str(path))
                self.load_model(str(path), dataset)
                return

        # No model found
        self.model_status.setText(f"Model not found: {config['model_path']}")
        self.model_status.setStyleSheet("color: red;")
        self.pipeline = None

    def load_model(self, path, dataset=None):
        """Load the classification model."""
        if dataset is None:
            dataset = self.dataset_combo.currentData()

        config = DATASET_CONFIG[dataset]

        try:
            self.pipeline = LiveDeadPipeline(
                path,
                crop_size=config['crop_size'],
                resize=config['resize']
            )
            self.model_status.setText(f"Model loaded ({dataset} mode)")
            self.model_status.setStyleSheet("color: green;")
            self.log(f"Loaded model from {path}")
        except Exception as e:
            self.model_status.setText(f"Error: {str(e)}")
            self.model_status.setStyleSheet("color: red;")
            self.pipeline = None

    def browse_model(self):
        """Browse for model file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model", "", "PyTorch Model (*.pth);;All Files (*)"
        )
        if path:
            self.model_path_edit.setText(path)
            self.load_model(path)

    def browse_input(self):
        """Browse for input directory."""
        path = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if path:
            self.input_dir_edit.setText(path)

    def browse_output(self):
        """Browse for output directory."""
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.output_dir_edit.setText(path)

    def run_pipeline(self):
        """Run the classification pipeline."""
        # Validate inputs
        if not self.pipeline:
            QMessageBox.warning(self, "Error", "Please load a model first")
            return

        input_dir = self.input_dir_edit.text()
        if not input_dir or not Path(input_dir).exists():
            QMessageBox.warning(self, "Error", "Please select a valid input directory")
            return

        output_dir = self.output_dir_edit.text()
        if not output_dir:
            QMessageBox.warning(self, "Error", "Please select an output directory")
            return

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Disable controls
        self.run_btn.setEnabled(False)
        self.progress_bar.setValue(0)

        dataset = self.dataset_combo.currentData()
        self.log(f"Starting pipeline ({dataset} mode)...")

        # Run in worker thread
        self.worker = PipelineWorker(
            self.pipeline,
            input_dir,
            output_dir,
            dataset_mode=dataset,
            visualize=self.visualize_check.isChecked(),
            use_cellpose=self.cellpose_check.isChecked() if dataset == 'ipsc' else False
        )
        self.worker.progress.connect(self.log)
        self.worker.progress_value.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self.on_pipeline_finished)
        self.worker.error.connect(self.on_pipeline_error)
        self.worker.start()

    def on_pipeline_finished(self, df):
        """Handle pipeline completion."""
        self.run_btn.setEnabled(True)
        self.results_df = df
        self.open_folder_btn.setEnabled(True)

        # Show summary
        total = len(df)
        if 'ground_truth' in df.columns and df['ground_truth'].notna().any():
            correct = (df['prediction'] == df['ground_truth']).sum()
            accuracy = 100 * correct / total if total > 0 else 0
            summary = f"Processed {total} cells\nAccuracy: {correct}/{total} ({accuracy:.1f}%)"
        else:
            live_count = (df['prediction'] == 'live').sum()
            dead_count = (df['prediction'] == 'dead').sum()
            summary = f"Processed {total} cells\nLive: {live_count} | Dead: {dead_count}"

        self.results_label.setText(summary)

        # Load annotated images for preview
        self.load_annotated_images()

    def on_pipeline_error(self, error_msg):
        """Handle pipeline error."""
        self.run_btn.setEnabled(True)
        self.log(f"Error: {error_msg}")
        QMessageBox.critical(self, "Error", error_msg)

    def load_annotated_images(self):
        """Load annotated images for preview."""
        output_dir = Path(self.output_dir_edit.text())
        self.annotated_images = list(output_dir.glob("*_annotated.png"))

        if self.annotated_images:
            self.current_image_idx = 0
            self.show_image(0)
            self.prev_btn.setEnabled(len(self.annotated_images) > 1)
            self.next_btn.setEnabled(len(self.annotated_images) > 1)
        else:
            self.image_label.setText("No annotated images")
            self.image_name_label.setText("")

    def show_image(self, idx):
        """Display annotated image at index."""
        if 0 <= idx < len(self.annotated_images):
            self.current_image_idx = idx
            path = self.annotated_images[idx]

            pixmap = QPixmap(str(path))
            scaled = pixmap.scaled(
                self.image_label.size() * 0.95,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled)
            self.image_name_label.setText(f"{path.stem} ({idx + 1}/{len(self.annotated_images)})")

    def show_prev_image(self):
        """Show previous annotated image."""
        if hasattr(self, 'current_image_idx'):
            new_idx = (self.current_image_idx - 1) % len(self.annotated_images)
            self.show_image(new_idx)

    def show_next_image(self):
        """Show next annotated image."""
        if hasattr(self, 'current_image_idx'):
            new_idx = (self.current_image_idx + 1) % len(self.annotated_images)
            self.show_image(new_idx)

    def open_output_folder(self):
        """Open the output folder in file explorer."""
        output_dir = self.output_dir_edit.text()
        if output_dir and Path(output_dir).exists():
            import subprocess
            if sys.platform == 'darwin':
                subprocess.run(['open', output_dir])
            elif sys.platform == 'win32':
                subprocess.run(['explorer', output_dir])
            else:
                subprocess.run(['xdg-open', output_dir])


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = LiveDeadGUI()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
