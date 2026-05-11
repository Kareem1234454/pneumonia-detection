# main.py Documentation Report

## Overview
`main.py` serves as the primary command-line interface (CLI) and entry point for the Pneumonia Detection System. It orchestrates the entire deep learning pipeline, allowing users to train models, evaluate them, run inference on single images, and launch the web interface.

## Key Functions & Modes
The script uses `argparse` to define four primary operational modes:
1. **`--mode train_vgg`**: Calls `train_vgg16()` from `src.transfer_learning`. This initiates the two-phase training process (frozen base training followed by fine-tuning) for the VGG16 model.
2. **`--mode evaluate`**: Loads the best available trained model and executes `evaluate_model()` from `src.evaluate` to generate performance metrics (Accuracy, ROC curve, Confusion Matrix).
3. **`--mode pipeline`**: Runs a full diagnostic pipeline on a single image provided via `--image`. It sequentially executes preprocessing, lung segmentation, prediction, and Grad-CAM explainability, saving the visualization figures to disk.
4. **`--mode demo`**: Launches the interactive Gradio web application defined in `app/app.py`, mapping the deep learning backend to a user-friendly UI.

## Execution Flow
The script relies heavily on modular imports. When run, it parses the arguments, determines the requested mode, and then calls the appropriate handler function. It effectively decouples the user interface (CLI/Web) from the core machine learning logic.
