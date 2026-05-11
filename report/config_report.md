# config.py Documentation Report

## Overview
`config.py` acts as the central repository for all global settings, hyperparameters, file paths, and constants used throughout the Pneumonia Detection System. Centralizing these configurations ensures consistency and allows for easy tuning without modifying the core logic scripts.

## Key Configurations
### 1. Directory Paths
It dynamically defines absolute paths using `os.path.abspath(__file__)` to map directories for raw data, processed data, models, and outputs (figures and results).

### 2. Image Processing Settings
Defines constants used by the Computer Vision pipeline:
- `IMG_SIZE` / `VGG_IMG_SIZE`: Set to `(224, 224)` to match VGG16 expectations.
- `CLAHE_CLIP` and `CLAHE_GRID`: Parameters for Contrast Limited Adaptive Histogram Equalization.
- `MEDIAN_BLUR`: Kernel size for noise reduction.

### 3. Training & Model Hyperparameters
- **Training**: `BATCH_SIZE = 32`, `EPOCHS = 20`, `LEARNING_RATE = 0.001`, `PATIENCE = 5`.
- **Model**: Defines model save paths (`best_model.h5`, `final_model.h5`), dropout rates, and the expected class names (`["NORMAL", "PNEUMONIA"]`).

### 4. App & Explainability Settings
- **Grad-CAM**: Specifies the target layer for gradients (`GRADCAM_LAYER`) and overlay opacity (`GRADCAM_ALPHA`).
- **Gradio App**: Defines the `APP_TITLE`, description, and port numbers.
