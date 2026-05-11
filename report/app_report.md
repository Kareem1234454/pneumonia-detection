# app.py Documentation Report

## Overview
`app.py` contains the logic for the interactive web application built using the Gradio framework. It bridges the gap between the complex deep learning backend and end-users (e.g., medical professionals), providing an intuitive, aesthetically pleasing interface for running pneumonia diagnostics.

## Architecture & Workflow
### 1. Model Loading (`get_model`)
Implements a lazy-loading singleton pattern. The model (preferring the fine-tuned VGG16) is loaded into a global `_MODEL` variable only once during startup to drastically reduce prediction latency.

### 2. Prediction Pipeline (`predict`)
When a user uploads an image, the `predict` function acts as the controller:
- Extracts the raw image.
- Passes the image through the custom `gradcam_single` pipeline (which internally preprocesses, segments, predicts, and generates heatmaps).
- Generates side-by-side outputs: The preprocessed image, the cleanly segmented lungs, and the Grad-CAM heatmap.
- Dynamically formats the prediction result text (red for Pneumonia, green for Normal) along with the confidence score.

### 3. UI Design (`build_interface`)
Constructs a premium layout using `gr.Blocks`. It integrates custom CSS for a dark-mode "glassmorphism" aesthetic, complete with styled headers, structured columns, and hover-animated panels. It handles the input/output mappings between the user's uploaded file and the Python backend.
