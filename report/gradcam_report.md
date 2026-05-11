# gradcam.py Documentation Report

## Overview
`gradcam.py` implements Gradient-weighted Class Activation Mapping (Grad-CAM), transforming the "black box" deep learning model into an interpretable system. It provides visual explanations for the AI's diagnostic decisions, which is critical for establishing trust with medical practitioners.

## Technical Implementation
### 1. Dynamic Layer Extraction
The module contains logic (`get_gradcam_models`) to dynamically inspect the Keras model architecture. It traverses nested structures (like the `vgg_base` within a `Sequential` model) to isolate the final convolutional layer (`block5_conv3`), which contains the highest-level spatial feature maps before they are flattened.

### 2. Gradient Calculation
Using `tf.GradientTape`, the script computes the gradient of the predicted class score with respect to the feature map activations of the target layer.

### 3. Heatmap Generation
- Performs Global Average Pooling over the spatial dimensions of the gradients to compute "importance weights" for each channel.
- Computes a weighted sum of the feature maps using these weights.
- Applies a ReLU activation to pass only features that have a positive influence on the target class.

### 4. Image Overlay (`create_gradcam_overlay`)
The resulting low-resolution heatmap is normalized, up-sampled using OpenCV, mapped to a JET color space (where red/white indicates high focus), and cleanly blended over the segmented lung image. This output is directly piped into the Gradio web interface.
