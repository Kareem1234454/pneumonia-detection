# preprocessing.py Documentation Report

## Overview
`preprocessing.py` implements the foundational image processing pipeline necessary to standardize chest X-ray images before they are fed into the machine learning models. High-quality preprocessing is vital for medical imaging to eliminate noise and standardize contrasts across different X-ray machines.

## Key Techniques & Functions
### 1. Contrast Enhancement (`apply_clahe`)
Utilizes Contrast Limited Adaptive Histogram Equalization (CLAHE). Unlike standard global histogram equalization, CLAHE operates on small local regions (tiles) of the image. This prevents the over-amplification of noise in homogenous areas of the lung, revealing hidden vascular and tissue details critical for detecting pneumonia infiltrates.

### 2. Noise Reduction (`remove_noise_medianBlur`, `apply_gaussian_blur`)
Provides filters to smooth the image and remove salt-and-pepper noise, which is common in older or lower-dose digital radiography scans.

### 3. Pipeline Integration
The script provides a unified `preprocess_image` function that reads, resizes, and applies CLAHE in a single call, returning both the enhanced 8-bit image and the `0-1` normalized float array. It also contains `visualize_preprocessing_pipeline`, a helper function to plot a multi-stage comparison (Original -> Resize -> Hist EQ -> CLAHE -> Gaussian -> Median) complete with pixel intensity histograms.
