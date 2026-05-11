# segmentation.py Documentation Report

## Overview
`segmentation.py` leverages classical Computer Vision morphology to physically isolate the lungs from the surrounding anatomy (heart, ribs, diaphragm, and background). By masking out irrelevant pixels, the model is forced to focus strictly on the pulmonary area, reducing false positives.

## Key Techniques & Functions
### 1. Binarization & Thresholding (`otsu_threshold`)
Uses Otsu's Thresholding algorithm to automatically calculate the optimal threshold value separating the dark lungs (background/air) from the bright surrounding tissues (bones/flesh).

### 2. Morphological Operations
The script uses `cv2.morphologyEx` (opening and closing) to clean the binary mask. Opening removes small noise artifacts, while closing fills in gaps or "holes" inside the lung regions caused by dense pneumonia opacities or blood vessels.

### 3. Contour Extraction & Masking (`segment_lungs`)
The core `segment_lungs` function limits operations to a central Region of Interest (ROI) to avoid border artifacts. It extracts external contours and isolates the two largest contours (representing the left and right lungs). It then generates a binary mask and applies a bitwise AND operation against the original image to output a clean, background-removed representation of the lungs.
