# dataset.py Documentation Report

## Overview
`dataset.py` is responsible for handling data loading, splitting, and augmenting the raw Kaggle dataset. (Note: Much of the augmentation logic utilized for VGG16 was natively integrated into `transfer_learning.py` using `ImageDataGenerator`, making `dataset.py` act as the foundation for the custom CNN approach).

## Key Concepts
- **Data Iteration**: Medical imaging datasets are too large to fit entirely into RAM. This script defines structures to yield batches of images on the fly.
- **Directory Parsing**: It maps the `data/raw/` or `data/processed/` directories, identifying images belonging to the `NORMAL` and `PNEUMONIA` classes.
- **Train/Val/Test Splits**: Ensures the model is trained, validated, and tested on strictly independent sets of images to prevent data leakage and provide an accurate measure of generalization.
