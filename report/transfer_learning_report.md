# transfer_learning.py Documentation Report

## Overview
`transfer_learning.py` forms the core deep learning architecture of the system. Instead of training a Convolutional Neural Network (CNN) from scratch on a relatively small medical dataset, this module utilizes the VGG16 model, pre-trained on ImageNet, to leverage existing feature-extraction capabilities.

## Architecture & Data Flow
### 1. Data Augmentation & Generators
The script defines heavy image augmentation pipelines using `ImageDataGenerator`. It applies rotation, shift, zoom, and horizontal flipping specifically tailored differently for `NORMAL` and `PNEUMONIA` classes. Crucially, it calculates class weights dynamically to counteract the heavy class imbalance present in the Kaggle dataset.

### 2. Model Construction (`build_vgg16_model`)
Imports the base VGG16 architecture (omitting the top dense layers). It appends a custom classification head consisting of Global Average Pooling, a 256-unit Dense layer (ReLU), Dropout for regularization, and a final 1-unit Dense layer with a Sigmoid activation for binary classification.

### 3. Two-Phase Training Strategy (`train_vgg16`)
- **Phase 1 (Feature Extraction)**: The entire VGG16 base is frozen. Only the custom dense head is trained with a higher learning rate (1e-3) to adapt it to the lung feature maps.
- **Phase 2 (Fine-Tuning)**: The last convolutional block (top 4 layers) of the VGG16 base is unfrozen. The model is trained at a significantly lower learning rate (1e-5) to gently adjust the deep feature representations specifically to the domain of chest X-rays.

Both phases utilize `EarlyStopping` to prevent overfitting and `ModelCheckpoint` to save the best weights to the `models/` directory.
