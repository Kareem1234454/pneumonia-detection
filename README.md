<div align="center">

# 🩺 Pneumonia Detection System

**An AI-powered, end-to-end diagnostic pipeline for detecting pneumonia from chest X-ray images, featuring robust lung segmentation, deep transfer learning, and Grad-CAM explainability.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv)](https://opencv.org/)
[![Gradio](https://img.shields.io/badge/Gradio-UI-red?style=for-the-badge&logo=gradio)](https://gradio.app)

</div>

---

## 📌 Overview

The **Pneumonia Detection System** is an advanced machine learning framework designed to accurately classify chest X-rays into `NORMAL` or `PNEUMONIA` categories. Built to address the "black box" problem in medical AI, this system goes beyond simple classification by physically isolating the lungs via morphological segmentation and visually highlighting pathological regions using Grad-CAM attention maps.

> **Disclaimer:** This software is developed for research and educational purposes. It is not intended for clinical use or to replace professional medical diagnosis.

---

## 🚀 Key Features

* **Advanced Image Preprocessing:** Utilizes Contrast Limited Adaptive Histogram Equalization (CLAHE) to reveal hidden pulmonary details while suppressing hardware noise.
* **Intelligent Lung Segmentation:** Isolates the lung regions using Otsu's Thresholding and morphological contouring, forcing the neural network to ignore irrelevant background artifacts (bones, medical tubing).
* **Deep Transfer Learning:** Employs a finely-tuned VGG16 architecture to extract high-level representations, overcoming the limitations of small medical datasets.
* **Clinical Explainability (Grad-CAM):** Generates thermal heatmaps that explicitly pinpoint the regions driving the model's diagnostic decisions, fostering trust and interpretability.
* **Premium Web Interface:** Includes a fully responsive, dark-mode Gradio web application for side-by-side verification of preprocessing, segmentation, and Grad-CAM outputs.

---

## 🏗️ System Architecture

1. **Data Ingestion:** Automatically scales and normalizes Kaggle's Chest X-Ray dataset, balancing classes dynamically during training.
2. **Computer Vision Pipeline:** 
   `Raw X-Ray -> CLAHE Enhancement -> Binarization -> Lung Mask Generation -> Isolated Lung Crop`
3. **Deep Learning Engine:** 
   `Pre-trained VGG16 Base -> Global Average Pooling -> Dense Block (Dropout + ReLU) -> Sigmoid Output`
4. **Explainability & UI:** Computes class activation gradients from the final convolutional block and overlays them onto the segmented lungs within a user-friendly web dashboard.

---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Kareem1234454/pneumonia-detection.git
cd pneumonia-detection
```

### 2. Set Up the Environment
It is recommended to use a virtual environment or Conda:
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Preparation

The system expects the [Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) hosted on Kaggle.

Download and extract the dataset into the `data/raw/` directory:
```text
data/
└── raw/
    ├── train/
    ├── val/
    └── test/
```

---

## 💻 Usage & Commands

The entire pipeline is managed via the `main.py` command-line interface.

### 🏥 Launch the Web Application (Recommended)
To start the interactive Gradio dashboard:
```bash
python main.py --mode demo
```
*The app will automatically run on `http://127.0.0.1:7860/`.*

### 🧠 Train the Model
To initiate the two-phase VGG16 transfer learning process (frozen base training followed by fine-tuning):
```bash
python main.py --mode train_vgg
```

### 📈 Evaluate Performance
To evaluate the best-saved model on the test dataset and generate a Confusion Matrix and ROC curve:
```bash
python main.py --mode evaluate
```

### 🔍 Run the Single-Image Pipeline
To test the pipeline via the CLI on a single image (generates and saves figures locally):
```bash
python main.py --mode pipeline --image path/to/your/xray.jpeg
```

---

## 📁 Project Structure

```text
Pneumonia_Detection_System/
├── app/
│   └── app.py                  # Gradio web interface and UI components
├── data/                       # Raw and processed dataset storage
├── models/                     # Saved h5 weights (vgg16_best.h5, etc.)
├── outputs/
│   ├── figures/                # Saved heatmaps, ROC curves, confusion matrices
│   └── results/                # CSV metrics reports
├── report/                     # Comprehensive system and file-level documentation
├── src/
│   ├── dataset.py              # Data generators and splits
│   ├── evaluate.py             # Performance metrics and visual diagnostics
│   ├── gradcam.py              # Explainability and attention heatmap logic
│   ├── preprocessing.py        # CLAHE and noise reduction algorithms
│   ├── segmentation.py         # Lung isolation and contouring
│   └── transfer_learning.py    # VGG16 architecture and training loops
├── config.py                   # Global hyperparameters, paths, and constants
├── main.py                     # Central CLI entry point
└── requirements.txt            # Python dependencies
```

---

## 📜 License
This project is open-source and available under the **MIT License**.

<div align="center">
<i>Architected for academic excellence and clinical interpretability.</i>
</div>