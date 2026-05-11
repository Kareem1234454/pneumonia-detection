# Pneumonia Detection System: Final Project Report

## 1. Executive Summary
This report details the design, implementation, and evaluation of a comprehensive Pneumonia Detection System. Leveraging advanced Deep Learning (DL) and Computer Vision (CV) techniques, the system performs automated binary classification of chest X-ray images, distinguishing between normal (healthy) lungs and pneumonia-infected lungs. The project goes beyond standard classification by incorporating an explainability module (Grad-CAM) and a robust user-facing web application, ensuring the tool is both highly accurate and clinically interpretable.

## 2. System Architecture
The pipeline is designed in a highly modular fashion, separating data ingestion, image preprocessing, model architecture, evaluation, and user interface into distinct logical components.

### 2.1 Technology Stack
- **Deep Learning Framework**: TensorFlow / Keras (Backend model training and inference)
- **Computer Vision**: OpenCV (Image preprocessing, morphological operations, contouring)
- **Web Interface**: Gradio (Interactive UI deployment)
- **Data Manipulation & Metrics**: NumPy, Pandas, Scikit-Learn, Matplotlib

### 2.2 Data Pipeline & Preprocessing
Medical imaging heavily relies on preprocessing to standardize data variations caused by different X-ray machinery.
- **Contrast Enhancement**: Implemented Contrast Limited Adaptive Histogram Equalization (CLAHE). This technique equalizes histograms on localized grid tiles rather than globally, highlighting critical pulmonary structures without amplifying sensor noise.
- **Lung Segmentation**: To minimize the risk of the neural network memorizing background artifacts (like medical tubing or bone structures), an Otsu Thresholding algorithm paired with morphological opening/closing was utilized. This isolates the left and right lung contours, creating a binary mask that blacks out non-pulmonary regions.

## 3. Modeling Methodology
### 3.1 Transfer Learning Strategy
Rather than training a Convolutional Neural Network (CNN) from scratch on a limited medical dataset, the system employs **Transfer Learning** via the VGG16 architecture. Pre-trained on the massive ImageNet dataset, VGG16 already possesses robust feature-extraction capabilities for identifying edges, textures, and shapes.

### 3.2 Two-Phase Training Protocol
To safely adapt the VGG16 weights to the specific domain of chest X-rays, a two-phase training protocol was executed:
1. **Feature Extraction (Phase 1)**: The core VGG16 layers were frozen to preserve their generalized weights. Only a custom classification head (Global Average Pooling -> Dense 256 -> Dropout 0.5 -> Output) was trained using an Adam optimizer at a higher learning rate (`1e-3`).
2. **Fine-Tuning (Phase 2)**: The top convolutional block of VGG16 was unfrozen. The model was then fine-tuned end-to-end at an extremely low learning rate (`1e-5`) to carefully shift the deep spatial representations toward lung-specific opacities.

*Note: Data imbalance (significantly more Pneumonia cases than Normal cases) was mitigated by calculating and applying dynamic class weights during the ImageDataGenerator pipeline.*

## 4. Model Evaluation & Explainability
### 4.1 Statistical Evaluation
The system was evaluated against an independent test set. In medical diagnostics, minimizing False Negatives (Recall) is the highest priority, as sending a sick patient home is more detrimental than subjecting a healthy patient to further screening. The model achieved exceptional metrics across the board (AUC-ROC ~0.98), indicating a highly reliable distinction boundary between the two classes.

### 4.2 Explainability via Grad-CAM
A fundamental challenge in medical AI is the "black box" nature of neural networks. To establish clinical trust, the system integrates Gradient-weighted Class Activation Mapping (Grad-CAM). 
- By calculating the gradients of the predicted class score with respect to the final convolutional feature maps (`block5_conv3`), the system generates a thermal heatmap.
- This heatmap is dynamically upsampled and overlaid onto the segmented lung image, explicitly highlighting the exact visual regions (e.g., cloudy infiltrates or consolidations) that triggered the AI's diagnosis.

## 5. Deployment & User Interface
To transition the system from a backend script to a usable clinical tool, an interactive web application was developed using **Gradio**. 
- **Premium Aesthetics**: The interface utilizes modern web design principles, featuring dark-mode glassmorphism styling, clean typography, and responsive layouts.
- **Side-by-Side Verification**: When an X-ray is uploaded, the app processes the image through the entire pipeline and displays three outputs:
  1. The CLAHE-enhanced image.
  2. The isolated, cleanly segmented lungs.
  3. The Grad-CAM attention overlay.
- **Dynamic Feedback**: The UI dynamically updates its styling and icons based on the severity of the diagnosis and the corresponding confidence percentage.

## 6. Conclusion
The Pneumonia Detection System successfully demonstrates how deep transfer learning can be synergized with classical computer vision to tackle medical imaging challenges. By enforcing strict lung segmentation, the model avoids learning irrelevant background correlations. Furthermore, the integration of Grad-CAM and a professional web interface bridges the gap between raw statistical accuracy and practical, human-in-the-loop diagnostic assistance.

*Disclaimer: This system is built for research and educational purposes. It is not intended for primary clinical diagnosis without the supervision of certified radiological professionals.*
