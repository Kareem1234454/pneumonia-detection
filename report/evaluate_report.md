# evaluate.py Documentation Report

## Overview
`evaluate.py` provides the statistical and graphical metrics necessary to objectively judge the model's performance on the test dataset. In medical diagnostic systems, accuracy alone is insufficient; a breakdown of false positives and false negatives is paramount.

## Key Metrics & Outputs
### 1. Classification Metrics (`print_classification_report`)
Calculates and prints standard machine learning metrics:
- **Precision**: The proportion of predicted pneumonia cases that were actually pneumonia.
- **Recall (Sensitivity)**: The proportion of actual pneumonia cases the model successfully detected.
- **F1 Score**: The harmonic mean of Precision and Recall.
- **Specificity**: The ability to correctly identify normal (healthy) lungs.

### 2. Visual Diagnostics
- **Confusion Matrix**: Generates a heatmap detailing the raw counts and percentages of True Positives, True Negatives, False Positives, and False Negatives.
- **ROC Curve**: Plots the Receiver Operating Characteristic curve and calculates the Area Under the Curve (AUC), giving a holistic view of the model's performance across all classification thresholds.

### 3. Data Persistence
Results are formatted and automatically saved as a structured CSV report in `outputs/results/`, allowing for tracking of model iterations over time.
