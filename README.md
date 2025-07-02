# Explainable AI for Chest Disease Diagnosis

A Grad-CAM Enhanced EfficientNetV2S Deep Learning Approach

# Introduction

This project presents an explainable deep learning system for automatic classification of chest X-ray images into three categories: Pneumonia, COVID-19, and Normal. Traditional CNNs often miss subtle disease markers in chest radiographs, and their black-box nature limits clinical trust. This project overcomes those limitations using EfficientNetV2S for accurate feature extraction, CLAHE for image enhancement, and Grad-CAM for explainability—making AI predictions more transparent and medically trustworthy.

# Technologies Used

Python 3.8+

TensorFlow / Keras

OpenCV (for CLAHE preprocessing)

NumPy, Pandas, Matplotlib

EfficientNetV2S (via TensorFlow/Keras Applications)

Grad-CAM (Custom Implementation)

Jupyter Notebook


# Dataset

A balanced custom dataset called CONOP Dataset was curated from multiple public sources:

COVID-19 Images: GitHub, SIRM, Radiopaedia, Mendeley

Normal & Pneumonia Images: Kaggle Chest X-ray datasets

View Used: Posteroanterior (PA) chest X-rays only

Each class contains 2,313 images, totaling 6,939 images

Training set: 1,850 images/class

Test set: 463 images/class


# Methodology / Project Workflow

## 1. Image Preprocessing (CLAHE)

CLAHE enhances local contrast in chest X-rays.

Improves visibility of lung structures without over-amplifying noise.

## 2. Feature Extraction (EfficientNetV2S)

Pretrained on ImageNet and fine-tuned on chest X-rays.

Uses compound scaling to optimize depth, width, and resolution.

## 3. Classification

Fully connected dense layers with dropout and batch normalization.

Final output layer uses softmax activation for 3-class prediction.

## 4. Explainability (Grad-CAM)

Highlights image regions influencing model predictions.

Provides visual justification for medical decisions.

# Results

Validation Accuracy: 95.68%

AUC Score: 0.9884

F1-Scores: Pneumonia (0.95), Normal (0.94), COVID-19 (0.98)

COVID-19 Detection: Precision 0.99, Recall 0.98

Grad-CAM heatmaps validate model's attention on medically relevant areas.

# Conclusion

This project demonstrates a reliable and interpretable AI model for diagnosing pulmonary diseases using chest X-rays. The integration of CLAHE, EfficientNetV2S, and Grad-CAM enables high diagnostic accuracy along with visual explainability. This system holds strong potential as a clinical decision-support tool and can be extended with broader datasets, hybrid models, or deployed as a web/mobile diagnostic app.

