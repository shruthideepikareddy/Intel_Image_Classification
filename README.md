# Intel Image Classification with Deep Learning

A comparative deep learning project for scene image classification using a custom CNN, ResNet50, and EfficientNetB0. The project investigates the impact of transfer learning, fine-tuning, and model explainability techniques on image classification performance using the Intel Image Classification dataset.

---

## Project Overview

This project was developed to compare different convolutional neural network architectures under identical training conditions.

Three different approaches were implemented and evaluated:

* Custom CNN built from scratch
* ResNet50 with Transfer Learning
* EfficientNetB0 with Transfer Learning and Fine-Tuning

The models were trained, evaluated, and compared using multiple performance metrics. Grad-CAM was also applied to visualize which image regions influenced model predictions.

---

## Dataset

**Intel Image Classification Dataset**

Classes:

* Buildings
* Forest
* Glacier
* Mountain
* Sea
* Street

The dataset contains RGB images belonging to six natural scene categories.

---

## Project Workflow

1. Data Loading
2. Data Preprocessing
3. Data Augmentation
4. Custom CNN Development
5. Transfer Learning with ResNet50
6. Transfer Learning with EfficientNetB0
7. Fine-Tuning
8. Model Evaluation
9. Error Analysis
10. Grad-CAM Visualization
11. Performance Comparison

---

## Models

### 1. Custom CNN

* Functional API
* Convolution + ReLU
* MaxPooling
* Batch Normalization
* Dropout
* Softmax Output

---

### 2. ResNet50

* ImageNet pretrained weights
* Frozen convolutional base
* Custom classification head
* Transfer Learning

---

### 3. EfficientNetB0

* ImageNet pretrained weights
* Transfer Learning
* Fine-Tuning
* Best performing model

---

## Techniques Used

* TensorFlow / Keras
* Functional API
* Data Augmentation
* EarlyStopping
* ModelCheckpoint
* ReduceLROnPlateau
* Transfer Learning
* Fine-Tuning
* Grad-CAM
* Confusion Matrix
* Classification Report

---

## Model Explainability

Grad-CAM was used to visualize the regions that contributed most to the model's predictions. This helped interpret model behavior and understand common misclassification patterns.

---

## Results

The models were evaluated using:

* Accuracy
* Validation Accuracy
* Loss
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Grad-CAM

The final comparison demonstrates the advantages of transfer learning and fine-tuning over a CNN trained from scratch.

---

## Technologies

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib
* Scikit-learn
* OpenCV

---

## Repository Structure

```
.
├── notebooks/
├── models/
├── outputs/
│   ├── confusion_matrix/
│   ├── gradcam/
│   └── predictions/
├── images/
├── README.md
└── requirements.txt
```

---

## Future Improvements

* Hyperparameter Optimization
* More EfficientNet Variants
* Vision Transformers (ViT)
* Test-Time Augmentation
* Ensemble Learning
* Streamlit Deployment

---

## Author

**Ece Nur Bayraktar**

Computer Engineering Student

Deep Learning • Computer Vision • Machine Learning
