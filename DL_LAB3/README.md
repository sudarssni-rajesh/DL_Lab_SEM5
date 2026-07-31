# CNN for Image Classification on CIFAR-10

**Course:** CS3807 – Deep Learning Laboratory  
**Semester:** V, B.Tech Artificial Intelligence & Data Science  
**Institution:** Shiv Nadar University Chennai

---

## Overview

This experiment demonstrates the implementation of a **Convolutional Neural Network (CNN)** using **TensorFlow/Keras** for multi-class image classification on the **CIFAR-10** dataset.

The notebook covers the complete deep learning workflow, including data preprocessing, CNN model construction, training, evaluation, visualization of feature maps, pooling comparisons, and hyperparameter analysis.

---

## Objective

The objectives of this experiment are to:

- Understand the architecture and working of Convolutional Neural Networks.
- Build and train a CNN for image classification using TensorFlow/Keras.
- Evaluate the trained model using standard classification metrics.
- Visualize learned feature maps from convolutional layers.
- Compare Max Pooling and Average Pooling.
- Analyze the impact of important CNN hyperparameters on model performance.

---

## Dataset

The experiment uses the **CIFAR-10** dataset provided by TensorFlow.

### Dataset Statistics

- **Total Images:** 60,000
- **Training Images:** 50,000
- **Testing Images:** 10,000
- **Image Size:** 32 × 32 pixels
- **Color Channels:** RGB (3 channels)
- **Classes:** 10

### Classes

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

The dataset is automatically downloaded using:

```python
from tensorflow.keras.datasets import cifar10
```

No manual download is required.

---

## Features Implemented

### 1. Data Exploration

- Display dataset information
- Visualize sample images
- Plot class distribution

---

### 2. Data Preprocessing

- Pixel normalization
- One-hot encoding of labels
- Train-validation split

---

### 3. CNN Architecture

The implemented CNN consists of:

- Three Convolutional Blocks
  - Conv2D
  - ReLU Activation
  - MaxPooling2D
- Flatten Layer
- Fully Connected Dense Layers
- Softmax Output Layer

The notebook also provides a detailed summary of all trainable parameters.

---

### 4. Model Training

The model is trained using:

- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Evaluation Metric:** Accuracy

Training and validation metrics are monitored across multiple epochs.

---

### 5. Performance Visualization

Training progress is visualized through:

- Training Accuracy
- Validation Accuracy
- Training Loss
- Validation Loss

These plots help identify convergence and overfitting.

---

### 6. Model Evaluation

The trained CNN is evaluated using:

- Test Accuracy
- Macro Precision
- Macro Recall
- Macro F1-Score
- Confusion Matrix
- Classification Report

---

### 7. Feature Map Visualization

Intermediate activations from convolutional layers are visualized to understand hierarchical feature extraction.

Observations include:

- Early layers detect edges and color patterns.
- Intermediate layers detect textures and shapes.
- Deeper layers capture class-specific semantic features.

---

### 8. Pooling Comparison

The notebook compares:

- Max Pooling
- Average Pooling

Performance differences are analyzed using evaluation metrics and training behavior.

---

### 9. Hyperparameter Exploration

The following CNN hyperparameters are investigated:

- Kernel Size
- Stride
- Padding
- Number of Filters
- Optimizer
- Batch Size

Their influence on convergence speed and classification accuracy is discussed.

---

## Repository Structure

```
DL_Lab_SEM5/
│
├── Experiment_3_CNN_CIFAR10/
│   ├── CNN_CIFAR10_Experiment.ipynb
│   └── README.md
```

---

## Requirements

The notebook is developed using:

- Python 3
- TensorFlow / Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

All required libraries are pre-installed in Google Colab.

---

## How to Run

1. Open **CNN_CIFAR10_Experiment.ipynb** in Google Colab.
2. Select:

   ```
   Runtime → Change runtime type → GPU (T4)
   ```

3. Run all cells sequentially:

   ```
   Runtime → Run All
   ```

The CIFAR-10 dataset will be downloaded automatically during the first execution.

---

## Results

| Metric | Value |
|----------|-------|
| Test Accuracy | *(Fill after execution)* |
| Macro Precision | *(Fill after execution)* |
| Macro Recall | *(Fill after execution)* |
| Macro F1-Score | *(Fill after execution)* |

---

## Key Observations

- CNNs successfully learn hierarchical visual representations from images.
- Initial convolutional layers capture low-level features such as edges, corners, and color gradients.
- Deeper convolutional layers learn high-level semantic patterns useful for distinguishing object categories.
- Max Pooling generally preserves stronger feature activations, while Average Pooling produces smoother feature maps.
- CNN performance is sensitive to hyperparameters such as kernel size, filter count, optimizer selection, and batch size.
- Appropriate preprocessing and parameter tuning significantly improve classification accuracy.

---

## Learning Outcomes

After completing this experiment, students will be able to:

- Understand the fundamentals of Convolutional Neural Networks.
- Implement CNNs using TensorFlow/Keras.
- Train deep learning models for image classification.
- Evaluate CNN performance using standard metrics.
- Interpret feature maps generated by convolutional layers.
- Compare pooling strategies and analyze hyperparameter effects.

---

## Author

**Name:** *Sudarssni Rajesh*  
**Course:** CS3807 – Deep Learning Laboratory  
**Program:** B.Tech Artificial Intelligence & Data Science  
**Institution:** Shiv Nadar University Chennai

---
