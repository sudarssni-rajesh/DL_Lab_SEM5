# Deep Learning Lab 2 – Multilayer Perceptron (MLP) on Fashion-MNIST

Implementation of a **Multilayer Perceptron (MLP)** using TensorFlow/Keras for image classification on the **Fashion-MNIST** dataset as part of the **CS3807 – Deep Learning Laboratory** course.

---

## 📌 Overview

This project demonstrates the complete workflow of building, training, optimizing, and evaluating a Multilayer Perceptron for multi-class image classification.

The notebook includes:

- Dataset exploration
- Data preprocessing
- Baseline MLP model construction
- Model training and evaluation
- Hyperparameter optimization using RandomizedSearchCV
- Performance comparison
- Visualization of training results
- Additional perceptron experiments on logic gates and XOR

---

## 📂 Dataset

**Dataset:** Fashion-MNIST

Fashion-MNIST consists of grayscale images of clothing items belonging to 10 different categories.

### Dataset Statistics

- **Training Images:** 60,000
- **Testing Images:** 10,000
- **Image Size:** 28 × 28 pixels
- **Classes:** 10

Example classes include:

- T-shirt/Top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle Boot

---

## 🚀 Features

- Fashion-MNIST dataset exploration
- Image preprocessing and normalization
- Baseline Multilayer Perceptron implementation
- TensorFlow/Keras model development
- Model evaluation using standard classification metrics
- Hyperparameter optimization with RandomizedSearchCV
- Confusion Matrix visualization
- Training & Validation Accuracy plots
- Training & Validation Loss plots
- Performance comparison between baseline and optimized models

---

## 📊 Visualizations

The notebook generates:

- Sample Fashion-MNIST Images
- Training Accuracy vs Epoch
- Validation Accuracy vs Epoch
- Training Loss vs Epoch
- Validation Loss vs Epoch
- Baseline Confusion Matrix
- Optimized Model Confusion Matrix
- Classification Report
- Performance Comparison

---

## ⚙️ Model Architecture

The project uses a fully connected **Multilayer Perceptron (MLP)** built with TensorFlow/Keras.

Typical architecture includes:

- Flatten Layer
- Dense Hidden Layers
- ReLU Activation
- Dropout (during hyperparameter optimization)
- Softmax Output Layer

---

## 🔍 Hyperparameter Optimization

Model performance is improved using **RandomizedSearchCV** with SciKeras.

Parameters explored include:

- Number of hidden units
- Number of hidden layers
- Learning rate
- Batch size
- Epochs
- Dropout rate
- Optimizer

The optimized model is retrained using the best hyperparameters before final evaluation.

---

## 📈 Evaluation Metrics

Model performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

---

## 🧠 Additional Experiments

### Logic Gates using Perceptron

Implementation of the Perceptron Learning Algorithm for:

- AND Gate
- OR Gate
- NOT Gate

Each experiment includes:

- Weight updates after every iteration
- Bias updates
- Decision boundary visualization
- Learning analysis

---

### XOR Gate

Implementation of a Single Layer Perceptron on the XOR problem.

The notebook demonstrates:

- Weight updates
- Decision boundary evolution
- Failure to converge
- Explanation of why XOR is not linearly separable
- Motivation for Multilayer Perceptrons

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- SciKeras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

---

## 📁 Repository Structure

```
.
├── DL_LAB2.ipynb
├── README.md
└── report/
```

---

## 🎯 Learning Outcomes

Through this experiment, the following concepts are explored:

- Image classification using neural networks
- Data preprocessing for deep learning
- Building Multilayer Perceptrons
- Hyperparameter tuning
- Model evaluation using classification metrics
- Understanding overfitting and generalization
- Comparing baseline and optimized models
- Understanding the limitations of Single Layer Perceptrons
- Explaining why XOR requires hidden layers

---

## 📖 Course

**CS3807 – Deep Learning Laboratory**

Shiv Nadar University Chennai

---

## 👩‍💻 Author

**Sudarssni Rajesh**

**Reg. No.: 24011101110**

B.Tech Artificial Intelligence & Data Science

Shiv Nadar University Chennai
