# MobileNetV2: Comprehensive Deep Learning Experiments

A comprehensive deep learning experiment using **MobileNetV2** for image classification on the Oxford-IIIT Pet dataset. This project explores weight initialization, regularization, batch normalization, optimization algorithms, hyperparameter tuning, transfer learning, and cross-validation.

## 📌 Project Overview

This project investigates how different training strategies affect the performance of a Convolutional Neural Network (CNN).

MobileNetV2 is used as the primary architecture for classifying images into 37 pet categories. The experiments compare models trained from scratch with ImageNet-pretrained models and evaluate their performance using accuracy, precision, recall, F1-score, and confusion matrices.

### Objectives

* Study the effect of different weight initialization techniques.
* Analyze regularization methods and overfitting.
* Understand the role of Batch Normalization.
* Compare different optimization algorithms.
* Perform CNN hyperparameter tuning.
* Compare feature extraction and fine-tuning.
* Evaluate model stability using K-Fold Cross-Validation.
* Select and evaluate a final model on an untouched test set.

---

## 🗂️ Dataset

### Oxford-IIIT Pet Dataset

The project uses the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/), downloaded automatically using `torchvision`.

| Property          | Details                              |
| ----------------- | ------------------------------------ |
| Dataset           | Oxford-IIIT Pet                      |
| Task              | Multi-class image classification     |
| Number of classes | 37                                   |
| Image size        | 224 × 224 × 3                        |
| Training data     | 80% of the trainval split            |
| Validation data   | 20% of the trainval split            |
| Test data         | Official test split                  |
| Data augmentation | Random horizontal flip               |
| Normalization     | ImageNet mean and standard deviation |

### Data Preprocessing

1. Resize images to 224 × 224 pixels.
2. Apply random horizontal flipping to training images.
3. Convert images into tensors.
4. Normalize using ImageNet statistics.

```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
```

The test set remains untouched until the final evaluation stage.

---

## 🧠 Model Architecture

### MobileNetV2

MobileNetV2 is a lightweight CNN architecture designed for efficient image classification.

Key architectural concepts:

* Depthwise separable convolutions.
* Inverted residual blocks.
* Linear bottlenecks.
* Efficient computation and reduced parameter requirements.

The final classifier is replaced to predict 37 pet categories.

```python
model = models.mobilenet_v2(
    weights=None,
    dropout=0.2
)

model.classifier[1] = nn.Linear(
    model.last_channel,
    num_classes=37
)
```

### Model Configurations

The notebook evaluates two major approaches:

**1. Training from scratch**

MobileNetV2 is initialized with randomly initialized weights.

Used for studying:

* Weight initialization.
* Regularization.
* Batch Normalization.
* Optimization algorithms.
* Hyperparameter tuning.

**2. Transfer learning**

MobileNetV2 is initialized with ImageNet-pretrained weights.

Used for:

* Feature extraction.
* Fine-tuning.
* Final model evaluation when selected by cross-validation.

---

## 🧪 Experiments Performed

### 1. Weight Initialization

Compares the following initialization techniques:

* Zero initialization.
* Random normal initialization.
* Xavier/Glorot initialization.
* He/Kaiming initialization.

All configurations use the same architecture and training settings.

**Evaluation:**

* Training loss vs epoch.
* Validation accuracy vs epoch.
* Best-performing initialization.

---

### 2. Regularization and Overfitting

Compares different regularization strategies:

| Configuration       | Description                            |
| ------------------- | -------------------------------------- |
| No Regularization   | No additional weight decay or dropout  |
| L2 Regularization   | Weight decay = 1e-4                    |
| Dropout             | Dropout rate = 0.5                     |
| Batch Normalization | MobileNetV2's default BatchNorm layers |

**Evaluation:**

* Training accuracy.
* Validation accuracy.
* Training loss.
* Validation loss.

---

### 3. Batch Normalization

A numerical Batch Normalization example is demonstrated using:

```python
x = np.array([2, 4, 6, 8], dtype=float)
```

The notebook calculates:

* Batch mean.
* Batch variance.
* Normalized values.
* Output using gamma = 1 and beta = 0.

The experiment also compares training with and without Batch Normalization.

---

### 4. Optimization Algorithms

The following optimizers are compared:

* SGD.
* SGD with Momentum = 0.9.
* RMSProp.
* Adam.

**Evaluation:**

* Training loss vs epoch.
* Validation accuracy vs epoch.
* Convergence behavior.
* Approximate training time.

---

### 5. CNN Hyperparameter Tuning

The project follows a **one-hyperparameter-at-a-time** tuning strategy.

#### Learning Rate

```python
lr_values = [0.001, 0.0001]
```

#### Batch Size

Different batch sizes are evaluated while keeping the other settings fixed.

#### Dropout Rate

Different dropout values are tested to study their effect on validation accuracy.

**Evaluation:**

* Learning rate vs validation accuracy.
* Batch size vs validation accuracy.
* Dropout rate vs validation accuracy.

---

### 6. Transfer Learning and Fine-Tuning

Uses ImageNet-pretrained MobileNetV2.

#### Case A — Feature Extraction

* Freeze the pretrained backbone.
* Train only the new classifier.
* Learning rate = 1e-3.
* Train for 8 epochs.

#### Case B — Fine-Tuning

* Unfreeze the later feature blocks.
* Keep earlier layers frozen.
* Train using a smaller learning rate.
* Learning rate = 1e-4.
* Train for 8 epochs.

A smaller learning rate is used during fine-tuning to reduce the risk of destroying useful pretrained features.

---

### 7. K-Fold Cross-Validation

Five-fold cross-validation is performed on the training data.

The test set is not used during cross-validation.

Four configurations are compared:

| Configuration                     | Optimizer | Learning Rate | Dropout  | Pretrained |
| --------------------------------- | --------- | ------------- | -------- | ---------- |
| C1: Best Initialization + Adam    | Adam      | 1e-3          | 0.2      | No         |
| C2: Best Initialization + RMSProp | RMSProp   | 1e-3          | 0.2      | No         |
| C3: Best Hyperparameters          | Adam      | Selected      | Selected | No         |
| C4: Transfer Learning             | Adam      | 1e-3          | 0.2      | Yes        |

Each configuration is evaluated across 5 folds.

Metrics recorded:

* Mean validation accuracy.
* Standard deviation.
* Accuracy for each fold.

---

### 8. Final Model Evaluation

The best configuration is selected based on mean cross-validation accuracy.

The selected model is then retrained using the complete training data (train + validation).

Finally, the model is evaluated on the untouched test set.

### Evaluation Metrics

* Test Accuracy.
* Precision.
* Recall.
* F1-score.
* Confusion Matrix.
* Training Time.
* Number of Trainable Parameters.

The notebook also identifies:

* Best-classified pet categories.
* Most frequently confused categories.
* Sample misclassified images.

---

## 📊 Results

The notebook generates 15 plots covering all major experiments.

| Plot    | Description                                          |
| ------- | ---------------------------------------------------- |
| Plot 1  | Training Loss vs Epoch — Weight Initialization       |
| Plot 2  | Validation Accuracy vs Epoch — Weight Initialization |
| Plot 3  | Training & Validation Accuracy — Regularization      |
| Plot 4  | Training & Validation Loss — Regularization          |
| Plot 5  | With vs Without Batch Normalization                  |
| Plot 6  | Training Loss vs Epoch — Optimizers                  |
| Plot 7  | Validation Accuracy vs Epoch — Optimizers            |
| Plot 8  | Learning Rate vs Validation Accuracy                 |
| Plot 9  | Batch Size vs Validation Accuracy                    |
| Plot 10 | Dropout Rate vs Validation Accuracy                  |
| Plot 11 | Feature Extraction vs Fine-Tuning                    |
| Plot 12 | Training & Validation Loss — Transfer Learning       |
| Plot 13 | 5-Fold Cross-Validation Accuracy                     |
| Plot 14 | Confusion Matrix — Test Set                          |
| Plot 15 | Sample Misclassified Images                          |

### Results Summary

The notebook calculates the following final metrics:

```text
Mean CV Accuracy:        Generated by notebook
CV Standard Deviation:   Generated by notebook
Test Accuracy:           Generated by notebook
Precision:               Generated by notebook
Recall:                  Generated by notebook
F1-score:                Generated by notebook
Training Time:           Generated by notebook
Number of Parameters:    Generated by notebook
```

> Note: Run the notebook to obtain the actual numerical results. The README does not assume any performance values.

---

## 🛠️ Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Torchinfo
* Jupyter Notebook

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### 2. Install Dependencies

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn torchinfo jupyter
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the MobileNetV2 experiment notebook.

---

## ▶️ How to Run

1. Open the notebook in Jupyter Notebook or Google Colab.
2. Run the setup and import cells.
3. Execute the dataset preparation section.
4. Run the model builder and training utilities.
5. Execute each experiment section.
6. Review the generated plots and metrics.
7. Run the final model evaluation.
8. Run the additional exercise configurations if required.

The Oxford-IIIT Pet dataset is automatically downloaded through `torchvision`.

---

## 📁 Project Structure

```text
MobileNetV2-Experiments/
│
├── data/
│   └── Oxford-IIIT Pet Dataset
│
├── CS3807_Experiment5_MobileNetV2.ipynb
│
├── README.md
│
└── results/
    ├── plots/
    └── evaluation_metrics/
```

The `results` folder is optional and can be used to store generated plots and evaluation outputs.

---

## 🔍 Additional Exercise

The notebook includes two additional configurations for comparison:

```python
extra_configs = {
    "New_C5": dict(
        optimizer="Adam",
        lr=1e-4,
        dropout=0.5,
        pretrained=False
    ),

    "New_C6": dict(
        optimizer="Adam",
        lr=1e-3,
        dropout=0.2,
        pretrained=True
    ),
}
```

These configurations are evaluated using 5-fold cross-validation.

They are compared against the originally selected best configuration based on:

1. Mean accuracy.
2. Standard deviation.
3. Approximate training time and compute cost.

---

## 📚 Key Concepts Covered

* CNN architecture.
* MobileNetV2.
* Transfer learning.
* Fine-tuning.
* Weight initialization.
* Xavier and He initialization.
* Regularization.
* Dropout.
* Batch Normalization.
* SGD, Momentum, RMSProp, and Adam.
* Learning rate and batch size.
* K-Fold Cross-Validation.
* Confusion Matrix.
* Precision, Recall, and F1-score.

---

## 👩‍💻 Author

**Sudarssni Rajesh**

B.Tech Artificial Intelligence & Data Science
Shiv Nadar University Chennai

---

## 📄 License

This project is intended for educational and academic purposes.

