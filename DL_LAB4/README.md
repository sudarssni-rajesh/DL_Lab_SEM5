# 🧠 CNN Architecture Comparison & Transfer Learning on CIFAR-10

### CS3807 — Deep Learning Laboratory

**B.Tech Artificial Intelligence & Data Science — Semester V**
**Shiv Nadar University Chennai**

A comparative study of five landmark Convolutional Neural Network (CNN) architectures — **LeNet-5, AlexNet, VGG16, GoogLeNet (InceptionV3), and ResNet50** — evaluated on the CIFAR-10 image classification dataset.

This experiment explores the historical evolution of CNN architectures while demonstrating the practical benefits of **transfer learning** using ImageNet-pretrained models.

---

## 📌 Project Overview

The objective of this experiment is to compare CNN architectures of increasing complexity and understand how architectural improvements and transfer learning affect image classification performance.

The following models are evaluated:

* **LeNet-5** — implemented from scratch
* **AlexNet (Scaled)** — implemented from scratch
* **VGG16** — ImageNet-pretrained transfer learning
* **GoogLeNet / InceptionV3** — ImageNet-pretrained transfer learning
* **ResNet50** — ImageNet-pretrained transfer learning

A fixed-seed subset of the CIFAR-10 dataset is used:

* **4,000 training images**
* **800 test images**
* **10 image classes**

All models are evaluated using the same standard classification metrics to enable a fair comparison.

---

## 🎯 Objectives

* Understand the evolution of CNN architectures.
* Implement **LeNet-5 and AlexNet from scratch**.
* Apply **transfer learning** using pretrained CNN backbones.
* Compare models based on:

  * Accuracy
  * Precision
  * Recall
  * F1-score
  * Training time
  * Number of parameters
* Understand the practical advantages of pretrained ImageNet models.
* Analyze the relationship between model complexity and classification performance.

---

## 🏗️ CNN Architectures Compared

| Model                       | Approach          | Input Size | Key Characteristics                                  |
| --------------------------- | ----------------- | ---------: | ---------------------------------------------------- |
| **LeNet-5**                 | From Scratch      |      32×32 | 2 convolution + pooling blocks, 3 dense layers       |
| **AlexNet (Scaled)**        | From Scratch      |      64×64 | Scaled-down AlexNet architecture                     |
| **VGG16**                   | Transfer Learning |      48×48 | Frozen ImageNet backbone + GAP + Dense head          |
| **GoogLeNet / InceptionV3** | Transfer Learning |      75×75 | Inception-based architecture with pretrained weights |
| **ResNet50**                | Transfer Learning |      48×48 | Residual connections + frozen ImageNet backbone      |

> **Note:** Keras does not provide pretrained weights for the original GoogLeNet architecture. Therefore, **InceptionV3** is used as the modern standard representative of the GoogLeNet/Inception family.

---

## 📊 Dataset — CIFAR-10

The experiment uses the **CIFAR-10** dataset available through `tensorflow.keras.datasets.cifar10`.

CIFAR-10 contains:

* **60,000 RGB images**
* Image resolution: **32 × 32**
* **10 classes**
* 50,000 training images
* 10,000 test images

For practical lab execution, a fixed-seed random subset is selected:

```text
Training: 4,000 images
Testing:    800 images
```

The ten classes are:

```text
airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck
```

---

## 🔄 Data Processing Pipeline

A common `tf.data` pipeline is used for the models.

```text
CIFAR-10 Images
      ↓
Resize to Model Input Size
      ↓
Model-specific preprocess_input()
      ↓
Cache
      ↓
Prefetch
      ↓
CNN Model
      ↓
10-Class Prediction
```

Different pretrained architectures require different input preprocessing, so the appropriate Keras `preprocess_input` function is applied for each transfer-learning model.

---

## 🔥 Transfer Learning

For **VGG16, InceptionV3, and ResNet50**, ImageNet-pretrained convolutional backbones are used.

The pretrained backbone is **frozen**, and a lightweight classification head is added:

```text
ImageNet Pretrained Backbone
            ↓
Global Average Pooling
            ↓
Dense(128)
            ↓
Dense(10)
            ↓
CIFAR-10 Prediction
```

This allows the experiment to take advantage of features learned from millions of ImageNet images without having to train the entire network from scratch.

---

## 📈 Results

| Model                                  |   Accuracy |  Precision |     Recall |   F1-Score | Training Time (s) | Parameters |
| -------------------------------------- | ---------: | ---------: | ---------: | ---------: | ----------------: | ---------: |
| **LeNet-5**                            |     0.4175 |     0.4271 |     0.4144 |     0.4005 |               8.5 |     83,126 |
| **AlexNet (Scaled)**                   |     0.4662 |     0.4746 |     0.4672 |     0.4567 |              19.3 |  2,546,186 |
| **VGG16 (Transfer)**                   |     0.5262 |     0.5176 |     0.5199 |     0.5173 |              15.1 | 14,781,642 |
| **GoogLeNet / InceptionV3 (Transfer)** |     0.5612 |     0.5677 |     0.5569 |     0.5559 |              30.0 | 22,066,346 |
| 🏆 **ResNet50 (Transfer)**             | **0.6538** | **0.6516** | **0.6487** | **0.6480** |              32.4 | 23,851,274 |

### 🏆 Best Performing Model

**ResNet50** achieved the highest performance across the experiment:

* **Accuracy:** 65.38%
* **Precision:** 65.16%
* **Recall:** 64.87%
* **F1-score:** 64.80%

The results show the following progression:

```text
LeNet-5
   ↓
AlexNet
   ↓
VGG16
   ↓
InceptionV3
   ↓
ResNet50
```

This demonstrates the overall improvement obtained from increasingly sophisticated CNN architectures and, particularly, the effectiveness of transfer learning on a relatively small dataset.

---

## ⏱️ Training Time vs Parameters

An important observation is that **training time does not directly correspond to the total number of parameters**.

For example:

* AlexNet has approximately **2.5M parameters** but trains every parameter from scratch.
* VGG16 has approximately **14.7M parameters**, but its pretrained backbone is frozen.
* As a result, VGG16 can train faster than AlexNet despite having substantially more parameters.

This highlights the computational advantage of **freezing pretrained layers** during transfer learning.

---

## 🧪 Methodology

### 1. Dataset Preparation

CIFAR-10 is loaded using TensorFlow/Keras and a fixed random seed is used to select the training and testing subsets.

### 2. Model Construction

Two architectures are trained from scratch:

```text
LeNet-5
AlexNet (Scaled)
```

Three architectures use ImageNet-pretrained backbones:

```text
VGG16
InceptionV3
ResNet50
```

### 3. Transfer Learning

For the pretrained models:

* ImageNet weights are loaded.
* Convolutional backbones are frozen.
* Global Average Pooling is applied.
* A small trainable classification head is added.
* The final layer contains **10 output neurons**, corresponding to CIFAR-10 classes.

### 4. Training

All models are trained using the same CIFAR-10 subset and evaluated on the held-out test subset.

### 5. Evaluation

Performance is measured using:

```text
Accuracy
Precision
Recall
F1-Score
Training Time
Parameter Count
```

---

## 🛠️ Technologies Used

* **Python**
* **TensorFlow / Keras**
* **Scikit-learn**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Google Colab**
* **LaTeX / Overleaf**

---

## 📦 Requirements

Install the required Python packages using:

```bash
pip install tensorflow>=2.20 scikit-learn pandas numpy matplotlib
```

Or use the provided Google Colab notebook.

---

## 🚀 How to Run

### Option 1 — Google Colab

1. Open `DL_LAb4.ipynb`.
2. Upload the notebook to Google Colab.
3. Select:

```text
Runtime → Change runtime type → GPU
```

4. Run all cells.

The experiment is designed to complete within a practical laboratory time budget on a **T4 GPU**.

### Option 2 — Local Environment

Clone the repository:

```bash
git clone <repository-url>
cd <repository-folder>
```

Install the dependencies:

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib
```

Open the notebook:

```bash
jupyter notebook DL_LAb4.ipynb
```

Run all cells sequentially.

---

## 📁 Repository Structure

```text
CNN-Architecture-Comparison/
│
├── DL_LAb4.ipynb
│
├── Experiment_CNN_Comparison_FINAL.tex
│
├── report_assets/
│   ├── *.eps
│   └── ...
│
└── README.md
```

### Files

**`DL_LAb4.ipynb`**
Complete executed Jupyter/Colab notebook containing dataset preparation, model implementations, training, evaluation, and results.

**`Experiment_CNN_Comparison_FINAL.tex`**
LaTeX source file for the complete laboratory report.

**`report_assets/`**
Figures and visualizations used in the final report.

**`README.md`**
Project documentation and experiment overview.

---

## 📄 Lab Report

The accompanying report covers:

* Objective
* Background Theory
* CNN Architecture Evolution
* Dataset
* Methodology
* Model Implementation
* Transfer Learning
* Results
* Performance Comparison
* Discussion
* Conclusion

The report is available as both a compiled PDF and an Overleaf-compatible LaTeX source.

---

## 💡 Key Findings

### 1. Architecture matters

More advanced CNN architectures generally achieved better classification performance on the selected CIFAR-10 subset.

### 2. Transfer learning is effective

Using ImageNet-pretrained features significantly improved performance compared with the smaller networks trained entirely from scratch.

### 3. ResNet50 performed best

ResNet50 achieved the highest accuracy and F1-score among the five architectures.

### 4. Parameter count isn't everything

A model with more parameters can still train faster when most of those parameters are frozen during transfer learning.

### 5. Residual connections help

ResNet's residual learning framework enables deeper networks to learn effectively and contributes to its strong performance.

---

## 🎓 Learning Outcomes

Through this experiment, the following concepts were explored:

* CNN architecture design
* Convolution and pooling
* Feature extraction
* Deep CNNs
* Image classification
* Transfer learning
* ImageNet pretrained models
* Residual networks
* Model evaluation metrics
* Computational efficiency
* Comparative deep learning experimentation

---

## 👤 Author

**Sudarssni Rajesh**
B.Tech Artificial Intelligence & Data Science
Semester V
**Shiv Nadar University Chennai**

---

## ⭐ Conclusion

This experiment demonstrates the evolution of CNN architectures from the relatively simple **LeNet-5** to deeper and more sophisticated architectures such as **VGG16, InceptionV3, and ResNet50**.

While LeNet-5 and AlexNet provide a useful understanding of CNNs trained from scratch, transfer learning allows significantly more powerful architectures to be applied effectively even when only a small subset of CIFAR-10 is available.

Among the evaluated models, **ResNet50 achieved the best overall performance with 65.38% accuracy**, demonstrating the practical advantage of modern deep CNN architectures combined with ImageNet pretraining.

