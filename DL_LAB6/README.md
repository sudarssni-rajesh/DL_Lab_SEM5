# Experiment 6 — RNN, LSTM & GRU for Sequence Learning and Video Understanding

**Course:** CS3807 – Deep Learning Laboratory  
**Experiment:** 6  
**Topic:** Sequence Learning, Human Activity Recognition, Video Understanding and Sequence-to-Sequence Learning

---

## 📌 Overview

This experiment provides an end-to-end implementation and comparison of:

- Vanilla RNN
- LSTM
- GRU
- CNN + LSTM for video understanding
- Encoder–Decoder LSTM for sequence-to-sequence learning

The main sequence-classification task uses the **UCI Human Activity Recognition (HAR) Using Smartphones** dataset.

Each sample is represented as a temporal sequence of:

```text
128 time steps × 9 sensor channels
```

The experiment also covers:

- Backpropagation Through Time (BPTT)
- Vanishing and exploding gradients
- Training and validation curves
- Confusion matrix analysis
- Model performance comparison
- Sequence-length analysis
- CNN feature extraction using MobileNetV2
- CNN + LSTM video understanding
- Encoder–Decoder sequence-to-sequence learning

---

## 🎯 Objectives

1. Understand sequential data representation in the form `(N, T, F)`.
2. Implement Vanilla RNN, LSTM and GRU for sequence classification.
3. Understand Backpropagation Through Time (BPTT).
4. Study the vanishing and exploding gradient problems.
5. Compare RNN, LSTM and GRU using multiple evaluation metrics.
6. Analyze training/validation curves and confusion matrices.
7. Study the effect of sequence length on classification performance.
8. Build a CNN + recurrent network pipeline for video understanding.
9. Understand encoder–decoder architecture for sequence-to-sequence learning.

---

## 📊 Dataset

### UCI Human Activity Recognition Using Smartphones

The primary dataset contains six activities:

- WALKING
- WALKING_UPSTAIRS
- WALKING_DOWNSTAIRS
- SITTING
- STANDING
- LAYING

The raw inertial signals are arranged as:

```text
(N, 128, 9)
```

where:

- `N` = number of samples
- `128` = number of time steps
- `9` = sensor channels

The nine channels consist of:

```text
Body Acceleration: X, Y, Z
Body Gyroscope:    X, Y, Z
Total Acceleration: X, Y, Z
```

The experiment uses a 70/15/15 train-validation-test split.

---

## 🔄 Experimental Pipeline

```text
Raw Sensor Signals
        ↓
Windowing
   128 × 9
        ↓
Normalization
        ↓
RNN / LSTM / GRU
        ↓
Dense Layer
        ↓
Softmax
        ↓
Activity Class
```

---

## 🧠 Model Architecture

All three recurrent models use the same classifier structure so that the main difference is the recurrent layer.

```text
Input: 128 × 9
       ↓
RNN / LSTM / GRU
32 Units
       ↓
Dropout (0.2)
       ↓
Dense (16, ReLU)
       ↓
Dense (6, Softmax)
```

### Training Configuration

| Parameter | Value |
|---|---|
| Recurrent Units | 32 |
| Dropout | 0.2 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 32 |
| Epochs | 30 |
| Loss Function | Sparse Categorical Cross-Entropy |

---

## 🔢 BPTT Numerical Verification

The RNN recurrence used is:

```text
hₜ = tanh(Wₓxₜ + Wₕhₜ₋₁ + b)
```

Using:

```text
x₁ = 0.5
x₂ = 0.7
x₃ = 0.2

h₀ = 0
Wₓ = 0.5
Wₕ = 0.8
b = 0.1
```

The calculated hidden states are:

```text
h₁ = 0.336376
h₂ = 0.616352
h₃ = 0.599958
```

The notebook verifies that the programmatic results match the manual calculations.

---

## 📈 Evaluation Metrics

Each recurrent model is evaluated using:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Number of trainable parameters
- Training time
- Confusion Matrix

The notebook generates:

1. Sensor signal visualization
2. Training vs validation loss
3. Training vs validation accuracy
4. Confusion matrices
5. Model performance comparison
6. Sequence length vs F1-score

---

## 📊 HAR Model Results

| Model | Accuracy (%) | Macro Precision (%) | Macro Recall (%) | Macro F1 (%) | Parameters |
|---|---:|---:|---:|---:|---:|
| RNN | 50.67 | 51.36 | 50.67 | 47.71 | 1,974 |
| LSTM | 66.44 | 71.80 | 66.44 | 58.36 | 6,006 |
| GRU | 76.67 | 76.74 | 76.67 | 76.43 | 4,758 |

The results show different behavior across the three recurrent architectures, particularly when distinguishing the static activities:

```text
SITTING
STANDING
LAYING
```

The dynamic WALKING-family activities have more distinctive temporal patterns.

---

# 🎥 Video Understanding — CNN + LSTM

A video can be represented as a sequence of image frames.

The implemented pipeline is:

```text
Video
  ↓
Frame Sampling
  ↓
MobileNetV2
  ↓
CNN Feature Extraction
  ↓
LSTM / GRU
  ↓
Dense
  ↓
Softmax
  ↓
Action Class
```

### Configuration

```text
Frames per video: 10
Frame size:       224 × 224 × 3
CNN:              MobileNetV2
Feature size:     1280
Recurrent units:  32
```

MobileNetV2 is used as a frozen feature extractor.

The recurrent network receives:

```text
10 × 1280
```

features for each video.

The experiment uses a synthetic video generator when real UCF101 video files are unavailable.

The five action classes are:

- Basketball
- Biking
- Walking
- Running
- TennisSwing

---

## 🧩 CNN + LSTM Architecture

```text
10 Video Frames
       ↓
MobileNetV2
       ↓
10 × 1280 Features
       ↓
LSTM
32 Units
       ↓
Dense
       ↓
Softmax
       ↓
Action Class
```

The CNN extracts **spatial information** from individual frames, while the recurrent network learns the **temporal relationship** between frames.

---

## 🔁 Sequence-to-Sequence Learning

The final section implements an encoder-decoder LSTM for a sequence reversal task.

Example:

```text
Input:
[1, 4, 7, 2]

Output:
[2, 7, 4, 1]
```

### Architecture

```text
Input Sequence
      ↓
Encoder LSTM
      ↓
Context State
      ↓
Decoder LSTM
      ↓
Output Sequence
```

The task uses synthetic sequences and evaluates:

- Token Accuracy
- Sequence Accuracy
- Training Loss
- Validation Loss

### Obtained Results

```text
Token Accuracy   = 1.0000
Sequence Accuracy = 1.0000
Final Training Loss = 0.0250
Final Validation Loss = 0.0244
```

---

## 🛠️ Technologies Used

- Python
- NumPy
- Matplotlib
- TensorFlow / Keras
- Scikit-learn
- MobileNetV2
- OpenCV

---

## 📦 Installation

Install the required libraries:

```bash
pip install numpy matplotlib tensorflow scikit-learn opencv-python
```

If OpenCV is not required for your run, it can be omitted.

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

### 2. Navigate into the project

```bash
cd <YOUR_PROJECT_FOLDER>
```

### 3. Install dependencies

```bash
pip install numpy matplotlib tensorflow scikit-learn opencv-python
```

### 4. Open the notebook

```text
DL_LAB6_24011101110.ipynb
```

You can use either:

- Jupyter Notebook
- JupyterLab
- VS Code with the Jupyter extension
- Google Colab

### 5. Run all cells

Run the notebook from beginning to end so that the dataset preparation, model training, evaluation and plots are generated in the correct order.

---

## 📁 Project Structure

```text
.
├── DL_LAB6_24011101110.ipynb
├── README.md
└── UCI_HAR_Dataset/
```

If real video data is used:

```text
.
├── DL_LAB6_24011101110.ipynb
├── README.md
├── UCI_HAR_Dataset/
└── ucf101_subset/
    ├── Basketball/
    ├── Biking/
    ├── Walking/
    ├── Running/
    └── TennisSwing/
```

---

## 🧠 Key Concepts

### Vanilla RNN

A Vanilla RNN maintains a hidden state that is updated at every time step. It can struggle to learn long-term dependencies because of vanishing gradients.

### LSTM

LSTM introduces a cell state and three gates:

- Forget Gate
- Input Gate
- Output Gate

These gates control how information is retained, added and exposed.

### GRU

GRU uses two gates:

- Update Gate
- Reset Gate

It has a simpler structure than LSTM while still providing gated memory.

### CNN + RNN

The CNN extracts spatial features from individual video frames, while the recurrent network models how those features change over time.

### Encoder–Decoder

The encoder converts an input sequence into a learned representation, while the decoder generates the output sequence one element at a time.

---

## 📌 Conclusion

This experiment demonstrates how recurrent neural networks can model temporal dependencies in sequential data.

The experiment compares Vanilla RNN, LSTM and GRU on human activity recognition and further extends recurrent modeling to video understanding using CNN-extracted features.

It also demonstrates sequence-to-sequence learning through an encoder-decoder LSTM on a synthetic sequence reversal task.

---

## 👤 Author

**24011101110**

**B.Tech Artificial Intelligence & Data Science**  
**Shiv Nadar University Chennai**

**Course:** CS3807 – Deep Learning Laboratory
