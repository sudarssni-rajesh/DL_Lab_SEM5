# Deep Learning Lab – Autoencoders & Variational Autoencoder

## 📌 Overview

This project explores different autoencoder architectures using the MNIST handwritten digit dataset.

The experiments cover image reconstruction, denoising, latent-space dimensionality, and generative modeling using:

- Fully Connected Autoencoder (FC-AE)
- Convolutional Autoencoder (CAE)
- Denoising Autoencoder (DAE)
- Variational Autoencoder (VAE)

The models are evaluated using MSE, MAE, and SSIM, along with visual comparisons of reconstructed images.

## 📊 Dataset

The project uses the MNIST handwritten digit dataset.

| Dataset | Images |
|---|---:|
| Training | 9,000 |
| Validation | 1,000 |
| Testing | 2,000 |

Each image is:

- Grayscale
- 28 × 28 pixels
- Normalized to the range [0, 1]

## 🧠 Experiments

### 1. Fully Connected Autoencoder

Architecture:

    784 → 128 → 32 → 16 → 32 → 128 → 784

Configuration:

- Latent dimension: 16
- Activation: ReLU
- Output activation: Sigmoid
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 128
- Epochs: 20
- Reconstruction loss: Binary Cross-Entropy

The model compresses each 28 × 28 image into a 16-dimensional latent representation and reconstructs it.

### 2. Convolutional Autoencoder

The Convolutional Autoencoder uses convolutional layers to preserve spatial information in the images.

    Input Image
         ↓
      Conv2D
         ↓
     MaxPooling
         ↓
      Conv2D
         ↓
     MaxPooling
         ↓
    Latent Representation
         ↓
      Conv2D
         ↓
     UpSampling
         ↓
      Conv2D
         ↓
     UpSampling
         ↓
    Reconstructed Image

The convolutional architecture is compared with the fully connected autoencoder based on reconstruction performance and visual quality.

### 3. FC-AE vs Convolutional AE

The two architectures are compared using:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Structural Similarity Index (SSIM)
- Number of trainable parameters
- Visual reconstruction quality

### 4. Denoising Autoencoder

The Denoising Autoencoder is trained to reconstruct clean MNIST images from noisy inputs.

#### Gaussian Noise

- σ = 0.1
- σ = 0.2
- σ = 0.3

#### Salt-and-Pepper Noise

- p = 0.05
- p = 0.10
- p = 0.20

The noisy image is provided as the input while the original clean image is used as the target.

This allows the model to learn how to remove noise while preserving the structure of the handwritten digit.

### 5. Latent Dimension Study

The effect of changing the latent-space dimensionality is investigated.

The experiment studies the relationship between latent dimension and:

- Reconstruction MSE
- MAE
- SSIM
- Compression
- Information retention
- Reconstruction quality

A smaller latent space provides greater compression but may result in information loss and poorer reconstruction.

### 6. Variational Autoencoder

A 2-dimensional Variational Autoencoder (VAE) is implemented to study the structure of the latent space.

Unlike a standard autoencoder, the encoder produces:

    μ       → Mean
    log σ²  → Log Variance

The latent vector is sampled using the reparameterization trick:

    z = μ + σ × ε

where:

    ε ~ N(0, I)

The VAE loss consists of:

    Total Loss = Reconstruction Loss + KL Divergence

The KL divergence regularizes the latent space towards a standard normal distribution.

## 📈 Evaluation Metrics

### Mean Squared Error (MSE)

Measures the average squared difference between the original and reconstructed pixels.

**Lower MSE indicates better reconstruction.**

### Mean Absolute Error (MAE)

Measures the average absolute pixel difference.

**Lower MAE indicates better reconstruction.**

### Structural Similarity Index (SSIM)

Measures the structural similarity between the original and reconstructed images.

**Higher SSIM indicates better structural similarity.**

## 📊 Visualizations

The project generates visualizations for:

- Original vs reconstructed images
- Reconstruction error maps
- Training and validation loss
- FC-AE vs CAE comparison
- Gaussian noise experiments
- Salt-and-pepper noise experiments
- Latent dimension comparison
- VAE latent-space visualization
- Generated digit samples
- Latent-space interpolation
- 2D latent manifold
- Reconstruction-error distributions
- Highest-error test images

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- scikit-image
- Jupyter Notebook

## 🚀 How to Run

### 1. Clone the repository

    git clone <your-repository-url>
    cd <repository-name>

### 2. Install the required libraries

    pip install tensorflow numpy pandas matplotlib scikit-image

### 3. Open the notebook

Open `Experiment_7.ipynb` using Jupyter Notebook, JupyterLab, or Google Colab.

### 4. Run the notebook

Run the cells from top to bottom.

The notebook will:

1. Load the MNIST dataset
2. Preprocess the images
3. Train the autoencoder models
4. Evaluate reconstruction performance
5. Generate visualizations
6. Save the experiment outputs

## 📁 Project Structure

    .
    ├── Experiment_7.ipynb
    ├── figures/
    │   ├── *.png
    │   ├── consolidated_results.csv
    │   ├── noise_results.csv
    │   ├── latent_dim_results.csv
    │   └── vae_results.csv
    │
    ├── exp7_outputs.zip
    └── README.md

## 🎯 Learning Outcomes

This experiment provides practical understanding of:

- Autoencoder architectures
- Encoder-decoder networks
- Latent representations
- Bottleneck layers
- Convolutional autoencoders
- Denoising autoencoders
- Image reconstruction
- Reconstruction loss
- MSE, MAE and SSIM
- Latent-space dimensionality
- Variational autoencoders
- KL divergence
- Reparameterization trick
- Latent-space interpolation
- Generative image sampling
- Reconstruction-error analysis

## 👨‍💻 Author

**Sudarssni Rajesh**

B.Tech Artificial Intelligence & Data Science  
Shiv Nadar University Chennai
