
# Effect of Attention Type and Attention Position in a Pretrained ResNet50

## Overview

This experiment investigates how different attention mechanisms and their insertion positions affect the classification performance of a pretrained ResNet50 model. The experiment uses the TensorFlow Flowers dataset to compare baseline performance against multiple attention-based architectures.

The objective is to understand whether attention improves classification accuracy, how its placement within the network influences performance, and whether the improvements justify the additional computational complexity.

## Objectives

- Implement and evaluate different attention mechanisms in ResNet50.
- Compare channel attention, spatial attention, self-attention, and combined attention approaches.
- Investigate the effect of inserting attention at different network stages.
- Evaluate models using classification metrics and visualisations.
- Analyse the trade-off between model performance and computational complexity.

## Dataset

**Dataset:** TensorFlow Flowers (`tf_flowers`)

The dataset contains five flower classes:

- Daisy
- Dandelion
- Roses
- Sunflowers
- Tulips

| Dataset property | Details |
|---|---|
| Total images | 3,670 |
| Number of classes | 5 |
| Training split | 70% |
| Validation split | 15% |
| Testing split | 15% |

The dataset is split using stratified sampling with a fixed random seed of 42.

## Model Architecture

ResNet50 pretrained on ImageNet is used as the backbone. Attention modules are integrated into selected feature stages to investigate their effect on the learned representations.

The experiment evaluates the following configurations:

- Baseline ResNet50
- Channel Attention / Squeeze-and-Excitation (SE)
- Spatial Attention
- Efficient Channel Attention (ECA)
- Convolutional Block Attention Module (CBAM)
- Self-Attention
- Combined Attention
- Channel-to-Spatial Attention
- Channel-to-Spatial Attention at different positions (P2, P3, P4, and P3 + P4)

## Methodology

1. Load and preprocess the TensorFlow Flowers dataset.
2. Apply image resizing, normalisation, and data augmentation.
3. Load the pretrained ResNet50 backbone.
4. Implement attention modules and integrate them at the selected feature stages.
5. Train the models using a common training protocol.
6. Evaluate each configuration on the test set.
7. Compare classification metrics, computational complexity, and visualisations.

### Training Configuration

- Optimiser: Adam
- Batch size: 32
- Dropout: 0.3
- Training schedule: 3 epochs with the backbone frozen, followed by 2 fine-tuning epochs
- Random seed: 42

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Number of trainable and additional parameters
- Inference time

Training and validation accuracy/loss curves, confusion matrices, feature maps, attention maps, and Grad-CAM-style visualisations are used to analyse model behaviour.

## Results

The baseline ResNet50 achieved a test accuracy of **92.09%** and a macro F1-score of **92.04%**.

| Model configuration | Test accuracy |
|---|---:|
| Baseline ResNet50 | 92.09% |
| Self-Attention at P3 | 93.53% |
| Combined Attention at P3 | 93.71% |
| Channel-to-Spatial Attention at P4 | 93.35% |

The combined attention configuration achieved the highest reported test accuracy, followed by self-attention at P3 and channel-to-spatial attention at P4.

The results indicate that the effectiveness of attention depends on both the mechanism and its insertion position. Increasing model complexity does not necessarily improve performance.

*Note: Results are based on a single run with a fixed seed and a five-epoch training schedule. The observed differences should therefore be interpreted cautiously.*

## Key Observations

- Attention mechanisms produce different effects depending on their design and placement.
- Self-attention at P3 and combined attention at P3 achieved the strongest reported results.
- Channel-to-spatial attention at P4 outperformed the same approach at earlier positions.
- Adding attention at multiple stages did not always improve accuracy.
- The additional parameter overhead was relatively small, but inference-time measurements were too noisy to establish a reliable performance ranking.
- The saved attention visualisations did not demonstrate strong or consistent localisation of the flower regions.

## Technologies Used

- Python
- TensorFlow / Keras
- ResNet50
- NumPy
- Matplotlib
- Scikit-learn

## Conclusion

This experiment compares multiple attention mechanisms and insertion positions within a pretrained ResNet50 for flower classification. The results show that selected configurations can improve upon the baseline, but the gains are modest and depend on the attention design and placement.

Further experiments using multiple random seeds, longer training schedules, and additional visualisations would help establish the reliability of the observed improvements.

## Author

**Sudarssni**  
B.Tech – Artificial Intelligence and Data Science
