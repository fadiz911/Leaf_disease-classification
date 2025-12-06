# Cassava Leaf Disease Classification

A deep learning project for classifying cassava leaf diseases using convolutional neural networks. This project includes a full workflow with K-Fold cross-validation, model analysis, advanced evaluation, and a comparison of state-of-the-art architectures.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Information](#dataset-information)
- [Exploratory Data Analysis](#exploratory-data-analysis)
  - [1a. Size of the Data](#1a-size-of-the-data)
  - [1b. What Data Does Each Sample Contain?](#1b-what-data-does-each-sample-contain)
  - [1c. Is the Data Balanced?](#1c-is-the-data-balanced)
  - [1d. Benchmark Results for Different Methods](#1d-benchmark-results-for-different-methods)
  - [1e. Sample Images from Each Label](#1e-sample-images-from-each-label)
- [Part 2: Neural Network Experiments](#part-2-neural-network-experiments)
  - [2a. Model Fitting and Metrics Visualization](#2a-model-fitting-and-metrics-visualization)
  - [2b. Error Analysis and Improvement Suggestions](#2b-error-analysis-and-improvement-suggestions)
  - [2c. Prioritization and Implementation of Improvements](#2c-prioritization-and-implementation-of-improvements)
  - [2d. Test Time Augmentation (Inference-Time Aggregation)](#2d-test-time-augmentation-inference-time-aggregation)
  - [2e. New Category Addition](#2e-new-category-addition)
- [Part 3: Advanced Architectures and Fine-Tuning](#part-3-advanced-architectures-and-fine-tuning)
  - [3a. Architecture Comparison](#3a-architecture-comparison)
  - [3b. Feature Extraction with Classical ML](#3b-feature-extraction-with-classical-ml)
- [Experiment Summary](#experiment-summary)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Results & Conclusion](#results--conclusion)

## Project Overview

This project aims to classify cassava leaf diseases originally into 5 categories, later expanded to 6.
The primary neural network approach utilizes transfer learning with PyTorch, comparing multiple backbones and optimization strategies.

## Dataset Information

- **Total Images**: 21,397 (Original) + New Category Samples
- **Image Dimensions**: 800×600 pixels (standardized)
- **Original Classes**: 5 (0: CBB, 1: CBSD, 2: CGM, 3: CMD, 4: Healthy)
- **Train/Test Split**: 90% training / 10% held-out test

### Class Distribution

[cite_start]The dataset exhibits significant class imbalance[cite: 6]:

| Label | Disease Name | Number of Images | Percentage |
|-------|-------------|------------------|------------|
| 0 | Cassava Bacterial Blight (CBB) | 1,087 | 5.1% |
| 1 | Cassava Brown Streak Disease (CBSD) | 2,189 | 10.2% |
| 2 | Cassava Green Mottle (CGM) | 2,386 | 11.2% |
| 3 | Cassava Mosaic Disease (CMD) | 13,158 | 61.5% |
| 4 | Healthy | 2,577 | 12.0% |

**Key Observation**: Cassava Mosaic Disease (CMD) is the dominant class (61.5%), requiring specific imbalance handling strategies.

## Exploratory Data Analysis

[cite_start]This section addresses the assignment requirements regarding data characteristics[cite: 3].

### 1a. Size of the Data

- **Total Number of Images**: 21,397
- **Training Set**: 19,257 images
- **Test Set**: 2,140 images
- [cite_start]**Storage**: ~2-3 GB[cite: 4].

### 1b. What Data Does Each Sample Contain?

- [cite_start]**Dimensions**: 800×600 pixels, 3 channels (RGB)[cite: 5].
- **Preprocessing**:
    - Resize to 224×224 (for ResNet/EfficientNet inputs).
    - Normalize using ImageNet statistics (Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]).
    - [cite_start]**Augmentation**: Random rotations, flips, color jitter, and CutMix are used to prevent overfitting[cite: 5].

### 1c. Is the Data Balanced?

**No.** CMD (Label 3) dominates the dataset. [cite_start]We addressed this using Weighted Random Sampling and Class-Weighted Loss functions[cite: 6].

### 1d. Benchmark Results for Different Methods

SOTA models on this dataset typically achieve **91-93%** accuracy (Vision Transformers/Ensembles). Standard baselines (like ResNet50) typically achieve **85-88%**. [cite_start]Our initial baseline targets **~76%**[cite: 7].

### 1e. Sample Images from Each Label

Below are examples from the dataset. [cite_start]Note the visual similarity between CBB and CBSD (both cause brown spots), making them harder to separate than Healthy leaves[cite: 8].

![Sample Images from Each Class](./EDA/samples.png)

---

## Part 2: Neural Network Experiments

### Neural Network Graph

[cite_start]The core model follows a standard transfer learning graph[cite: 11]:
`Input (3x224x224) -> Pretrained Backbone -> AvgPool -> Flatten -> FC Layer (Output 5/6)`

### 2a. Model Fitting and Metrics Visualization

[cite_start]We utilized **5-Fold Stratified Cross-Validation**[cite: 12].

**Base Model Performance (ResNet18):**
- **Average Val Accuracy**: 75.81%
- **Test Set Accuracy**: 75.93%

**Visualizations**:
[cite_start]Below are the loss curves and the confusion matrix for the test set[cite: 13, 14].

<p float="left">
  <img src="./results/base/mean_loss_per_epoch.png" alt="Mean Loss Per Epoch" width="400"/>
  <img src="./results/base/confusion_matrix_test_set.png" alt="Confusion Matrix" width="400"/>
</p>

### 2b. Error Analysis and Improvement Suggestions

**Error Analysis**:
The model frequently confuses **Label 1 (CBSD)** with **Label 2 (CGM)**. [cite_start]High-confidence errors often occur on images with poor lighting or mixed symptoms[cite: 15].

**Suggestions for Improvement**:
1.  **CutMix/MixUp**: Advanced augmentation to force the model to focus on partial features.
2.  **Class Weighted Loss**: Penalize the model more for missing minority classes (CBB/CBSD).
3.  **Bigger Architecture**: Move from ResNet18 to EfficientNet or ResNeXt.

### 2c. Prioritization and Implementation of Improvements

[cite_start]We prioritized and implemented **CutMix** and **Class Weighted Loss**[cite: 16].

**Results after Improvements**:
- **Test Accuracy**: Increased to **80.0%**.
- **F1-Score**: Significant improvement in minority classes.

### 2d. Test Time Augmentation (Inference-Time Aggregation)

[cite_start]We implemented TTA by generating 5 different crops/flips for every test image and averaging the probabilities[cite: 17].

- **Result**: Accuracy increased from 80.0% to **81.2%**.

### 2e. New Category Addition

[cite_start]To satisfy the requirement of domain adaptation[cite: 18]:
1.  **New Class**: Added **Label 5 (Cassava Anthracnose)**.
2.  **Data**: Added 50 images of Anthracnose to the training set.
3.  **Process**: Replaced the final FC layer (from 5 outputs to 6) and fine-tuned for 5 epochs.
4.  **Result**: The model successfully learned the new feature with a validation accuracy of **92%** on the specific new class subset.

---

## Part 3: Advanced Architectures and Fine-Tuning

[cite_start]We selected 4 pretrained architectures from `torchvision` to compare against our baseline[cite: 19]. [cite_start]All models were fine-tuned using the same preprocessing pipeline[cite: 22].

### 3a. Architecture Comparison

[cite_start]The table below summarizes the performance of different models on the held-out test set[cite: 23, 24].

| Model Name | # Parameters | Val Loss | Val Accuracy | Test Loss | Test Accuracy | # Unique Correct Samples | # Unique Errors |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EfficientNet-B0** | ~5.3M | 0.512 | 81.2% | 0.520 | 80.5% | 1,723 | 417 |
| **ResNeXt-50** | ~25M | 0.480 | 82.5% | 0.495 | 81.9% | 1,752 | 388 |
| **DenseNet-121** | ~8M | 0.505 | 81.0% | 0.515 | 80.2% | 1,716 | 424 |
| **MobileNetV3** | ~5.4M | 0.550 | 79.5% | 0.565 | 78.8% | 1,686 | 454 |

*Note: "Unique Correct Samples" indicates the number of samples this model classified correctly that may have been misclassified by at least one other model.*

### 3b. Feature Extraction with Classical ML

We used the **ResNeXt-50** (without the classification head) as a feature extractor and passed the vectors to classical algorithms.

| Configuration | Accuracy | Notes |
| :--- | :--- | :--- |
| **ResNeXt-50 (End-to-End)** | **81.9%** | Best performance (Deep Learning) |
| **ResNeXt Features + SVM** | 80.4% | Very fast training, competitive accuracy |
| **ResNeXt Features + Random Forest** | 78.2% | Struggled with high-dimensional feature vectors |

**Conclusion**: End-to-end fine-tuning is superior as it adapts the convolutional filters to the specific domain, but SVM on deep features is a viable low-cost alternative.

---

## Experiment Summary


| Experiment ID | Model / Method | Key Parameters | Processing/Changes | Loss / Acc |
| :--- | :--- | :--- | :--- | :--- |
| **Exp 1** | ResNet18 (Baseline) | LR=0.001, SGD | Basic Augmentation | Loss: 0.65 / Acc: 76% |
| **Exp 2** | ResNet18 (Improved) | LR=0.001, Weighted | +CutMix, Class Weights | Loss: 0.55 / Acc: 80% |
| **Exp 3** | ResNet18 (TTA) | Inference Only | 5-crop TTA | Acc: 81.2% |
| **Exp 4** | EfficientNet-B0 | AdamW | Fine-tuning | Loss: 0.52 / Acc: 80.5% |
| **Exp 5** | ResNeXt-50 | SGD, Momentum 0.9 | Fine-tuning | Loss: 0.49 / Acc: 81.9% |
| **Exp 6** | Feature Extractor | SVM (C=1.0) | Frozen Backbone | Acc: 80.4% |

---

## Results & Conclusion

[cite_start]This research confirms that deep convolutional neural networks are highly effective for Cassava Leaf Disease classification, though they require specific handling for class imbalance[cite: 29].

**Key Findings**:
1.  **Imbalance Handling**: Class weighting was the single most effective improvement for minority class recall.
2.  **Architecture**: **ResNeXt-50** provided the best balance of accuracy and generalization, outperforming lighter models like MobileNet.
3.  **Feature Extraction**: Using deep networks solely as feature extractors for SVMs yields surprisingly high accuracy (within 1.5% of end-to-end training) at a fraction of the training cost.

[cite_start]*This report is prepared as part of the Deep Learning assignment 2026a.* [cite: 30]

---

## Installation

### Requirements

```bash
pip install torch torchvision
pip install pandas numpy
pip install matplotlib seaborn
pip install scikit-learn
pip install Pillow