# Data Analysis and Machine Learning Pipeline

This directory contains the core logic for processing Wi-Fi Channel State Information (CSI), engineering features, and training machine learning models optimized for edge devices (TinyML).

## Overview

The analysis pipeline is designed to transform chaotic, high-dimensional raw CSI data into robust predictions of occupant presence. The workflow is divided into four main stages:
1. **Parsing:** Cleaning and synchronizing raw serial data.
2. **Feature Engineering:** Extracting statistical and temporal characteristics from subcarriers.
3. **Training:** Developing various architectures, from Random Forests to Deep Multi-Layer Perceptrons (MLP).
4. **Optimization:** Quantizing models to INT8 format for deployment on ESP32 or other edge microcontrollers.

## Key Components

### 1. Data Processing & Feature Engineering
* `parse_robust.py`: Handles raw CSI strings, filtering out noise and malformed packets.
* `features.py` & `improved_features.py`: Implement the feature extraction logic. These scripts calculate variance, mean, and signal perturbations across selected subcarriers to create a compact input vector for the models.
* `prepare_dataset.py`: Aggregates processed files into balanced datasets for training and validation.

### 2. Model Training
We evaluated several architectures to find the best trade-off between accuracy and resource consumption:
* `train_rf.py`: Baseline implementation using Random Forest.
* `train_small_mlp.py`: A lightweight MLP designed for minimal SRAM usage.
* `train_deep_mlp.py`: A high-accuracy model with multiple hidden layers for complex environment analysis.
* `train_tinyml_mlp.py`: The final production model architecture specifically tailored for TinyML constraints.

### 3. TinyML Optimization
* `quantize_and_analyze.py`: **Critical tool for the engineering thesis.** It performs post-training quantization (PTQ) to convert weights from FP32 to INT8. It also provides detailed metrics on:
    * Inference latency (ms).
    * Model size reduction (kB).
    * Accuracy impact after quantization.

## Usage Workflow

To reproduce the results or train a new model:

1. **Parse Raw Data:**
   ```bash
   python parse_robust.py --input ../data/raw/ --output ./processed/
   ```

2. **Generate Features:**
    ```bash
    python improved_features.py --input ./processed/ --output ./features/
    ```

3. **Train Model:**
    ```bash
    python train_tinyml_mlp.py
    ```

4. **Quantize & Evaluate:**
    ```bash
    python quantize_and_analyze.py --model ../models/model_tinyml_mlp.pth
    ```

## Requirements

The scripts require a Python 3.10+ environment with the following libraries:
* torch (PyTorch)
* scikit-learn
* pandas
* numpy
* joblib

---
*This analysis suite is part of the LILO Engineering Thesis project at AGH University of Krakow.*