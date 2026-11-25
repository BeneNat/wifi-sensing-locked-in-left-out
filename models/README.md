
# Trained Machine Learning Models

This directory contains the serialized machine learning models developed for the **LILO** (Locked-In Left-Out) system. These models are designed to classify vehicle occupancy based on Wi-Fi Channel State Information (CSI) features.

## Overview

The repository includes several model architectures evaluated during the research phase. The models range from classical machine learning baselines to deep neural networks and finally to quantized versions optimized for edge deployment (TinyML).

## Model Catalog

### 1. Classical Baseline
* **`model_rf.joblib`**: A Random Forest classifier. This model serves as the performance baseline for comparing classical algorithms against neural network approaches.

### 2. Neural Network Architectures (PyTorch)
* **`model_small_mlp.pth`**: A lightweight Multi-Layer Perceptron (MLP). Optimized for low memory footprint, suitable for basic edge devices.
* **`model_deep_mlp.pth`**: A deep MLP architecture with multiple hidden layers, providing higher accuracy in complex environmental scenarios at the cost of increased computational requirements.
* **`model_tinyml_mlp.pth`**: The final production architecture. This model is specifically tailored for the TinyML pipeline, balancing accuracy and resource efficiency.

### 3. Optimized & Quantized Models
* **`model_tinyml_mlp_quantized.pth`**: The post-training quantized version of the TinyML MLP. This model uses **INT8** precision, significantly reducing flash and RAM usage while maintaining performance levels necessary for real-time occupant detection on the ESP32.

## Metadata

* **`model_metadata.json`**: Contains essential information about the models, including:
    * Feature scaling parameters (means and standard deviations).
    * Subcarrier indices used during training.
    * Model versioning and performance metrics.
    * Input/Output vector dimensions.

## Usage

These models are utilized by the scripts in the `server/` directory for real-time inference. The `.pth` files can be loaded using PyTorch, while the `.joblib` file is intended for Scikit-Learn.

To evaluate or re-quantize these models, refer to the tools in the `analysis/` directory, specifically `quantize_and_analyze.py`.

---
*The model suite is a core output of the LILO Engineering Thesis project at AGH University of Krakow.*