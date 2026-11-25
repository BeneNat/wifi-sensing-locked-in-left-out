# LILO Inference and Control Server

This directory contains the central server-side logic for the **LILO** (Locked-In Left-Out) system. The server acts as the integration layer, responsible for real-time data acquisition from hardware, feature extraction, neural network inference, and alert distribution.

## Overview

The server orchestrates the data flow from the physical environment to the user's mobile device. It handles the high-frequency stream of Wi-Fi Channel State Information (CSI) from the ESP32 modules, processes it through optimized machine learning models, and manages the communication state via MQTT.

## Key Components

### 1. Data Acquisition & Parsing
* **`capture_csi_serial.py`**: The main entry point for real-time operation. It manages the serial connection with the ESP32 station, capturing raw CSI bursts.
* **`serial_parser.py`**: A robust utility for decoding raw serial strings into structured numerical matrices.
* **`capture_csi_debug.py`**: A diagnostic tool used for validating the stability of the serial stream and visualizing raw signal perturbations.

### 2. Real-time Inference Pipeline
* **`live_features.py`**: Implements the real-time version of the feature engineering pipeline. It maintains a sliding window of CSI data to calculate temporal and statistical features identical to those used during model training.
* **`inference_test.py`**: A script for validating model performance on live or recorded data streams before full system deployment.

### 3. Communication & Alerting
* **`client_mqtt_sub.py`**: An MQTT client that subscribes to environmental data (e.g., from the BME680 sensor) to provide context for the occupancy detection.
* **`test_mqtt_publish.py`**: A utility for testing the notification delivery system and verifying that the mobile application correctly receives alert packets.

## System Workflow

1. **Capture:** `capture_csi_serial.py` receives raw CSI packets via UART.
2. **Process:** `live_features.py` transforms raw subcarriers into a feature vector.
3. **Classify:** The system loads a quantized model from the `../models/` directory to perform inference.
4. **Notify:** If presence is detected and environmental conditions (temperature) are hazardous, an alert is published to the MQTT broker.

## Installation

### Prerequisites
* Python 3.10+
* Virtual environment (recommended)

### Setup
1. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure the MQTT broker (e.g., Mosquitto) is running and accessible at the IP address configured in the scripts.
3. Connect the ESP32 station module to the server's USB port.

## Requirements
The server depends on the following key libraries:
* pyserial: For low-latency UART communication.
* paho-mqtt: For robust messaging between system components.
* torch: For running neural network inference.
* numpy & pandas: For high-performance numerical data manipulation.

---
*The server orchestration layer is a core part of the LILO Engineering Thesis project at AGH University of Krakow.*