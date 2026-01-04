# ESP32 Firmware for LILO Data Acquisition

This directory contains the firmware for the hardware layer of the **LILO** system. The firmware is responsible for generating Wi-Fi traffic and capturing Channel State Information (CSI) packets, which are essential for occupant detection.

## Overview

The system utilizes a pair of ESP32 modules to monitor the propagation of Wi-Fi signals within the vehicle cabin. By analyzing how these signals are distorted (CSI), the system can identify human or animal presence. The firmware is developed using the **ESP-IDF** (Espressif IoT Development Framework).

## Folder Structure

The firmware is organized into several modules depending on the hardware role:

* **`active_ap/`**: Firmware for the **Access Point** module. It creates a stable Wi-Fi network and acts as the signal transmitter.
* **`active_sta/`**: Firmware for the **Station** module. It connects to the AP and captures CSI data from incoming packets, forwarding the raw data via Serial (UART) for further analysis.
* **`passive/`**: Sniffer mode firmware for non-intrusive monitoring of existing Wi-Fi traffic.
* **`_components/`**: Shared header files and modular logic used across different firmware versions:
    * `csi_component.h`: Logic for CSI initialization and callback handling.
    * `sd_component.h`: Support for local data logging on SD cards.
    * `sockets_component.h`: UDP/TCP communication logic.
    * `nvs_component.h`: Non-Volatile Storage management.
* **`temperature_sensor/`**: An Arduino-based sketch (`.ino`) for the environmental monitoring unit, providing real-time cabin temperature data to assess heatstroke risks.

## Hardware Requirements

* **Modules:** ESP-WROOM-32 (38-pin development boards recommended).
* **Sensors:** DS18B20 or similar temperature sensors (for the `temperature_sensor` module).
* **Connection:** USB-C or Micro-USB for power and serial data transmission.

## Setup and Installation

### Prerequisites
1.  Install the **ESP-IDF SDK** (v4.4 or newer recommended).
2.  Set up the environment variables (e.g., `source $IDF_PATH/export.sh`).

### Building and Flashing
Navigate to the desired module directory (e.g., `active_sta/`) and run:

```bash
# Set the target chip
idf.py set-target esp32

# Build the firmware
idf.py build

# Flash the firmware and monitor serial output
idf.py -p [PORT] flash monitor
```

## CSI Configuration
The CSI capture is configured to monitor the HT20 (High Throughput 20MHz) bandwidth. The system filters packets to focus on stable subcarriers, ensuring high sensitivity to micro-movements such as breathing.

---
*This firmware suite is a core component of the LILO Engineering Thesis project at AGH University of Krakow.*