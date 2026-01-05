# Environmental Monitoring Module (BME680)

This directory contains the Arduino-based firmware for the environmental monitoring unit of the **LILO** system. This module is responsible for real-time tracking of the vehicle cabin's conditions, primarily to assess the risk of hyperthermia.

## Overview

The firmware utilizes an **ESP32** microcontroller paired with a **BME680** sensor to measure critical environmental data. These measurements act as a vital safety layer, correlating occupant detection data with real-time temperature trends to trigger multi-level emergency alerts.

## Key Functionalities

* **Sensor Fusion:** Captures temperature, humidity, atmospheric pressure, and gas resistance (VOC).
* **Air Quality Analysis:** Includes a simplified Indoor Air Quality (IAQ) calculation based on gas sensor resistance.
* **Real-time Connectivity:** Connects to a local Wi-Fi network to transmit data.
* **MQTT Integration:** Publishes environmental metrics to an MQTT broker for centralized processing.
* **Status Feedback:** Utilizes the built-in LED as a heartbeat indicator to confirm the device is operational.

## Hardware Requirements

* **Microcontroller:** ESP32 (e.g., ESP-WROOM-32).
* **Sensor:** Bosch BME680 (connected via I2C).
* **Interface:** I2C (SDA/SCL pins).

## Software Dependencies

The firmware requires the following Arduino libraries:
* `WiFi.h` & `Wire.h` (Built-in)
* `PubSubClient` (by Nick O'Leary)
* `Adafruit_BME680` & `Adafruit_Sensor`

## Configuration

Before flashing the firmware, update the following constants in `temperature_sensor.ino`:
* `ssid` & `password`: Your Wi-Fi network credentials.
* `mqtt_server`: The IP address of your MQTT broker (default: `192.168.1.107`).

## Data Transmission Format

The module publishes data every 8 seconds to the following topics:

1. **`esp32/bme680_full` (JSON Format):**
   ```json
   {
     "temperature": 24.50,
     "humidity": 45.20,
     "pressure": 1013.25,
     "gas": 45.20,
     "iaq": 50
   }
   ```
   
2. **esp32/bme680_temp (Plain String):**
* Publishes the raw temperature value (e.g., 24.50) for lightweight monitoring.

## Installation
1. Open temperature_sensor.ino in the Arduino IDE.
2. Install the required libraries via the Library Manager.
3. Configure your Wi-Fi and MQTT settings.
4. Select ESP32 Dev Module as your board.
5. Compile and upload to the device.

---
*This environmental sensing unit is a critical safety component of the LILO Engineering Thesis project at AGH University of Krakow.*