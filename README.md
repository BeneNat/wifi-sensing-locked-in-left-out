# wifi-sensing-left-alive
Detecting the Presence of Living Beings Left in Cars Using Wi-Fi Channel Sensing.

Proof-of-concept pipeline:
ESP32 (CSI capture) → MQTT → server (ingest + inference).
Includes data parsing, feature extraction, and TinyML MLP model for presence detection.
