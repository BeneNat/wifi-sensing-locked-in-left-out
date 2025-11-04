# server/test_mqtt_publish.py
import json
import time
import paho.mqtt.client as mqtt

MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC_PRED = "csi/prediction"

client = mqtt.Client()
client.connect(MQTT_BROKER, MQTT_PORT, 60)
print("📡 Connected to MQTT broker.")

for i in range(3):
    msg = json.dumps({
        "source": "TEST",
        "timestamp": time.time(),
        "prediction": i,
        "confidence": 0.9
    })
    client.publish(MQTT_TOPIC_PRED, msg)
    print(f"➡️ Published test message {i}")
    time.sleep(1)

client.disconnect()
