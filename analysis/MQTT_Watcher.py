import paho.mqtt.client as mqtt
import json

# Configuration
broker = "192.168.1.107"    # MQTT broker address
port = 1883
topics_to_subscribe = [
    "csi/prediction",
    "csi/conf",
    "csi/command",
    "esp32/bme680_full",
    "esp32/bme680_temp",
    "csi/heartbeat",
    "csi/#",
    "esp32/#"
]

# MQTT Callbacks
def on_connect(client, userdata, flags, reasonCode, properties):
    if reasonCode == 0:
        print("[MQTT] Connected to MQTT Broker successfully!")

        topic_list = [(topic, 0) for topic in topics_to_subscribe]
        client.subscribe(topic_list)

        print(f"[MQTT] Subscribed to topics: {', '.join(topics_to_subscribe)}")
    else:
        print(f"[MQTT] Failed to connect, reason code {reasonCode}\n")


def on_message(client, userdata, msg):
    try:
        payload_str = msg.payload.decode('utf-8')

        try:
            data = json.loads(payload_str)
            display_payload = json.dumps(data, indent=4)
        except json.JSONDecodeError:
            display_payload = payload_str

        print(f"\n--- Message Received ---")
        print(f"Topic: **{msg.topic}**")
        print(f"Payload:")
        print(display_payload)
        print(f"------------------------")

    except Exception as e:
        print(f"[ERROR] Error processing message on topic {msg.topic}: {e}")


# Main
client = mqtt.Client(client_id="Subscriber_AllTopics", callback_api_version=mqtt.CallbackAPIVersion.VERSION2)

client.on_connect = on_connect
client.on_message = on_message

print(f"[MQTT] Attempting to connect to broker at {broker}:{port}...")

try:
    client.connect(broker, port, 60)
except Exception as e:
    print(f"[ERROR] Connection error: {e}")
    exit()

print("\n--- Awaiting Messages ---")

client.loop_forever()