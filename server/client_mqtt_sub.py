# client_mqtt_sub.py
import paho.mqtt.client as mqtt

def on_message(client, userdata, msg):
    print(f"{msg.topic} → {msg.payload.decode()}")

client = mqtt.Client()
client.connect("broker.hivemq.com", 1883, 60)
client.subscribe("csi/prediction")

print("📡 Listening for predictions on 'csi/prediction' ...")
client.on_message = on_message
client.loop_forever()
