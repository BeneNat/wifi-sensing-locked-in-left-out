import time
import json
import paho.mqtt.client as mqtt
import numpy as np

client = mqtt.Client()
client.connect("localhost",1883,60)

for i in range(5):
    payload = {"device_id":"sim1","timestamp":time.time(),"features": np.random.randn(28).tolist()}
    client.publish("csi/sim1/window", json.dumps(payload), qos=1)
    print("published", i)
    time.sleep(1)
