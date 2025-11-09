import serial
import threading
import datetime
import os
import signal
import sys
import time
import json
import numpy as np
import paho.mqtt.client as mqtt

# Ensure imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from server.serial_parser import parse_csi_line
from server.live_features import LiveFeatureBuffer

SAVE_DIR = "data_logs"
os.makedirs(SAVE_DIR, exist_ok=True)

# SETTINGS
DESIRED_ROLES = ["AP", "STA"]
BAUD_RATE = 921600

# MQTT local broker
MQTT_BROKER = "192.168.1.107"
MQTT_PORT = 1883

# Topics
MQTT_TOPIC_PRED = "csi/prediction"
MQTT_TOPIC_CONF = "csi/conf"
MQTT_TOPIC_CMD = "csi/command"
MQTT_TOPIC_ACK = "csi/confirmation"
MQTT_TOPIC_TEMP = "esp32/bme680_temp"

# Temperature thresholds (°C)
SAFE_TEMP_MIN = 0.0
SAFE_TEMP_MAX = 30.0
CURRENT_TEMP = None

# SNOOZE control
SNOOZE_FLAG = threading.Event()
STOP_EVENT = threading.Event()

def signal_handler(sig, frame):
    print("\nStopping...")
    STOP_EVENT.set()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# MQTT
def on_message(client, userdata, msg):
    global CURRENT_TEMP

    try:
        payload = msg.payload.decode("utf-8").strip().upper()
        topic = msg.topic
        print(f"[MQTT] MQTT message on {topic}: {payload}")

        # Handle SNOOZE command
        if topic == MQTT_TOPIC_CMD and payload == "SNOOZE":
            if not SNOOZE_FLAG.is_set():
                SNOOZE_FLAG.set()
                print("[MQTT] SNOOZE command received -> pausing predictions for 1 min")
                threading.Thread(target=snooze_timer, args=(60,), daemon=True).start()

        # Handle ACK confirmation
        elif topic == MQTT_TOPIC_ACK and payload == "ACK":
            if not SNOOZE_FLAG.is_set():
                SNOOZE_FLAG.set()
                print("[MQTT] ACK received -> pausing predictions for 5 minutes")
                threading.Thread(target=snooze_timer, args=(300,), daemon=True).start()

        # Handle temperature updates
        elif topic == MQTT_TOPIC_TEMP:
            try:
                CURRENT_TEMP = float(payload)
                print(f"[MQTT] Updated temperature: {CURRENT_TEMP:.2f} °C")
            except ValueError:
                print(f"[MQTT] Invalid temperature payload: {payload}")

    except Exception as e:
        print("[MQTT] MQTT on_message error:", e)


def snooze_timer(duration_seconds):
    try:
        time.sleep(duration_seconds)
    except Exception:
        pass
    SNOOZE_FLAG.clear()
    print(f"[MQTT] Pause finished ({duration_seconds/60:.1f} min) -> resuming predictions")

def init_mqtt():
    try:
        client = mqtt.Client()
        client.on_message = on_message
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.subscribe(MQTT_TOPIC_CMD)
        client.subscribe(MQTT_TOPIC_ACK)
        client.subscribe(MQTT_TOPIC_TEMP)
        client.loop_start()
        print(f"[MQTT] MQTT connected to {MQTT_BROKER}:{MQTT_PORT}")
        return client
    except Exception as e:
        print("[MQTT] MQTT init failed:", e)
        return None

MQTT_CLIENT = init_mqtt()

# SERIAL WORKER

def log_serial_worker(role, port):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = os.path.join(SAVE_DIR, f"csi_{role}_{timestamp}.csv")
    print(f"[{role}] opening {port} @ {BAUD_RATE}, logging -> {fname}")

    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=2)
        ser.reset_input_buffer()
        ser.reset_output_buffer()
    except Exception as e:
        print(f"[{role}] cannot open {port}: {e}")
        return

    feature_buffer = LiveFeatureBuffer(window_size=60, step=25)

    with open(fname, "w", encoding="utf-8", buffering=1) as fh:
        fh.write("# iso_ts,raw_line\n")
        lines = 0
        parsed = 0

        while not STOP_EVENT.is_set():
            try:
                raw = ser.readline()
                if not raw:
                    continue
                lines += 1
                line = raw.decode('utf-8', errors='ignore').strip()
            except Exception as e:
                print(f"[{role}] decode error: {e}")
                continue

            # filter out non-CSI lines
            if "CSI_DATA" not in line:
                if lines % 500 == 0:
                    print(f"[{role}] seen {lines} lines, preview: {line[:120]}")
                continue

            ts = datetime.datetime.now().isoformat(timespec="milliseconds")
            fh.write(f"{ts},{line}\n")

            parsed_payload = parse_csi_line(line)
            if not parsed_payload:
                print(f"[{role}] CSI_DATA but parse failed (line #{lines})")
                continue

            payload = parsed_payload.get("payload")
            if payload is None or not isinstance(payload, np.ndarray) or payload.size == 0:
                print(f"[{role}] parsed but empty payload")
                continue

            parsed += 1
            print(f"[{role}] CSI parsed #{parsed} len={payload.size} mean={np.mean(payload):.2f} std={np.std(payload):.2f}")

            feats, pred = feature_buffer.add_sample(payload)

            # Skip predictions if SNOOZE is active
            if SNOOZE_FLAG.is_set():
                continue

            if feats is not None and pred:
                print(f"[{role}] -> FEATS len={len(feats)} pred={pred}")

                if MQTT_CLIENT:
                    try:
                        raw_pred = int(pred["pred"])
                        # Determine output value based on temperature and movement
                        if raw_pred == 0:
                            final_pred = 0  # clear
                        else:
                            if CURRENT_TEMP is None:
                                # If temperature not yet received, assume safe temporarily
                                final_pred = 1
                            elif SAFE_TEMP_MIN <= CURRENT_TEMP <= SAFE_TEMP_MAX:
                                final_pred = 1  # medium alert
                            else:
                                final_pred = 2  # high alert (presence + unsafe temperature)

                        MQTT_CLIENT.publish(MQTT_TOPIC_PRED, final_pred)
                        MQTT_CLIENT.publish(MQTT_TOPIC_CONF, round(float(pred["conf"]) * 100, 2))

                        print(f"[{role}] MQTT publish: pred={final_pred}, temp={CURRENT_TEMP}, conf={pred['conf']:.2f}")

                    except Exception as e:
                        print(f"[{role}] mqtt pub failed: {e}")

    try:
        ser.close()
    except Exception:
        pass
    print(f"[{role}] closed, file saved: {fname}")

# MAIN CONTROL
def start_capture():
    mapping = {
        "AP": "COM3",
        "STA": "COM4",
    }

    print("Role -> port mapping:", mapping)

    threads = []
    for role in ["AP", "STA"]:
        if mapping.get(role):
            t = threading.Thread(target=log_serial_worker, args=(role, mapping[role]), daemon=False)
            t.start()
            threads.append(t)
        else:
            print(f"[WARNING] No port assigned for role {role}")

    try:
        while not STOP_EVENT.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        STOP_EVENT.set()

    for t in threads:
        t.join(timeout=2.0)

    if MQTT_CLIENT:
        try:
            MQTT_CLIENT.loop_stop()
            MQTT_CLIENT.disconnect()
        except Exception:
            pass

    print("Capture stopped.")

if __name__ == "__main__":
    start_capture()
