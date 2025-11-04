# server/capture_csi_serial.py
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

# ------------- SETTINGS -------------
DESIRED_ROLES = ["AP", "STA"]
BAUD_RATE = 921600

# MQTT local broker (adjust IP if not localhost)
MQTT_BROKER = "192.168.0.100"
MQTT_PORT = 1883

# Topics
MQTT_TOPIC_PRED = "csi/prediction"
MQTT_TOPIC_CONF = "csi/conf"
MQTT_TOPIC_CMD = "csi/command"

# SNOOZE control
SNOOZE_FLAG = threading.Event()
STOP_EVENT = threading.Event()
# ------------------------------------

def signal_handler(sig, frame):
    print("\n🛑 Stopping...")
    STOP_EVENT.set()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ============ MQTT ============

def on_message(client, userdata, msg):
    """Handle incoming messages (commands)."""
    try:
        payload = msg.payload.decode("utf-8").strip().upper()
        print(f"📩 MQTT message on {msg.topic}: {payload}")

        if payload == "SNOOZE":
            if not SNOOZE_FLAG.is_set():
                SNOOZE_FLAG.set()
                print("😴 SNOOZE command received → pausing predictions for 1 min")
                threading.Thread(target=snooze_timer, daemon=True).start()

    except Exception as e:
        print("⚠️ MQTT on_message error:", e)

def snooze_timer():
    """Handle SNOOZE duration (1 minutes)."""
    try:
        time.sleep(60)  # 1 minutes
    except Exception:
        pass
    SNOOZE_FLAG.clear()
    print("⏰ SNOOZE finished → resuming predictions")

def init_mqtt():
    try:
        client = mqtt.Client()
        client.on_message = on_message
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.subscribe(MQTT_TOPIC_CMD)
        client.loop_start()
        print(f"📡 MQTT connected to {MQTT_BROKER}:{MQTT_PORT}")
        return client
    except Exception as e:
        print("⚠️ MQTT init failed:", e)
        return None

MQTT_CLIENT = init_mqtt()

# ============ SERIAL WORKER ============

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
                        MQTT_CLIENT.publish(
                            MQTT_TOPIC_PRED,
                            int(pred["pred"])
                            #json.dumps({
                            #    "source": role,
                            #    "timestamp": ts,
                            #    "prediction": int(pred["pred"]),
                            #})
                        )
                        MQTT_CLIENT.publish(
                            MQTT_TOPIC_CONF,
                            round(float(pred["conf"])*100, 2)
                            #json.dumps({
                            #    "source": role,
                            #    "timestamp": ts,
                            #    "confidence": float(pred["conf"]),
                            #})
                        )
                    except Exception as e:
                        print(f"[{role}] mqtt pub failed: {e}")

    try:
        ser.close()
    except Exception:
        pass
    print(f"[{role}] closed, file saved: {fname}")

# ============ MAIN CONTROL ============

def start_capture():
    mapping = {
        "AP": "COM3",  # adjust to your setup
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
            print(f"[WARN] No port assigned for role {role}")

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
