# Debug capture + live inference.
import serial
import serial.tools.list_ports
import threading
import datetime
import os
import signal
import sys
import time
import json
import numpy as np
import paho.mqtt.client as mqtt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from server.serial_parser import parse_csi_line
from server.live_features import LiveFeatureBuffer

# CONFIG
PORTS = {
    "AP": "COM3",
    "STA": "COM4",
}
BAUD_RATE = 921600
SAVE_DIR = "data_logs"
os.makedirs(SAVE_DIR, exist_ok=True)

# quick overrides
WINDOW_SIZE = 200    # smaller to get faster feedback
STEP = 15

# MQTT (optional)
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC_PRED = "csi/prediction"

RETRY_OPEN_DELAY = 1.5

# global control
stop_event = threading.Event()
def signal_handler(sig, frame):
    print("\nStop signal received")
    stop_event.set()
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# MQTT
def init_mqtt():
    try:
        client = mqtt.Client()
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
        print(f"[MQTT] MQTT connected to {MQTT_BROKER}:{MQTT_PORT}")
        return client
    except Exception as e:
        print(f"[WARNING] MQTT init failed: {e}")
        return None

mqtt_client = init_mqtt()

# helper
def list_ports_print():
    ports = list(serial.tools.list_ports.comports())
    print("Detected serial ports:")
    for p in ports:
        print(f" - {p.device} : {p.description}")
    return ports

# worker
def log_serial_debug(label, port):
    # counters & stats
    total_lines = 0
    total_csi_lines = 0
    total_parsed = 0
    last_payload_len = None

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = os.path.join(SAVE_DIR, f"debug_{label}_{timestamp}.csv")
    print(f"[{label}] will write debug CSV -> {fname}")

    # open port
    ser = None
    while ser is None and not stop_event.is_set():
        try:
            ser = serial.Serial(port, BAUD_RATE, timeout=2)
            try:
                ser.flushInput(); ser.reset_input_buffer()
            except Exception:
                pass
            print(f"[{label}] OPEN {port} @ {BAUD_RATE}")
        except Exception as e:
            print(f"[{label}] open error {e}, retrying in {RETRY_OPEN_DELAY}s")
            time.sleep(RETRY_OPEN_DELAY)

    if ser is None:
        print(f"[{label}] exiting (no serial)")
        return

    # feature buffer / model
    feature_buffer = LiveFeatureBuffer(window_size=WINDOW_SIZE, step=STEP)

    with open(fname, "w", encoding="utf-8", buffering=1) as fh:
        fh.write("# iso_ts,raw_line,parsed_len,parsed_mean,parsed_std\n")
        while not stop_event.is_set():
            try:
                raw = ser.readline()
                if not raw:
                    continue
                total_lines += 1

                # decode safely
                try:
                    line = raw.decode("utf-8", errors="replace").strip()
                except Exception:
                    line = str(raw)[:200]

                # debug print trimmed
                if total_lines % 50 == 0:
                    print(f"[{label}] lines read: {total_lines}")

                # Only process lines containing CSI_DATA
                if "CSI_DATA" not in line:
                    # print very short preview occasionally
                    if total_lines % 200 == 0:
                        print(f"[DEBUG {label}] non-CSI line preview: {line[:120]}")
                    continue

                total_csi_lines += 1
                ts = datetime.datetime.now().isoformat(timespec="milliseconds")
                parsed = parse_csi_line(line)

                parsed_len = 0
                parsed_mean = ""
                parsed_std = ""

                if parsed is None:
                    # Save raw so we can inspect later
                    fh.write(f"{ts},{line},0,,\n")
                    print(f"[{label}] CSI_DATA found but parse failed (line #{total_csi_lines})")
                    continue

                payload = parsed.get("payload")
                if payload is None or not isinstance(payload, np.ndarray) or payload.size == 0:
                    fh.write(f"{ts},{line},0,,\n")
                    print(f"[{label}] parsed but empty payload")
                    continue

                total_parsed += 1
                parsed_len = int(payload.size)
                parsed_mean = float(np.mean(payload))
                parsed_std = float(np.std(payload))
                last_payload_len = parsed_len

                # log CSV and console
                fh.write(f"{ts},{line.replace(',', ';')},{parsed_len},{parsed_mean:.3f},{parsed_std:.3f}\n")
                print(f"[{label}] CSI parsed #{total_parsed} len={parsed_len} mean={parsed_mean:.2f} std={parsed_std:.2f}")

                # pass to features & predict
                feats, pred = feature_buffer.add_sample(payload)
                if feats is not None:
                    # show a compact summary
                    print(f"[{label}] -> FEATS (len={len(feats)}): {feats[:8]} ...")
                    # show model raw output (live_features prints debug already)
                    print(f"[{label}] -> PRED: {pred}")

                    # publish MQTT
                    if mqtt_client:
                        try:
                            mqtt_client.publish(MQTT_TOPIC_PRED, json.dumps({
                                "source": label,
                                "timestamp": ts,
                                "prediction": pred["pred"],
                                "confidence": pred["conf"]
                            }))
                        except Exception as e:
                            print(f"[{label}] mqtt pub error {e}")

            except Exception as e:
                print(f"[{label}] Loop Exception: {e}")
                break

    try:
        ser.close()
    except Exception:
        pass

    print(f"[{label}] finished: total_lines={total_lines} csi_lines={total_csi_lines} parsed={total_parsed} last_payload_len={last_payload_len}")

# run
if __name__ == "__main__":
    print("Starting debug capture")
    list_ports_print()
    threads = []
    for label, port in PORTS.items():
        t = threading.Thread(target=log_serial_debug, args=(label, port), daemon=False)
        t.start()
        threads.append(t)

    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        stop_event.set()

    for t in threads:
        t.join(timeout=2.0)

    if mqtt_client:
        try:
            mqtt_client.loop_stop(); mqtt_client.disconnect()
        except Exception:
            pass

    print("Debug capture ended")
