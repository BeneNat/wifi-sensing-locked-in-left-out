# server/capture_csi_serial.py
import serial
import serial.tools.list_ports
import threading
import datetime
import os
import signal
import sys
import time

# -------- CONFIGURATION --------
# Update COM mapping to match Device Manager output
PORTS = {
    "AP": "COM3",   # change to actual AP COM
    "STA": "COM4"   # change to actual STA COM
}
#BAUD_RATE = 115200
BAUD_RATE = 921600
SAVE_DIR = "data_logs"
os.makedirs(SAVE_DIR, exist_ok=True)

# How many seconds to wait before reattempt to open a port on failure
RETRY_OPEN_DELAY = 2.0

# -------- STOP EVENT --------
stop_event = threading.Event()

def signal_handler(sig, frame):
    print("\n🛑 Stopping CSI logging...")
    stop_event.set()

signal.signal(signal.SIGINT, signal_handler)  # CTRL+C
signal.signal(signal.SIGTERM, signal_handler) # termination

def available_ports_text():
    ports = serial.tools.list_ports.comports()
    return ", ".join([p.device for p in ports])

# -------- FUNCTION TO READ SERIAL --------
def log_serial(label, port):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(SAVE_DIR, f"csi_{label}_{timestamp}.csv")
    print(f"[{label}] Trying to open {port} at {BAUD_RATE} baud. Available ports: {available_ports_text()}")
    ser = None

    # Try to open port with retries until stop_event is set
    while ser is None and not stop_event.is_set():
        try:
            ser = serial.Serial(port, BAUD_RATE, timeout=1)
            print(f"[{label}] Opened {port} -> logging to {filename}")
        except Exception as e:
            print(f"[{label}] Could not open {port}: {e}. Retrying in {RETRY_OPEN_DELAY}s")
            time.sleep(RETRY_OPEN_DELAY)

    if ser is None:
        print(f"[{label}] Exiting thread because stop_event set before open.")
        return

    # Ensure file is line-buffered and flush on each line
    with open(filename, "w", buffering=1, encoding="utf-8") as f:
        # write header for convenience
        f.write("# timestamp_iso, raw_line\n")
        f.flush()
        while not stop_event.is_set():
            try:
                raw = ser.readline()  # returns bytes or b'' on timeout
                if not raw:
                    continue
                try:
                    line = raw.decode('utf-8', errors='ignore').strip()
                except Exception:
                    line = raw.decode(errors='ignore').strip()
                if not line:
                    continue
                # Only log CSI_DATA lines (but you can log everything)
                if line.startswith("CSI_DATA"):
                    ts = datetime.datetime.now().isoformat(timespec="milliseconds")
                    f.write(f"{ts},{line}\n")
                    f.flush()
                    print(f"[{label}] {ts} {line[:120]}")  # print prefix to console
            except Exception as e:
                print(f"[{label}] Error while reading/writing: {e}")
                break

    try:
        ser.close()
    except Exception:
        pass
    print(f"[{label}] Thread exiting, file saved: {filename}")

# -------- START THREADS --------
def start_capture():
    threads = []
    for label, port in PORTS.items():
        t = threading.Thread(target=log_serial, args=(label, port), daemon=False)
        t.start()
        threads.append(t)

    print("🟢 CSI logging started for ports:", PORTS)
    print("Press Ctrl+C to stop logging and close files.")

    try:
        # Wait for stop_event set by signal handler
        while not stop_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        stop_event.set()

    # wait for all threads to finish
    for t in threads:
        t.join(timeout=5.0)

    print("✅ All threads stopped. CSV files saved in:", SAVE_DIR)

if __name__ == "__main__":
    # quick sanity: make sure pyserial is the right package
    if getattr(serial, 'Serial', None) is None:
        print("ERROR: pyserial not found or shadowed. Make sure 'pyserial' is installed and there's no local file named 'serial.py'.")
        sys.exit(1)

    start_capture()
