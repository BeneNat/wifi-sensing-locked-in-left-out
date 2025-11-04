# tutorial_1_5_parse_robust.py
import re
import os

DATA_DIR = "data_utf8"

def safe_parse_int_list(s):
    """Parse list of ints safely; skip malformed tokens."""
    values = []
    for token in s.split():
        token = token.strip()
        if not token:
            continue
        # accept only tokens like -12, 45, 0
        if re.match(r"^-?\d+$", token):
            try:
                values.append(int(token))
            except ValueError:
                continue
    return values

def parse_csi_from_file(file_path):
    csi_pattern = re.compile(r"\[([-\d\s]+)\]")
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if "CSI_DATA" not in line:
                continue
            match = csi_pattern.search(line)
            if not match:
                continue
            nums = safe_parse_int_list(match.group(1))
            # Skip if empty or obviously broken
            if len(nums) < 10:
                continue
            samples.append(nums)
    return samples

# ---- main test ----
for fname in os.listdir(DATA_DIR):
    if not fname.endswith(".csv"):
        continue
    path = os.path.join(DATA_DIR, fname)
    try:
        samples = parse_csi_from_file(path)
        if not samples:
            print(f"⚠️  {fname}: No valid CSI samples found")
        else:
            print(f"✅ {fname}: Parsed {len(samples)} CSI packets, first shape: {len(samples[0])}")
    except Exception as e:
        print(f"❌ {fname}: Error {e}")
