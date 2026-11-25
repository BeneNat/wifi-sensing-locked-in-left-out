import os

input_dir = "data_raw"  # Folder for captured training csv
output_dir = "data_utf8"    # Output folder qith fixed encoding
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if not fname.endswith(".csv"):
        continue

    src = os.path.join(input_dir, fname)
    dst = os.path.join(output_dir, fname)

    with open(src, "r", encoding="utf-16") as f:
        data = f.read()
    
    data = data.replace("\x00", "")

    with open(dst, "w", encoding="utf-8") as f:
        f.write(data)

    print(f"Converted {fname} -> UTF-8")