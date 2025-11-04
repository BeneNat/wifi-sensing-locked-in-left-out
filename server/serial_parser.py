# server/serial_parser.py
import re
import numpy as np

# find all bracket groups and return the last one (actual payload)
_BRACKET_RE = re.compile(r'\[([^\]]+)\]')

# capture signed ints and floats, e.g. -12, 34, 3.14, -0.5
_NUM_RE = re.compile(r'-?\d+(?:\.\d+)?')

def parse_csi_line(line):
    """
    Extract numeric CSI payload from a single 'CSI_DATA..., [...numbers...]' line.
    Returns {"payload": np.ndarray} or None.
    Robust: uses the last bracket pair, extracts only numeric tokens (ints/floats).
    """
    try:
        if not line:
            return None

        matches = _BRACKET_RE.findall(line)
        if not matches:
            return None

        content = matches[-1]  # take the last bracket group (most likely the payload)
        # extract numeric tokens only
        nums = _NUM_RE.findall(content)
        if not nums:
            return None

        arr = [float(x) for x in nums]
        return {"payload": np.array(arr, dtype=np.float32)}

    except Exception as e:
        # keep parser robust: return None on error
        # you can enable a debug print if you want to inspect bad lines:
        # print(f"⚠️ parse_csi_line error: {e}, line={line[:200]}")
        return None
