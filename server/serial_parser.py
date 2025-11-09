import re
import numpy as np

_BRACKET_RE = re.compile(r'\[([^\]]+)\]')

_NUM_RE = re.compile(r'-?\d+(?:\.\d+)?')

def parse_csi_line(line):
    try:
        if not line:
            return None

        matches = _BRACKET_RE.findall(line)
        if not matches:
            return None

        content = matches[-1]
        nums = _NUM_RE.findall(content)
        if not nums:
            return None

        arr = [float(x) for x in nums]
        return {"payload": np.array(arr, dtype=np.float32)}

    except Exception as e:
        # print(f"[ERROR] parse_csi_line error: {e}, line={line[:200]}")
        return None
