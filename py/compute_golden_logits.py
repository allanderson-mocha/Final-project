import sys
from pathlib import Path

import numpy as np


def load_signed_mem(path: Path, bit_width: int) -> np.ndarray:
    """
    Loads values from mem file into numpy array.

    Parameters
    ----------
    path : Path
        Path of local .mem file.
    bit_width : int
        Expected bit width of values.

    Returns
    -------
    vals : np.ndarray
    """
    vals = []
    with open(path, "r") as file:
        for line in file:
            stripped_line = line.strip()
            if not stripped_line:
                continue

            # Handle negative number interpretation for Python
            if stripped_line.startswith("-"):
                val = -int(stripped_line[1:], 16)
            else:
                val = int(stripped_line, 16)

            # Convert to signed range e.g. int8 → -128..127
            max_val = 2 ** (bit_width - 1)
            val = (val + max_val) % (2 * max_val) - max_val
            vals.append(val)

    return np.array(vals)


# Parameters
IN_DIM = 64
HIDDEN = 8
OUT = 10
DATA_W = 8
ACC_W_BIAS = 40

base = Path(".")
W1_path = base / "W1_q.mem"  # 64 x 8, int8
b1_path = base / "b1_q.mem"  # 8 entries, int32
W2_path = base / "W2_q.mem"  # 8 x 10, int8
b2_path = base / "b2_q.mem"  # 10 entries, int32

if (
    not W1_path.exists()
    or not b1_path.exists()
    or not W2_path.exists()
    or not b2_path.exists()
):
    print("ERROR: could not find one of the mem files in current directory.")
    print("Expected: W1_q.mem, b1_q.mem, W2_q.mem, b2_q.mem")
    sys.exit(1)

W1_flat = load_signed_mem(W1_path, 8)
b1_flat = load_signed_mem(b1_path, 32)
W2_flat = load_signed_mem(W2_path, 8)
b2_flat = load_signed_mem(b2_path, 32)

assert W1_flat.size == IN_DIM * HIDDEN, f"W1 size mismatch: {W1_flat.size}"
assert b1_flat.size == HIDDEN, f"b1 size mismatch: {b1_flat.size}"
assert W2_flat.size == HIDDEN * OUT, f"W2 size mismatch: {W2_flat.size}"
assert b2_flat.size == OUT, f"b2 size mismatch: {b2_flat.size}"

W1 = W1_flat.reshape(IN_DIM, HIDDEN)
b1 = b1_flat.reshape(HIDDEN)
W2 = W2_flat.reshape(HIDDEN, OUT)
b2 = b2_flat.reshape(OUT)


def rtl_like_inference(bus_in_bytes: np.ndarray) -> dict:
    """
    Simulate RTL logic for one layer forward-pass MLP.

    Parameters
    ----------
    bus_in_bytes : np.ndarray
        Inputs of DATA_W bits.

    Returns
    -------
    inference_log : dict
        Log of values passed throughout the network.
    """
    Xq = np.array(bus_in_bytes, dtype=np.int64)
    Xq_signed = Xq.copy()
    Xq_signed[Xq_signed >= 128] -= 256

    Z1 = (Xq_signed.astype(np.int64) @ W1.astype(np.int64)) + b1.astype(np.int64)

    # ReLU activation
    A1 = np.maximum(Z1, 0)

    # Truncate to 8 bits
    mask = (1 << DATA_W) - 1  # 0xFF
    hidden_out_hw = (A1.astype(np.int64) & mask).astype(np.int64)
    hidden_out_hw[hidden_out_hw >= (1 << (DATA_W - 1))] -= (
        1 << DATA_W
    )  # Convert to signed

    # Outer layer logits
    Z2 = (hidden_out_hw.astype(np.int64) @ W2.astype(np.int64)) + b2.astype(np.int64)

    return {
        "Xq_signed": Xq_signed,
        "Z1_int32": Z1.astype(np.int64),
        "A1_int32": A1.astype(np.int64),
        "hidden_out_hw": hidden_out_hw.astype(np.int64),
        "Z2_int32": Z2.astype(np.int64),
    }


dummy_bus = np.array([127] * IN_DIM, dtype=np.int64)
res = rtl_like_inference(dummy_bus)

print("=== GOLDEN (DUMMY) TEST: bus_in = all 127 ===")
print("Hidden pre-accum (Z1_int32):")
print(res["Z1_int32"].tolist())
print("\nHidden post-ReLU (A1_int32):")
print(res["A1_int32"].tolist())
print("\nHidden outputs seen by RTL (signed 8-bit):")
print(res["hidden_out_hw"].tolist())
print(
    "\nFinal integer logits (Z2_int32) -- paste these into your testbench as expected values:"
)
print(res["Z2_int32"].tolist())
