"""
Compress vector of 512 values into 8x64 matrix
Takes W1_q.mem and prints to W1_q_packed.mem
"""

from pathlib import Path

input_file = Path(__file__).parent.parent / "mem" / "W1_q_orig.mem"
output_file = Path(__file__).parent.parent / "mem" / "W1_q.mem"


with open(input_file, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

if len(lines) != 64 * 8:
    raise ValueError(f"Expected 512 lines (64x8), got {len(lines)}")

packed_lines = []
for node in range(8):
    node_weights = lines[node * 64: (node + 1) * 64]
    packed_line = "".join(node_weights)
    packed_lines.append(packed_line)

with open(output_file, "w") as f:
    for line in packed_lines:
        f.write(line + "\n")

print(f"Done! Packed {len(packed_lines)} lines into {output_file}")
