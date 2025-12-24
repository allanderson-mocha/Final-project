import subprocess
import time

SENDER_CABLE = "1"  # USB-Blaster [USB-0]
RECEIVER_CABLE = "2"  # USB-Blaster [USB-1]

print("Starting relay...")
print("Reading from FPGA Sender (USB-0) and forwarding to FPGA Receiver (USB-1)")

# Start terminal on FPGA sender
p_sender = subprocess.Popen(
    [
        r"C:\intelFPGA_lite\18.1\quartus\bin64\nios2-terminal.exe",
        "--cable",
        SENDER_CABLE,
        "--instance",
        "0",
    ],
    stdout=subprocess.PIPE,
    text=True,
)

# Start terminal on FPGA receiver
p_receiver = subprocess.Popen(
    [
        r"C:\intelFPGA_lite\18.1\quartus\bin64\nios2-terminal.exe",
        "--cable",
        RECEIVER_CABLE,
        "--instance",
        "0",
    ],
    stdin=subprocess.PIPE,
    text=True,
)

# Infinite relay loop
while True:
    line = p_sender.stdout.readline().strip()
    if line:
        print("Forwarding:", line)
        p_receiver.stdin.write(line + "\n")
        p_receiver.stdin.flush()
