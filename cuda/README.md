# CUDA MLP IMPLEMENTATION

## Description

This phase implements the same 64 → 8 → 10 MLP forward pass used in Phase 1 (Python) and Phase 2 (FPGA), but now optimized to run on the GPU using CUDA.
We compare correctness, probability distribution, runtime, and predicted class against the Python reference implementation.

The CUDA version mirrors the FPGA architecture but uses GPU parallelism and shared-memory tiling to accelerate the two matrix multiplications.

## Functionality

The CUDA implementation performs the same forward pass as Python but accelerates each matrix multiplication using tiled shared-memory GEMM, a standard high-performance pattern used in cuBLAS-like libraries.

#### GEMM Kernel (X @ W1 and H @ W2)

The kernel:

- Splits matrices into 16×16 tiles
- Loads tiles into \_\_shared\_\_ memory
- Runs inner products in parallel across blocks
- Writes the final output to global memory

This provides much higher memory efficiency than naïve per-thread loops.

#### Bias + ReLU Kernel

After the first GEMM:

H[row, col] = max(0, H[row, col] + b1[col])

This is equivalent to the Python implementation:

hidden = np.maximum(X @ W1 + b1, 0)

#### Second GEMM + Bias

Same as layer 1 but without ReLU.

#### Argmax Kernel

Computes:
predicted_class = argmax(logits)

Matches Python exactly.

## Files

- phase3.cu
- inputs.h
- W1_q.mem
- W2_q.mem
- b1_q.mem
- b2_q.mem
- mlp

## Compilation

At the top of a Colab cell add before pasting .cu code:

```python
%%writefile phase3_gemm.cu
```

Then compile and run:

```bash
nvcc -arch=sm_75 -O3 phase3_gemm.cu -o mlp
./mlp
```

## Running

You will see:

- Hidden layer output
- Logits
- Predicted class
- Python reference label
- Match/mismatch
- Total execution time
