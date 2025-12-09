#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

#include "weights.h"  // W1[64*8], b1[8], W2[8*10], b2[10]
#include "inputs.h"   // X_sample[64]  (for now: single sample)

// Problem sizes
#define INPUT_DIM   64
#define HIDDEN_DIM  8
#define OUTPUT_DIM  10
#define BATCH       1   // can increase later (M dimension)

// Tiling parameters
#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

// ============================
// Tiled shared-memory GEMM
// C[M x N] = A[M x K] * B[K x N]
// Row-major: A[m*K + k], B[k*N + n], C[m*N + n]
// ============================
__global__ void gemm_tiled_kernel(
    const float* __restrict__ A,  // [M, K]
    const float* __restrict__ B,  // [K, N]
    float* __restrict__ C,        // [M, N]
    int M, int N, int K)
{
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    int row = blockIdx.y * TILE_M + threadIdx.y; // m
    int col = blockIdx.x * TILE_N + threadIdx.x; // n

    float acc = 0.0f;

    // Loop over tiles of K dimension
    int numTiles = (K + TILE_K - 1) / TILE_K;
    for (int t = 0; t < numTiles; ++t) {
        int tiled_k_A = t * TILE_K + threadIdx.x; // column index for A tile
        int tiled_k_B = t * TILE_K + threadIdx.y; // row index for B tile

        // Load tile of A into shared mem
        if (row < M && tiled_k_A < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + tiled_k_A];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile of B into shared mem
        if (tiled_k_B < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[tiled_k_B * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product for this tile
        for (int k = 0; k < TILE_K; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// ============================
// Bias + ReLU (for hidden layer)
// ============================
__global__ void add_bias_relu(
    float* __restrict__ H,        // [M, N]
    const float* __restrict__ b,  // [N]
    int M, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    int idx = row * N + col;
    float v = H[idx] + b[col];
    H[idx] = v > 0.0f ? v : 0.0f;
}

// ============================
// Bias only (for output layer)
// ============================
__global__ void add_bias(
    float* __restrict__ C,        // [M, N]
    const float* __restrict__ b,  // [N]
    int M, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    int idx = row * N + col;
    C[idx] += b[col];
}

// ============================
// Argmax per sample (batch)
// class_idx[m] = argmax_n logits[m, n]
// ============================
__global__ void argmax_batch(
    const float* __restrict__ logits,  // [M, N]
    int* __restrict__ class_idx,
    int M, int N)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;

    const float* row = &logits[m * N];
    float max_val = row[0];
    int max_i = 0;
    for (int i = 1; i < N; ++i) {
        if (row[i] > max_val) {
            max_val = row[i];
            max_i = i;
        }
    }
    class_idx[m] = max_i;
}

// ============================
// Host test
// ============================
int main() {
    // ------------ Host buffers -------------
    // We’ll support batch, but for now BATCH=1 and reuse X_sample for every row.
    float X_host[BATCH * INPUT_DIM];
    float H_host[BATCH * HIDDEN_DIM];
    float logits_host[BATCH * OUTPUT_DIM];
    int   class_idx_host[BATCH];

    // Copy weights from headers
    // W1: [64, 8], row-major -> K=64, N=8
    // W2: [8, 10], row-major -> K=8,  N=10
    // b1: [8], b2: [10]
    // (we’ll copy directly to device later; no need for host copies beyond this)

    // Initialize X_host: for now, replicate X_sample for each batch row
    for (int b = 0; b < BATCH; ++b) {
        std::memcpy(&X_host[b * INPUT_DIM], X_sample, INPUT_DIM * sizeof(float));
    }

    // ------------ Device buffers -------------
    float *d_X, *d_W1, *d_W2, *d_H, *d_logits;
    float *d_b1, *d_b2;
    int   *d_class_idx;

    cudaMalloc(&d_X,      BATCH * INPUT_DIM   * sizeof(float));
    cudaMalloc(&d_W1,     INPUT_DIM * HIDDEN_DIM * sizeof(float)); // 64x8
    cudaMalloc(&d_W2,     HIDDEN_DIM * OUTPUT_DIM * sizeof(float)); // 8x10
    cudaMalloc(&d_H,      BATCH * HIDDEN_DIM  * sizeof(float));
    cudaMalloc(&d_logits, BATCH * OUTPUT_DIM  * sizeof(float));
    cudaMalloc(&d_b1,     HIDDEN_DIM * sizeof(float));
    cudaMalloc(&d_b2,     OUTPUT_DIM * sizeof(float));
    cudaMalloc(&d_class_idx, BATCH * sizeof(int));

    // Copy host -> device
    cudaMemcpy(d_X,  X_host, sizeof(X_host), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, W1,     INPUT_DIM * HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, W2,     HIDDEN_DIM * OUTPUT_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, b1,     HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, b2,     OUTPUT_DIM * sizeof(float), cudaMemcpyHostToDevice);

    // ------------ Launch configuration -------------
    dim3 blockG(TILE_N, TILE_M); // 16x16
    // Hidden GEMM: [BATCH x INPUT_DIM] * [INPUT_DIM x HIDDEN_DIM]
    int M_h = BATCH;
    int K_h = INPUT_DIM;
    int N_h = HIDDEN_DIM;
    dim3 gridG_hidden((N_h + TILE_N - 1) / TILE_N,
                      (M_h + TILE_M - 1) / TILE_M);

    // Output GEMM: [BATCH x HIDDEN_DIM] * [HIDDEN_DIM x OUTPUT_DIM]
    int M_o = BATCH;
    int K_o = HIDDEN_DIM;
    int N_o = OUTPUT_DIM;
    dim3 gridG_out((N_o + TILE_N - 1) / TILE_N,
                   (M_o + TILE_M - 1) / TILE_M);

    // Bias/activation grids
    dim3 blockA(16, 16);
    dim3 gridA_hidden((N_h + blockA.x - 1) / blockA.x,
                      (M_h + blockA.y - 1) / blockA.y);
    dim3 gridA_out((N_o + blockA.x - 1) / blockA.x,
                   (M_o + blockA.y - 1) / blockA.y);

    // Argmax config
    dim3 blockArg(128);
    dim3 gridArg((BATCH + blockArg.x - 1) / blockArg.x);

    // ------------ Timing setup -------------
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // 1) Hidden layer: GEMM (X @ W1) then bias+ReLU
    gemm_tiled_kernel<<<gridG_hidden, blockG>>>(d_X, d_W1, d_H, M_h, N_h, K_h);
    add_bias_relu<<<gridA_hidden, blockA>>>(d_H, d_b1, M_h, N_h);

    // 2) Output layer: GEMM (H @ W2) then bias
    gemm_tiled_kernel<<<gridG_out, blockG>>>(d_H, d_W2, d_logits, M_o, N_o, K_o);
    add_bias<<<gridA_out, blockA>>>(d_logits, d_b2, M_o, N_o);

    // 3) Argmax per sample
    argmax_batch<<<gridArg, blockArg>>>(d_logits, d_class_idx, BATCH, OUTPUT_DIM);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    // Check for kernel errors
    cudaDeviceSynchronize();
    printf("Kernel error: %s\n", cudaGetErrorString(cudaGetLastError()));
    printf("Total inference time (BATCH=%d) = %.6f ms\n", BATCH, ms);

    // ------------ Copy back and print one sample -------------
    cudaMemcpy(H_host,        d_H,      BATCH * HIDDEN_DIM * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(logits_host,   d_logits, BATCH * OUTPUT_DIM * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(class_idx_host,d_class_idx, BATCH * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Hidden[0]:\n");
    for (int i = 0; i < HIDDEN_DIM; i++)
        printf("%f ", H_host[i]);
    printf("\n\nLogits[0]:\n");
    for (int i = 0; i < OUTPUT_DIM; i++)
        printf("%f ", logits_host[i]);
    printf("\nPredicted class[0] = %d\n", class_idx_host[0]);

    // ------------ Cleanup -------------
    cudaFree(d_X);
    cudaFree(d_W1);
    cudaFree(d_W2);
    cudaFree(d_H);
    cudaFree(d_logits);
    cudaFree(d_b1);
    cudaFree(d_b2);
    cudaFree(d_class_idx);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}