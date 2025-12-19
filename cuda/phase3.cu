#include <cstdio>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

#include "inputs.h"   // X_sample[64], PYTHON_LABEL

// Problem sizes
#define INPUT_DIM   64
#define HIDDEN_DIM  8
#define OUTPUT_DIM  10
#define BATCH       1   // single input

// Tiling parameters
#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

// ============================
// GEMM kernel
// ============================
__global__ void gemm_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    float acc = 0.0f;

    int numTiles = (K + TILE_K - 1) / TILE_K;
    for (int t = 0; t < numTiles; ++t) {
        int tiled_k_A = t * TILE_K + threadIdx.x;
        int tiled_k_B = t * TILE_K + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < M && tiled_k_A < K) ? A[row*K + tiled_k_A] : 0.0f;

        Bs[threadIdx.y][threadIdx.x] =
            (tiled_k_B < K && col < N) ? B[tiled_k_B*N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_K; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row*N + col] = acc;
}

// ============================
__global__ void add_bias_relu(
    float* __restrict__ H,
    const float* __restrict__ b,
    int M, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    int idx = row * N + col;
    float v = H[idx] + b[col];
    H[idx] = v > 0.0f ? v : 0.0f;
}

__global__ void add_bias(
    float* __restrict__ C,
    const float* __restrict__ b,
    int M, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    C[row*N + col] += b[col];
}

__global__ void argmax_batch(
    const float* __restrict__ logits,
    int* __restrict__ class_idx,
    int M, int N)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;

    const float* row = &logits[m*N];
    float max_val = row[0];
    int max_i = 0;
    for (int i = 1; i < N; ++i)
        if (row[i] > max_val) { max_val = row[i]; max_i = i; }

    class_idx[m] = max_i;
}

bool load_mem_int8(const char* filename, int8_t* dst, int count) {
    printf("Opening %s ...\n", filename);

    FILE* f = fopen(filename, "r");
    if (!f) {
        printf("Could not open file!\n");
        return false;
    }

    for (int i = 0; i < count; i++) {
        unsigned int v;
        int r = fscanf(f, "%x", &v);

        if (r != 1) {
            printf("Parse error at index %d (r=%d)\n", i, r);
            return false;
        }

        dst[i] = (int8_t)(v & 0xFF);
    }

    fclose(f);
    return true;
}


bool load_mem_int32(const char* filename, int32_t* dst, int count) {
    FILE* f = fopen(filename, "r");
    if (!f) return false;
    for (int i = 0; i < count; i++) {
        unsigned int v;
        if (fscanf(f, "%x", &v) != 1) return false;
        dst[i] = (int32_t)v;
    }
    fclose(f);
    return true;
}


// ============================
// Host
// ============================
int main() {
    float X_host[BATCH * INPUT_DIM];
    float H_host[BATCH * HIDDEN_DIM];
    float logits_host[BATCH * OUTPUT_DIM];
    int   class_idx_host[BATCH];

    for (int b = 0; b < BATCH; ++b)
        std::memcpy(&X_host[b*INPUT_DIM], X_sample,
                    INPUT_DIM * sizeof(float));

    // Device malloc
    float *d_X, *d_W1, *d_W2, *d_H, *d_logits;
    float *d_b1, *d_b2;
    int *d_class_idx;

    cudaMalloc(&d_X, INPUT_DIM*sizeof(float));
    cudaMalloc(&d_W1, INPUT_DIM*HIDDEN_DIM*sizeof(float));
    cudaMalloc(&d_W2, HIDDEN_DIM*OUTPUT_DIM*sizeof(float));
    cudaMalloc(&d_H, HIDDEN_DIM*sizeof(float));
    cudaMalloc(&d_logits, OUTPUT_DIM*sizeof(float));
    cudaMalloc(&d_b1, HIDDEN_DIM*sizeof(float));
    cudaMalloc(&d_b2, OUTPUT_DIM*sizeof(float));
    cudaMalloc(&d_class_idx, sizeof(int));

    // ---------------- Load Quantized MEM Files ----------------
    int8_t  W1_q[64*8];
    int32_t b1_q[8];
    int8_t  W2_q[8*10];
    int32_t b2_q[10];

    if (!load_mem_int8("./W1_q_orig.mem", W1_q, 64*8)) { printf("Failed W1_q.mem\n"); return -1; }
    if (!load_mem_int32("./b1_q.mem", b1_q, 8))  { printf("Failed b1_q.mem\n"); return -1; }
    if (!load_mem_int8("./W2_q.mem", W2_q, 8*10)) { printf("Failed W2_q.mem\n"); return -1; }
    if (!load_mem_int32("./b2_q.mem", b2_q, 10))  { printf("Failed b2_q.mem\n"); return -1; }

    float W1_f[64*8];
    float b1_f[8];
    float W2_f[8*10];
    float b2_f[10];

    for (int i = 0; i < 64*8; i++) W1_f[i] = (float)W1_q[i];
    for (int i = 0; i < 8; i++)     b1_f[i] = (float)b1_q[i];
    for (int i = 0; i < 8*10; i++)  W2_f[i] = (float)W2_q[i];
    for (int i = 0; i < 10; i++)    b2_f[i] = (float)b2_q[i];



    cudaMemcpy(d_X, X_host, INPUT_DIM*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, W1_f, 64*8 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, b1_f, 8 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, W2_f, 8*10 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, b2_f, 10 * sizeof(float), cudaMemcpyHostToDevice);


    dim3 blockG(TILE_N, TILE_M);
    dim3 gridG_hidden((HIDDEN_DIM + TILE_N - 1)/TILE_N,
                      (BATCH + TILE_M - 1)/TILE_M);
    dim3 gridG_out((OUTPUT_DIM + TILE_N - 1)/TILE_N,
                   (BATCH + TILE_M - 1)/TILE_M);

    dim3 blockA(16, 16);
    dim3 gridA_hidden((HIDDEN_DIM + 15)/16, (BATCH + 15)/16);
    dim3 gridA_out((OUTPUT_DIM + 15)/16, (BATCH + 15)/16);

    dim3 blockArg(128);
    dim3 gridArg((BATCH + 127)/128);

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    gemm_tiled_kernel<<<gridG_hidden, blockG>>>(d_X, d_W1, d_H, 1, 8, 64);
    add_bias_relu<<<gridA_hidden, blockA>>>(d_H, d_b1, 1, 8);

    gemm_tiled_kernel<<<gridG_out, blockG>>>(d_H, d_W2, d_logits, 1, 10, 8);
    add_bias<<<gridA_out, blockA>>>(d_logits, d_b2, 1, 10);

    argmax_batch<<<gridArg, blockArg>>>(d_logits, d_class_idx, 1, 10);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(H_host,      d_H,      8*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(logits_host, d_logits, 10*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(class_idx_host, d_class_idx, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Kernel error: %s\n", cudaGetErrorString(cudaGetLastError()));
    printf("Total inference time = %.6f ms\n\n", ms);

    // Softmax probabilities
    float probs[10];
    float maxLog = logits_host[0];
    for (int i = 1; i < 10; i++)
        if (logits_host[i] > maxLog) maxLog = logits_host[i];

    float sum = 0.0f;
    for (int i = 0; i < 10; i++) {
        probs[i] = expf(logits_host[i] - maxLog);
        sum += probs[i];
    }
    for (int i = 0; i < 10; i++)
        probs[i] /= sum;

    printf("Probabilities:\n");
    for (int i = 0; i < 10; i++)
        printf("%.6f ", probs[i]);
    printf("\n\n");

    // Accuracy for 1 input
    int predicted = class_idx_host[0];
    printf("Predicted class = %d\n", predicted);
    printf("Python reference label = %d\n", PYTHON_LABEL);

    if (predicted == PYTHON_LABEL)
        printf("Accuracy (1 sample) = 100%%\n");
    else
        printf("Accuracy (1 sample) = 0%%\n");

    // Cleanup
    cudaFree(d_X);
    cudaFree(d_W1);
    cudaFree(d_W2);
    cudaFree(d_H);
    cudaFree(d_logits);
    cudaFree(d_b1);
    cudaFree(d_b2);
    cudaFree(d_class_idx);

    return 0;
}
