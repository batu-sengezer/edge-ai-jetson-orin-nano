#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cmath>

#define TILE 16

// Version A: CPU baseline (plain triple loop)
void matmul_cpu(const float* A, const float* B, float* C, int N) {
    for (int row = 0; row < N; ++row)
        for (int col = 0; col < N; ++col) {
            float s = 0.0f;
            for (int k = 0; k < N; ++k) s += A[row*N+k] * B[k*N+col];
            C[row*N+col] = s;
        }
}

// Version B: naive CUDA - one thread per output element
__global__ void matmul_naive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float s = 0.0f;
        for (int k = 0; k < N; ++k) s += A[row*N+k] * B[k*N+col];
        C[row*N+col] = s;
    }
}

// Version C: tiled CUDA - shared-memory tiling
__global__ void matmul_tiled(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y*TILE + threadIdx.y;
    int col = blockIdx.x*TILE + threadIdx.x;
    float s = 0.0f;
    for (int t = 0; t < (N+TILE-1)/TILE; ++t) {
        // cooperatively load one tile of A and B into shared memory
        if (row < N && t*TILE+threadIdx.x < N)
            As[threadIdx.y][threadIdx.x] = A[row*N + t*TILE+threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;
        if (col < N && t*TILE+threadIdx.y < N)
            Bs[threadIdx.y][threadIdx.x] = B[(t*TILE+threadIdx.y)*N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; ++k) s += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < N && col < N) C[row*N+col] = s;
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main() {
    int sizes[] = {256, 512, 1024, 2048};
    for (int si = 0; si < 4; ++si) {
        int N = sizes[si];
        size_t bytes = (size_t)N*N*sizeof(float);
        float *A = (float*)malloc(bytes), *B = (float*)malloc(bytes);
        float *C_cpu = (float*)malloc(bytes), *C_gpu = (float*)malloc(bytes);
        for (int i = 0; i < N*N; ++i) { A[i] = rand()/(float)RAND_MAX; B[i] = rand()/(float)RAND_MAX; }

        // ---- CPU (skip for large N, too slow) ----
        double cpu_ms = -1;
        if (N <= 1024) {
            double t0 = now_ms();
            matmul_cpu(A, B, C_cpu, N);
            cpu_ms = now_ms() - t0;
        }

        float *dA, *dB, *dC;
        cudaMalloc(&dA, bytes); cudaMalloc(&dB, bytes); cudaMalloc(&dC, bytes);
        cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, B, bytes, cudaMemcpyHostToDevice);
        dim3 block(TILE, TILE);
        dim3 grid((N+TILE-1)/TILE, (N+TILE-1)/TILE);

        // ---- naive CUDA ----
        matmul_naive<<<grid, block>>>(dA, dB, dC, N); cudaDeviceSynchronize(); // warmup
        double t1 = now_ms();
        matmul_naive<<<grid, block>>>(dA, dB, dC, N); cudaDeviceSynchronize();
        double naive_ms = now_ms() - t1;

        // ---- tiled CUDA ----
        matmul_tiled<<<grid, block>>>(dA, dB, dC, N); cudaDeviceSynchronize(); // warmup
        double t2 = now_ms();
        matmul_tiled<<<grid, block>>>(dA, dB, dC, N); cudaDeviceSynchronize();
        double tiled_ms = now_ms() - t2;
        cudaMemcpy(C_gpu, dC, bytes, cudaMemcpyDeviceToHost);

        printf("N=%4d | CPU: %10.3f ms | naive: %8.3f ms | tiled: %8.3f ms",
               N, cpu_ms, naive_ms, tiled_ms);
        if (cpu_ms > 0) printf(" | naive speedup: %.1fx", cpu_ms/naive_ms);
        printf(" | tiled/naive: %.2fx\n", naive_ms/tiled_ms);

        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        free(A); free(B); free(C_cpu); free(C_gpu);
    }
    return 0;
}
