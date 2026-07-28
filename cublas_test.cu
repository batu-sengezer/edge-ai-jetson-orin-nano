#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>

double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main() {
    int N = 1024;
    size_t bytes = (size_t)N * N * sizeof(float);
    float *A = (float*)malloc(bytes), *B = (float*)malloc(bytes), *C = (float*)malloc(bytes);
    for (int i = 0; i < N*N; ++i) { A[i] = rand()/(float)RAND_MAX; B[i] = rand()/(float)RAND_MAX; }

    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytes); cudaMalloc(&dB, bytes); cudaMalloc(&dC, bytes);
    cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, bytes, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, dB, N, dA, N, &beta, dC, N);
    cudaDeviceSynchronize();

    double t0 = now_ms();
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, dB, N, dA, N, &beta, dC, N);
    cudaDeviceSynchronize();
    double elapsed = now_ms() - t0;

    printf("cuBLAS N=%d: %.3f ms\n", N, elapsed);

    cublasDestroy(handle);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(A); free(B); free(C);
    return 0;
}
