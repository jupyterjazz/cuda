#include <stdio.h>
#include <stdlib.h>

__global__
void matrix_addition_v1(float *A, float *B, float *C, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N){
       C[row * N + col] = A[row * N + col] + B[row * N + col];  // coalesced access
    }
}

__global__
void matrix_addition_v2(float *A, float *B, float *C, int N){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < N && col < N){
        C[row * N + col] = A[row * N + col] + B[row * N + col];  // strided access
    }
}


int main(){
    const int N = 1024;
    const int num_iterations = 100; // for experiments
    size_t size = N * N * sizeof(float);
    float *A = (float*)malloc(size);
    float *B = (float*)malloc(size);
    float *C1 = (float*)malloc(size);
    float *C2 = (float*)malloc(size);

    for (int i = 0; i < N * N; i++){
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
    }

    float *d_A, *d_B, *d_C1, *d_C2;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C1, size);
    cudaMalloc(&d_C2, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    
    dim3 block_size(16, 16);
    dim3 grid_size((N + block_size.x - 1) / block_size.x, (N + block_size.y - 1) / block_size.y);

    cudaEvent_t start_v1, stop_v1, start_v2, stop_v2;
    cudaEventCreate(&start_v1);
    cudaEventCreate(&stop_v1);
    cudaEventCreate(&start_v2);
    cudaEventCreate(&stop_v2);

    // warmup
    matrix_addition_v1<<<grid_size, block_size>>>(d_A, d_B, d_C1, N);
    matrix_addition_v2<<<grid_size, block_size>>>(d_A, d_B, d_C2, N);
    cudaDeviceSynchronize();

    // coalesced access
    cudaEventRecord(start_v1);
    for (int i = 0; i < num_iterations; i++) {
        matrix_addition_v1<<<grid_size, block_size>>>(d_A, d_B, d_C1, N);
    }
    cudaEventRecord(stop_v1);
    cudaEventSynchronize(stop_v1);

    // strided access
    cudaEventRecord(start_v2);
    for (int i = 0; i < num_iterations; i++) {
        matrix_addition_v2<<<grid_size, block_size>>>(d_A, d_B, d_C2, N);
    }
    cudaEventRecord(stop_v2);
    cudaEventSynchronize(stop_v2);

    // calculate times
    float time_v1, time_v2;
    cudaEventElapsedTime(&time_v1, start_v1, stop_v1);
    cudaEventElapsedTime(&time_v2, start_v2, stop_v2);
    float avg_time_v1 = time_v1 / num_iterations;
    float avg_time_v2 = time_v2 / num_iterations;

    printf("=== Performance Comparison ===\n");
    printf("Matrix size: %d x %d\n", N, N);
    printf("Number of iterations: %d\n", num_iterations);
    printf("Block size: %d x %d\n", block_size.x, block_size.y);
    printf("Grid size: %d x %d\n", grid_size.x, grid_size.y);
    printf("\n");
    printf("coalesced access:\n");
    printf("  Total time: %.4f ms\n", time_v1);
    printf("  Average time per kernel: %.4f ms\n", avg_time_v1);
    printf("\n");
    printf("strided access:\n");
    printf("  Total time: %.4f ms\n", time_v2);
    printf("  Average time per kernel: %.4f ms\n", avg_time_v2);
    printf("\n");
    printf("Performance ratio (coalesced/strided): %.2fx\n", avg_time_v2 / avg_time_v1);
    printf("\n");

    // verify that outputs match
    matrix_addition_v1<<<grid_size, block_size>>>(d_A, d_B, d_C1, N);
    cudaDeviceSynchronize();
    cudaMemcpy(C1, d_C1, size, cudaMemcpyDeviceToHost);

    matrix_addition_v2<<<grid_size, block_size>>>(d_A, d_B, d_C2, N);
    cudaDeviceSynchronize();
    cudaMemcpy(C2, d_C2, size, cudaMemcpyDeviceToHost);

    bool matching = true;
    for (int i = 0; i < N * N; i++){
        if (C1[i] != C2[i]){
            matching = false;
            printf("values at [%d][%d] do not match. %f != %f\n", i/N, i%N, C1[i], C2[i]);
            break;
        }
    }
    if (matching){
        printf("Both matrices match, kernels produce identical results\n");
    } else {
        printf("Returned matrices do not match\n");
    }

    cudaEventDestroy(start_v1);
    cudaEventDestroy(stop_v1);
    cudaEventDestroy(start_v2);
    cudaEventDestroy(stop_v2);

    free(A); free(B); free(C1); free(C2);
    cudaFree(d_A); cudaFree(d_B);
    cudaFree(d_C1); cudaFree(d_C2);

    return 0;
}