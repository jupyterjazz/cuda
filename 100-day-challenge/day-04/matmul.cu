#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__
void matmul(float *A, float *B, float *C, int N){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0.0f;
    if (row < N && col < N){
        for (int k = 0; k < N; k++){
            sum += A[row * N + k] * B[N * k + col];
        }
        C[row * N + col] = sum;
    }
}


int main(){
    int N = 1024;
    size_t size = N * N * sizeof(float);
    float *A = (float*)malloc(size);
    float *B = (float*)malloc(size);
    float *C = (float*)malloc(size);

    for (int i = 0; i < N * N; i++){
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    dim3 block_size(16, 16);
    dim3 grid_size((N + block_size.x - 1) / block_size.x, (N + block_size.y - 1) / block_size.y);

    matmul<<<grid_size, block_size>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    
    printf("Matrix multiplication finished\n");
    
    int random_row = rand() % N;
    int random_col = rand() % N;
    
    printf("Random row %d from matrix A:\n", random_row);
    for (int i = 0; i < N; i++){
        printf("%f ", A[random_row * N + i]);
    }
    printf("\n\n");
    
    printf("Random column %d from matrix B:\n", random_col);
    for (int i = 0; i < N; i++){
        printf("%f ", B[N * i + random_col]);
    }
    printf("\n\n");
    
    printf("Result in C[%d][%d] = %f\n", random_row, random_col, C[random_row * N + random_col]);
    printf("\n");

    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}