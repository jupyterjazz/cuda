#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16


__global__ void tiledMatMul(float *A, float *B, float *C, int width) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (width + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < width && (t * TILE_SIZE + tx) < width) {
            As[ty][tx] = A[row * width + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if ((t * TILE_SIZE + ty) < width && col < width) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * width + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < width && col < width) {
        C[row * width + col] = sum;
    }
}

void initMatrix(float *mat, int size) {
    for (int i = 0; i < size; ++i) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

int main() {
    int N = 1024;
    size_t size = N * N * sizeof(float);
    
    float *A = (float*)malloc(size);
    float *B = (float*)malloc(size);
    float *C = (float*)malloc(size);
    
    initMatrix(A, N * N);
    initMatrix(B, N * N);
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    
    printf("Grid size: %dx%d\n", gridSize.x, gridSize.y);
    printf("Block size: %dx%d\n", blockSize.x, blockSize.y);
    
    tiledMatMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    
    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    printf("done\n");
    return 0;
}
