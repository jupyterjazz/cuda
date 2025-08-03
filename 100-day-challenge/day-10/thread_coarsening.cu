#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define TILE_SIZE 16
#define COARSE_FACTOR 4

__global__ void tiledMatMulCoars(float *A, float *B, float *C, int width) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * blockDim.y + ty;
    int colStart = bx * blockDim.x * COARSE_FACTOR + tx;
    
    float sum[COARSE_FACTOR];
    for (int i = 0; i < COARSE_FACTOR; i++){
        sum[i] = 0.0f;
    }
    
    for (int t = 0; t < (width + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < width && (t * TILE_SIZE + tx) < width) {
            As[ty][tx] = A[row * width + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        for (int i = 0; i < COARSE_FACTOR; i++){
            int col = colStart + i * TILE_SIZE;

            if ((t * TILE_SIZE + ty) < width && col < width) {
                Bs[ty][tx] = B[(t * TILE_SIZE + ty) * width + col];
            } else {
                Bs[ty][tx] = 0.0f;
            }
            __syncthreads();
            
            for (int k = 0; k < TILE_SIZE; ++k) {
                sum[i] += As[ty][k] * Bs[k][tx];
            }
            
            __syncthreads();
        }
    }
    for (int i = 0; i < COARSE_FACTOR; i++){
        int col = colStart + i * TILE_SIZE;
        if (row < width && col < width) {
            C[row * width + col] = sum[i];
        }
    }   
}


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
    int N = 2048;
    const int num_iterations = 100; // for experiments
    size_t size = N * N * sizeof(float);
    
    float *A = (float*)malloc(size);
    float *B = (float*)malloc(size);
    float *C_regular = (float*)malloc(size);
    float *C_coarsened = (float*)malloc(size);
    
    initMatrix(A, N * N);
    initMatrix(B, N * N);
    
    float *d_A, *d_B, *d_C_regular, *d_C_coarsened;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C_regular, size);
    cudaMalloc(&d_C_coarsened, size);
    
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize_regular((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    
    dim3 gridSize_coarsened((N + TILE_SIZE * COARSE_FACTOR - 1) / (TILE_SIZE * COARSE_FACTOR), 
                           (N + TILE_SIZE - 1) / TILE_SIZE);
    
    printf("Matrix size: %d x %d\n", N, N);
    printf("Number of iterations: %d\n", num_iterations);
    printf("Tile size: %d\n", TILE_SIZE);
    printf("Coarse factor: %d\n", COARSE_FACTOR);
    printf("Block size: %dx%d\n", blockSize.x, blockSize.y);
    printf("Regular kernel - Grid size: %dx%d\n", gridSize_regular.x, gridSize_regular.y);
    printf("Coarsened kernel - Grid size: %dx%d\n", gridSize_coarsened.x, gridSize_coarsened.y);
    printf("\n");
    
    cudaEvent_t start_regular, stop_regular, start_coarsened, stop_coarsened;
    cudaEventCreate(&start_regular);
    cudaEventCreate(&stop_regular);
    cudaEventCreate(&start_coarsened);
    cudaEventCreate(&stop_coarsened);
    
    // warmup
    tiledMatMul<<<gridSize_regular, blockSize>>>(d_A, d_B, d_C_regular, N);
    tiledMatMulCoars<<<gridSize_coarsened, blockSize>>>(d_A, d_B, d_C_coarsened, N);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start_regular);
    for (int i = 0; i < num_iterations; i++) {
        tiledMatMul<<<gridSize_regular, blockSize>>>(d_A, d_B, d_C_regular, N);
    }
    cudaEventRecord(stop_regular);
    cudaEventSynchronize(stop_regular);
    
    cudaEventRecord(start_coarsened);
    for (int i = 0; i < num_iterations; i++) {
        tiledMatMulCoars<<<gridSize_coarsened, blockSize>>>(d_A, d_B, d_C_coarsened, N);
    }
    cudaEventRecord(stop_coarsened);
    cudaEventSynchronize(stop_coarsened);
    
    // calculate time
    float time_regular, time_coarsened;
    cudaEventElapsedTime(&time_regular, start_regular, stop_regular);
    cudaEventElapsedTime(&time_coarsened, start_coarsened, stop_coarsened);
    float avg_time_regular = time_regular / num_iterations;
    float avg_time_coarsened = time_coarsened / num_iterations;
    
    printf("\n=== Performance Results ===\n");
    printf("Regular tiled matrix multiplication:\n");
    printf("  Total time: %.4f ms\n", time_regular);
    printf("  Average time per kernel: %.4f ms\n", avg_time_regular);
    printf("\n");
    printf("Coarsened matrix multiplication:\n");
    printf("  Total time: %.4f ms\n", time_coarsened);
    printf("  Average time per kernel: %.4f ms\n", avg_time_coarsened);
    printf("\n");
    printf("Speedup from coarsening: %.2fx\n", avg_time_regular / avg_time_coarsened);
    printf("\n");
    
    // verify outputs 
    tiledMatMul<<<gridSize_regular, blockSize>>>(d_A, d_B, d_C_regular, N);
    cudaDeviceSynchronize();
    tiledMatMulCoars<<<gridSize_coarsened, blockSize>>>(d_A, d_B, d_C_coarsened, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(C_regular, d_C_regular, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(C_coarsened, d_C_coarsened, size, cudaMemcpyDeviceToHost);
    
    bool results_match = true;
    float max_diff = 0.0f;
    int diff_count = 0;
    const float tolerance = 1e-5f;
    
    for (int i = 0; i < N * N; i++) {
        float diff = fabs(C_regular[i] - C_coarsened[i]);
        if (diff > tolerance) {
            results_match = false;
            diff_count++;
            if (diff > max_diff) {
                max_diff = diff;
            }
        }
    }
    
    if (results_match) {
        printf("Outputs from both kernels match\n");
    } else {
        printf("Mismatch between the outputs, something's wrong\n");
        printf("Number of different values: %d out of %d\n", diff_count, N * N);
        printf("Maximum difference: %f\n", max_diff);
    }
    
    cudaEventDestroy(start_regular);
    cudaEventDestroy(stop_regular);
    cudaEventDestroy(start_coarsened);
    cudaEventDestroy(stop_coarsened);
    
    free(A);
    free(B);
    free(C_regular);
    free(C_coarsened);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_regular);
    cudaFree(d_C_coarsened);
    return 0;
}
