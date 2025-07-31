#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 256

__global__ void rowSumKernel(float* input, float* output, int rows, int cols) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int col = tid;
    
    if (row >= rows) return;
    
    sdata[tid] = 0.0f;
    
    // Handle case if row width is bigger than block size
    float sum = 0.0f;
    while (col < cols) {
        sum += input[row * cols + col];
        col += blockDim.x;
    }
    sdata[tid] = sum;
    
    __syncthreads();
    
    // reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[row] = sdata[0];
    }
}

int main() {
    const int rows = 8;
    const int cols = 1000;
    
    float* h_input = (float*)malloc(rows * cols * sizeof(float));
    float* h_output = (float*)malloc(rows * sizeof(float));
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            h_input[i * cols + j] = (float)rand() / (float)RAND_MAX;
        }
    }
    
    float *d_input, *d_output;
    
    size_t input_size = rows * cols * sizeof(float);
    size_t output_size = rows * sizeof(float);
    
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, output_size);
    
    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
    
    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize(rows);  // one block per row
    
    rowSumKernel<<<gridSize, blockSize>>>(d_input, d_output, rows, cols);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);
    
    printf("Final matrix (row sums, all rows) from GPU:\n");
    for (int i = 0; i < rows; i++) {
        printf("Row %d GPU sum: %f\n", i, h_output[i]);
    }

    printf("\nVerifying sums on CPU:\n");
    for (int i = 0; i < rows; i++) {
        float cpu_sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            cpu_sum += h_input[i * cols + j];
        }
        printf("Row %d CPU sum: %f\n", i, cpu_sum);
    }

    cudaFree(d_input);
    cudaFree(d_output);
    
    free(h_input);
    free(h_output);
    
    return 0;
}
