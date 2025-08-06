#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

// Phase 1: compute exponentials and partial sums for each block
__global__ void softmax_phase1(float *input, float *exp_values, float *partial_sums, int size) {
    extern __shared__ float shared_mem[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    if (idx < size) {
        exp_values[idx] = expf(input[idx]);
        shared_mem[tid] = exp_values[idx];
    } else {
        shared_mem[tid] = 0.0f;
    }
    
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial_sums[blockIdx.x] = shared_mem[0];
    }
}

// Phase 2: compute final sum and normalize
__global__ void softmax_phase2(float *exp_values, float *partial_sums, float *output, int size, int num_blocks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ float total_sum;
    if (threadIdx.x == 0) {
        total_sum = 0.0f;
        for (int i = 0; i < num_blocks; i++) {
            total_sum += partial_sums[i];
        }
    }
    __syncthreads();
    
    if (idx < size) {
        output[idx] = exp_values[idx] / total_sum;
    }
}

int main() {
    const int size = 1024;
    const int bytes = size * sizeof(float);
    
    float *h_input = new float[size];
    float *h_output = new float[size];
    
    for (int i = 0; i < size; i++) {
        h_input[i] = (float)(rand() % 100) / 10.0f;
    }
    
    float *d_input, *d_output, *d_exp_values, *d_partial_sums;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMalloc(&d_exp_values, bytes);
    
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    size_t shared_mem_size = block_size * sizeof(float);
    
    // allocate memory for partial sums (one per block)
    cudaMalloc(&d_partial_sums, grid_size * sizeof(float));
    
    printf("Using %d blocks of %d threads each for %d elements\n", grid_size, block_size, size);
    
    // Phase 1: compute exponentials and partial sums
    softmax_phase1<<<grid_size, block_size, shared_mem_size>>>(d_input, d_exp_values, d_partial_sums, size);
    cudaDeviceSynchronize();
    
    // Phase 2: compute final sum and normalize
    softmax_phase2<<<grid_size, block_size>>>(d_exp_values, d_partial_sums, d_output, size, grid_size);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    
    printf("First 10 input values:  ");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", h_input[i]);
    }
    printf("\n");
    
    printf("First 10 output values: ");
    for (int i = 0; i < 10; i++) {
        printf("%.6f ", h_output[i]);
    }
    printf("\n");
    
    float sum_check = 0.0f;
    for (int i = 0; i < size; i++) {
        sum_check += h_output[i];
    }
    printf("Total sum: %.8f \n", sum_check);
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_exp_values);
    cudaFree(d_partial_sums);
    delete[] h_input;
    delete[] h_output;
    
    return 0;
}
