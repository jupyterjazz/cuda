#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCK_SIZE 256
#define EPSILON 1e-5f

__global__ void layer_norm_kernel(float* input, float* output, int batch_size, int seq_len, int hidden_dim) {
    int block_id = blockIdx.x;
    int tid = threadIdx.x;
    
    int batch_idx = block_id / seq_len;
    int seq_idx = block_id % seq_len;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len) return;
    
    int base_idx = batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim;
    
    extern __shared__ float sdata[];
    
    // compute mean
    float sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        sum += input[base_idx + i];
    }
    sdata[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    float mean = sdata[0] / hidden_dim;
    __syncthreads();
    
    // compute variance
    float var_sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float diff = input[base_idx + i] - mean;
        var_sum += diff * diff;
    }
    sdata[tid] = var_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    float variance = sdata[0] / hidden_dim;
    float std_dev = sqrtf(variance + EPSILON);
    __syncthreads();
    
    // normalization
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        output[base_idx + i] = (input[base_idx + i] - mean) / std_dev;
    }
}

int main() {
    const int batch_size = 8;
    const int seq_len = 16;
    const int hidden_dim = 512;
    const int tensor_size = batch_size * seq_len * hidden_dim;
    
    float* h_input = (float*)malloc(tensor_size * sizeof(float));
    float* h_output = (float*)malloc(tensor_size * sizeof(float));
    
    for (int i = 0; i < tensor_size; i++) {
        h_input[i] = (float)rand() / RAND_MAX;
    }
    
    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, tensor_size * sizeof(float));
    cudaMalloc(&d_output, tensor_size * sizeof(float));
    
    cudaMemcpy(d_input, h_input, tensor_size * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 grid(batch_size * seq_len); // one block per hidden_dim
    dim3 block(BLOCK_SIZE);
    size_t shared_mem = BLOCK_SIZE * sizeof(float);
    
    layer_norm_kernel<<<grid, block, shared_mem>>>(d_input, d_output, batch_size, seq_len, hidden_dim);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, tensor_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Shape: (%d, %d, %d)\n", batch_size, seq_len, hidden_dim);
    
    printf("Sample input (first 5 elements of [0,0]): ");
    for (int i = 0; i < 5; i++) {
        printf("%f ", h_input[i]);
    }
    printf("\n");
    
    printf("Sample output (first 5 elements of [0,0]): ");
    for (int i = 0; i < 5; i++) {
        printf("%f ", h_output[i]);
    }
    printf("\n");
    
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
