#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MASK_DIM 7
#define MASK_RADIUS (MASK_DIM / 2)
#define SIGMA 1.5
__constant__ float mask_c[MASK_DIM][MASK_DIM];

__global__ void convolution(float *input, float *output, int input_width, int input_height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < input_height && col < input_width) { 
        float sum = 0.0f;
        
        for (int i = 0; i < MASK_DIM; i++) {
            for (int j = 0; j < MASK_DIM; j++) {
                int input_row = row + i - MASK_RADIUS;
                int input_col = col + j - MASK_RADIUS;
                
                if (input_row >= 0 && input_row < input_height && 
                    input_col >= 0 && input_col < input_width) {
                    sum += input[input_row * input_width + input_col] * mask_c[i][j];
                }
            }
        }
        output[row * input_width + col] = sum;
    }
}

void initialize_convolution_mask(float h_mask[MASK_DIM][MASK_DIM]) {
    float sum = 0.0f;
    int r = MASK_RADIUS;
    float s = 2.0f * SIGMA * SIGMA;
    
    // Gaussian blur kernel
    for (int x = -r; x <= r; x++) {
        for (int y = -r; y <= r; y++) {
            float val = expf(-(x*x + y*y) / s) / (M_PI * s);
            h_mask[x + r][y + r] = val;
            sum += val;
        }
    }
    // normalize
    for (int i = 0; i < MASK_DIM; i++) {
        for (int j = 0; j < MASK_DIM; j++) {
            h_mask[i][j] /= sum;
        }
    }
}

void cpu_convolution(float *input, float *output, float mask[MASK_DIM][MASK_DIM], int width, int height) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            float sum = 0.0f;
            for (int i = 0; i < MASK_DIM; i++) {
                for (int j = 0; j < MASK_DIM; j++) {
                    int input_row = row + i - MASK_RADIUS;
                    int input_col = col + j - MASK_RADIUS;
                    if (input_row >= 0 && input_row < height && 
                        input_col >= 0 && input_col < width) {
                        sum += input[input_row * width + input_col] * mask[i][j];
                    }
                }
            }
            output[row * width + col] = sum;
        }
    }
}

int main() {
    const int width = 2048;
    const int height = 4096;
    const int bytes = width * height * sizeof(float);

    float h_mask[MASK_DIM][MASK_DIM];
    initialize_convolution_mask(h_mask);

    float *h_input = new float[width * height];
    float *h_output = new float[width * height];

    for (int i = 0; i < width * height; i++) {
        h_input[i] = (float)(rand() % 100) / 10.0f;
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    // copy mask to constant memory
    cudaMemcpyToSymbol(mask_c, h_mask, MASK_DIM * MASK_DIM * sizeof(float));

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    int block_size = 16;
    dim3 block(block_size, block_size);
    dim3 grid((width + block_size - 1) / block_size, (height + block_size - 1) / block_size);

    convolution<<<grid, block>>>(d_input, d_output, width, height);

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    // verify with CPU
    float *h_cpu_output = new float[width * height];
    cpu_convolution(h_input, h_cpu_output, h_mask, width, height);
    
    float max_diff = 0.0f;
    for (int i = 0; i < width * height; i++) {
        float diff = fabs(h_output[i] - h_cpu_output[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("Max difference: %f\n", max_diff);

    cudaFree(d_input);
    cudaFree(d_output);

    delete[] h_input;
    delete[] h_output;
    delete[] h_cpu_output;
}
