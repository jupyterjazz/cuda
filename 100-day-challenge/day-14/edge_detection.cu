#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define MASK_DIM 3
#define MASK_RADIUS (MASK_DIM / 2)

__constant__ int sobel_x_c[MASK_DIM][MASK_DIM];
__constant__ int sobel_y_c[MASK_DIM][MASK_DIM];

__global__ void sobel_edge_detection(float *input, float *output, int input_width, int input_height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < input_height && col < input_width) {
        float gx = 0.0f, gy = 0.0f;
        
        for (int i = 0; i < MASK_DIM; i++) {
            for (int j = 0; j < MASK_DIM; j++) {
                int input_row = row + i - MASK_RADIUS;
                int input_col = col + j - MASK_RADIUS;
                
                if (input_row >= 0 && input_row < input_height && 
                    input_col >= 0 && input_col < input_width) {
                    float pixel = input[input_row * input_width + input_col];
                    gx += pixel * sobel_x_c[i][j];
                    gy += pixel * sobel_y_c[i][j];
                }
            }
        }
        output[row * input_width + col] = sqrtf(gx * gx + gy * gy);
    }
}


int main() {
    int width, height, channels;
    unsigned char *img = stbi_load("human.jpeg", &width, &height, &channels, 0);
    if (!img) {
        printf("Failed to load human.jpeg\n");
        return -1;
    }
    printf("Loaded image: %dx%d with %d channels\n", width, height, channels);

    const int bytes = width * height * sizeof(float);
    int h_sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int h_sobel_y[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    float *h_input = new float[width * height];
    float *h_output = new float[width * height];

    for (int i = 0; i < width * height; i++) {
        if (channels == 3) {
            h_input[i] = 0.299f * img[i*channels] + 0.587f * img[i*channels+1] + 0.114f * img[i*channels+2];
        } else {
            h_input[i] = img[i];
        }
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpyToSymbol(sobel_x_c, h_sobel_x, MASK_DIM * MASK_DIM * sizeof(int));
    cudaMemcpyToSymbol(sobel_y_c, h_sobel_y, MASK_DIM * MASK_DIM * sizeof(int));

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    int block_size = 16;
    dim3 block(block_size, block_size);
    dim3 grid((width + block_size - 1) / block_size, (height + block_size - 1) / block_size);

    sobel_edge_detection<<<grid, block>>>(d_input, d_output, width, height);

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    unsigned char *output_img = new unsigned char[width * height];
    for (int i = 0; i < width * height; i++) {
        output_img[i] = (unsigned char)fminf(255.0f, h_output[i]);
    }

    stbi_write_png("edges.png", width, height, 1, output_img, width);
    printf("Edge detection finished, image saved as edges.png\n");

    cudaFree(d_input);
    cudaFree(d_output);
    stbi_image_free(img);
    delete[] h_input;
    delete[] h_output;
    delete[] output_img;
}
