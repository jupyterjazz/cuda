#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLUR_SIZE 3

__global__
void blur_kernel(unsigned char *image, unsigned char *blurred, unsigned int width, unsigned int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height){
        unsigned int average = 0;
        for (int i = x - BLUR_SIZE; i <= x + BLUR_SIZE; i++){
            for (int j = y - BLUR_SIZE; j <= y + BLUR_SIZE; j++){
                if (i >= 0 && i < width && j >= 0 && j < height){
                    average += image[j * width + i];
                }
            }
        }
        blurred[y * width + x] = average / (2 * BLUR_SIZE + 1) / (2 * BLUR_SIZE + 1);
    }
}

void create_test_image(unsigned char *image, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // checkboard pattern
            if ((x / 32 + y / 32) % 2 == 0) {
                image[y * width + x] = 255;
            } else {
                image[y * width + x] = 0;
            }
        }
    }
    // Change some pixels randomly
    for (int i = 0; i < width * height / 10; i++) {
        int x = rand() % width;
        int y = rand() % height;
        image[y * width + x] = rand() % 256;
    }
}

int main() {
    unsigned int width = 512;
    unsigned int height = 512;
    size_t image_size = width * height * sizeof(unsigned char);

    unsigned char *host_image = (unsigned char*)malloc(image_size);
    unsigned char *host_blurred = (unsigned char*)malloc(image_size);
    
    create_test_image(host_image, width, height);
    
    unsigned char *device_image, *device_blurred;
    cudaMalloc(&device_image, image_size);
    cudaMalloc(&device_blurred, image_size);
    
    cudaMemcpy(device_image, host_image, image_size, cudaMemcpyHostToDevice);

    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, 
                   (height + block_size.y - 1) / block_size.y);
    
    printf("Launching kernel with grid: (%d, %d), block: (%d, %d)\n", 
           grid_size.x, grid_size.y, block_size.x, block_size.y);
    
    blur_kernel<<<grid_size, block_size>>>(device_image, device_blurred, width, height);
        
    cudaDeviceSynchronize();
    
    printf("Executed successfully\n");
    
    cudaMemcpy(host_blurred, device_blurred, image_size, cudaMemcpyDeviceToHost);

    printf("Original -> Blurred\n");
    for (int i = 0; i < 5; i++) {
        int x = 100 + i * 50;
        int y = 100;
        int idx = y * width + x;
        printf("Pixel (%d,%d): %3d -> %3d\n", x, y, host_image[idx], host_blurred[idx]);
    }
        
    cudaFree(device_image);
    cudaFree(device_blurred);
    free(host_image);
    free(host_blurred);
    
    return 0;
}

