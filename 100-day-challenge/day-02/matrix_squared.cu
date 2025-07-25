#include <stdio.h>
#include <stdlib.h>

__global__
void square_matrix(int *matrix, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n){
        matrix[i * n + j] = matrix[i * n + j] * matrix[i * n + j];
    }
}

int main(void){
    int n = 1024;
    int *matrix = (int *)malloc(n * n * sizeof(int));
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            matrix[i * n + j] = rand() % 100;
        }
    }
    printf("Original matrix (showing first 16x16 elements):\n");
    for (int i = 0; i < 16; i++){
        for (int j = 0; j < 16; j++){
            printf("%d ", matrix[i * n + j]);
        }
        printf("\n");
    }
    int *d_matrix;
    cudaMalloc((void **)&d_matrix, n * n * sizeof(int));
    cudaMemcpy(d_matrix, matrix, n * n * sizeof(int), cudaMemcpyHostToDevice);
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    square_matrix<<<grid, block>>>(d_matrix, n);
    cudaMemcpy(matrix, d_matrix, n * n * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Squared matrix (showing first 16x16 elements):\n");
    for (int i = 0; i < 16; i++){
        for (int j = 0; j < 16; j++){
            printf("%d ", matrix[i * n + j]);
        }
        printf("\n");
    }
    cudaFree(d_matrix);
    free(matrix);
    return 0;
}