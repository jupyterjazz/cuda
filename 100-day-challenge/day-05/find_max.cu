
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ 
void find_max(int* arr, int* output) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int myVal = arr[idx];

    bool IAmMax = true;
    for (int i = 0; i < blockDim.x; ++i) {
        int otherVal = arr[blockIdx.x * blockDim.x + i];
        if (otherVal > myVal || (otherVal == myVal && i < tid)) {
            IAmMax = false;
            break;
        }
    }
    __syncthreads();
    if (IAmMax) {
        output[blockIdx.x] = myVal;
    }
}


int main(){
    int N = 1024;
    size_t size = N * sizeof(int);
    int *arr = (int*)malloc(size);
    int *output = (int*)malloc(sizeof(int));
    for (int i = 0; i < N; i++){
        arr[i] = rand() % 1000;
    }

    int *d_arr, *d_output;
    cudaMalloc(&d_arr, size);
    cudaMalloc(&d_output, sizeof(int));

    cudaMemcpy(d_arr, arr, size, cudaMemcpyHostToDevice);
    find_max<<<1, N>>>(d_arr, d_output); // one block
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Array elements:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    printf("Max element: %d\n", output[0]);
    free(arr);
    free(output);
    cudaFree(d_arr);
    cudaFree(d_output);
    return 0;
}