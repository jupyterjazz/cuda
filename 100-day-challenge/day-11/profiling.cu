#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_profiler_api.h>

__global__
void saxpy(int n, float a, float *x, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        y[i] = a * x[i] + y[i];
    }
}

int main(void)
{
    int N = 1 << 20;
    float a = 2.0f;
    float *x, *y, *d_x, *d_y;
    x = (float *)malloc(N * sizeof(float));
    y = (float *)malloc(N * sizeof(float));

    cudaMalloc((void **)&d_x, N * sizeof(float));
    cudaMalloc((void **)&d_y, N * sizeof(float));

    srand((unsigned int)time(NULL));
    for (int i = 0; i < N; i++) {
        x[i] = (float)rand() / RAND_MAX;
        y[i] = (float)rand() / RAND_MAX;
    }

    printf("First 5 elements of x and y before SAXPY:\n");
    for (int i = 0; i < 5 && i < N; i++) {
        printf("x[%d] = %f, y[%d] = %f\n", i, x[i], i, y[i]);
    }

    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // perform SAXPY
    saxpy<<<(N+255)/256, 256>>>(N, a, d_x, d_y);
    cudaDeviceSynchronize();

    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("First 5 elements of y after SAXPY:\n");
    for (int i = 0; i < 5 && i < N; i++){
        printf("y[%d] = %f\n", i, y[i]);
    }

    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);

    cudaProfilerStop();

    return 0;
}