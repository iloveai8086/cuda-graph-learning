//
// Created by ros on 1/10/24.
//
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

__global__ void warmupKernel() {
    // 执行一些无关紧要的计算任务
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    for (int i = 0; i < 1000; ++i) {
        result += sinf(float(i)) / cosf(float(i));
    }
    // 避免编译器优化，确保计算不会被消除
    if (idx < 1) {
        printf("Warmup result: %f\n", result);
    }

}

void warmupGPU() {
    // 设置 GPU 设备
    cudaSetDevice(0);

    // 启动 "warmup" 函数
    warmupKernel<<<1, 1>>>();
    cudaDeviceSynchronize();

}

// 定义数组维度
const int ROWS = 10;
const int COLS = 128;
const int NUM_ARRAYS = 12;

int multi_cpy() {
    // 创建和初始化12个数组
    int arrays[NUM_ARRAYS][ROWS][COLS];

    // 在 GPU 上分配内存
    int *arrays_gpu[NUM_ARRAYS];
    for (int i = 0; i < NUM_ARRAYS; ++i) {
        cudaMalloc((void **) &arrays_gpu[i], sizeof(int) * ((i < 4) ? ROWS * COLS : ROWS));
    }

    // 将数组拷贝到 GPU，并统计耗时
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ARRAYS; ++i) {
        cudaMemcpy(arrays_gpu[i], arrays[i], sizeof(int) * ((i < 4) ? ROWS * COLS : ROWS), cudaMemcpyHostToDevice);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Multi data copy time: " << duration.count() << " milliseconds" << std::endl;


    // 释放 GPU 上的内存
    for (int i = 0; i < NUM_ARRAYS; ++i) {
        cudaFree(arrays_gpu[i]);
    }

    return 0;
}

int pinned_multi_cpy() {
    // 创建和初始化12个数组
    int *arrays[NUM_ARRAYS];

    // 在 GPU 上分配内存
    int *arrays_gpu[NUM_ARRAYS];
    for (int i = 0; i < NUM_ARRAYS; ++i) {
        cudaMalloc((void **) &arrays_gpu[i], sizeof(int) * ((i < 4) ? ROWS * COLS : ROWS));
        cudaHostAlloc((void **) &arrays[i], sizeof(int) * ((i < 4) ? ROWS * COLS : ROWS), cudaHostAllocDefault);
    }

    // 将数组拷贝到 GPU，并统计耗时
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ARRAYS; ++i) {
        cudaMemcpy(arrays_gpu[i], arrays[i], sizeof(int) * ((i < 4) ? ROWS * COLS : ROWS), cudaMemcpyHostToDevice);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Pinned Memory, Multi data copy time: " << duration.count() << " milliseconds" << std::endl;


    // 释放 GPU 上的内存
    for (int i = 0; i < NUM_ARRAYS; ++i) {
        cudaFree(arrays_gpu[i]);
    }

    return 0;
}

int SIZE = ROWS * COLS * 4 + ROWS * 8;

int single_cpy() {
    // 创建和初始化数组
    int array[SIZE];

    // 在 GPU 上分配内存
    int *array_gpu;
    cudaMalloc((void **) &array_gpu, sizeof(int) * SIZE);

    // 将数组拷贝到 GPU，并统计耗时
    auto start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(array_gpu, array, sizeof(int) * SIZE, cudaMemcpyHostToDevice);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Single data copy time: " << duration.count() << " milliseconds" << std::endl;

    // 释放 GPU 上的内存
    cudaFree(array_gpu);

    return 0;
}

int pinned_single_cpy() {
    // 创建和初始化数组
    int *array;

    // 在 GPU 上分配内存
    int *array_gpu;
    cudaMalloc((void **) &array_gpu, sizeof(int) * SIZE);
    cudaHostAlloc((void **) &array, sizeof(int) * SIZE, cudaHostAllocDefault);

    // 将数组拷贝到 GPU，并统计耗时
    auto start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(array_gpu, array, sizeof(int) * SIZE, cudaMemcpyHostToDevice);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Pinned Memory, Single data copy time: " << duration.count() << " milliseconds" << std::endl;

    // 释放 GPU 上的内存
    cudaFree(array_gpu);

    return 0;
}

int main(void) {
    warmupGPU();
    multi_cpy();
    pinned_multi_cpy();
    single_cpy();
    pinned_single_cpy();
    return 0;
}
