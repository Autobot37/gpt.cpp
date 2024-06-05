#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include "mkl.h"

void fill_matrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main() {
    srand(static_cast<unsigned int>(time(nullptr)));

    const int size = 1024;
    const int batch_size = 1;
    const float one = 1.0f;
    const float zero = 0.0f;

    float* A = new float[batch_size * size * size];
    float* B = new float[batch_size * size * size];
    float* C = new float[batch_size * size * size];

    for (int i = 0; i < batch_size; ++i) {
        fill_matrix(A + i * size * size, size, size);
        fill_matrix(B + i * size * size, size, size);
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    cblas_sgemm_batch_strided(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                size, size, size,
                                one, A, size, size * size,
                                B, size, size * size,
                                zero, C, size, size * size,
                                batch_size);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

    std::cout << "Matrix size: " << size << "x" << size << ", Batch size: " << batch_size << ", Time taken: " << duration / 1000.0f << " ms" << std::endl;

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}