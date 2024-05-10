#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <omp.h>

void matmul_naive(float* out,float* a,float* b,int n,int m,int k){
    #pragma omp parallel for collapse(2)
    for(int i = 0;i<n;i++){
        for(int j = 0;j<k;j++){
            float sum = 0.0f;
            for(int col = 0;col<m;col++){
                sum += a[i * m + col] * b[col* k + j];
            }
            out[i * n + j] = sum;
        }
    }
}

__global__ void matmul_forward_kernel1(float* out,float* a,float* b,int n,int m,int k){

}

void matmul_forward1(float* out, float* a,float* b, int n,int m,int k){
    
}
int main(){ 
    int dim = 512;

    float* a = (float*)calloc(dim * dim, sizeof(float));
    float* b = (float*)calloc(dim * dim, sizeof(float));
    float* out = (float*)calloc(dim * dim, sizeof(float));
    
    clock_t start, end;
    double cpu_time_used;

    start = clock();
    
    matmul_naive(out,a,b,dim,dim,dim);

    end = clock();

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("this takes %.6lf seconds\n",cpu_time_used);
    printf("wet pants\n");

    free(a);
    free(b);
    free(out);

    return 0;
}