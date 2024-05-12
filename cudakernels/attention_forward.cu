#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <cuda_runtime.h>

void attention_forward(float* out, float* preatt, float* att, float* qkv, int B, int T, int C, int NH){
    int hs = C/NH;
    float scale = 1.0 / sqrtf(hs);
    #pragma omp parallel for collapse(3) schedule(dynamic)
    for(int b = 0;b<B;b++){
        for(int t = 0;t<T;t++){
            for(int h = 0;h<NH;h++){
                
                //q @ k 
                float* query = qkv + b * T * 3 * C + t * 3 * C + h * hs;
                float* preatt_p = preatt + b*NH*T*T + h*T*T + t*T;
                float* att_p = att + b*NH*T*T + h*T*T + t*T;

                float maxval = 1e-5f;
                #pragma omp simd reduction(max:maxval)
                for(int t2=0;t2<=t;t2++){
                    float* key = qkv + b * T * 3 * C + t2 * 3 * C + h*hs + C;
                    float val = 0.0f;
                    #pragma omp simd reduction(+:val)
                    for(int i = 0;i<hs;i++){
                        val += query[i] * key[i];
                    }
                    val *= scale;
                    if(val>maxval){
                        maxval = val;
                    }
                    preatt_p[t2] = val;
                }
                //softmax
                float sum = 0.0f;
                #pragma omp simd reduction(+:sum)
                for(int t2=0;t2<=t;t2++){
                    float val = expf(preatt_p[t2] - maxval);
                    att_p[t2] = val;
                    sum += val;
                }
                float expinv = (sum==0.0f) ? 0.0f : 1.0f/sum;
                #pragma omp simd
                for(int t2=0;t2<T;t2++){
                    if(t2<=t){
                        att_p[t2] *= expinv;
                    }
                    else{
                        att_p[t2] = 0.0f;
                    }
                }   
                //accumulating
                float* out_p = out + b*T*C + t*C + h*hs;
                #pragma omp simd
                for(int t2=0;t2<hs;t2++){
                    float val = 0.0f;
                    #pragma omp simd reduction(+:val)
                    for(int i = 0;i<T;i++){
                        float value = qkv[b*T*3*C + i*3*C + 2*C + h*hs + t2];
                        val += att_p[i] * value;
                    }
                    out_p[t2] = val;
                }
            }
        }
    }
} 
//ok first parellizing across only B and T

__global__ void attention_forward_kernel(float* out, float* preatt, float* att, float* qkv, int B, int T, int C, int NH){
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y * blockDim.y + threadIdx.y;
    if(b>=B || t>=T){
        return;
    }
    int hs = C/NH;
    float scale = 1.0 / sqrtf(hs);
    for(int h = 0;h<NH;h++){
        float* query = qkv + b * T * 3 * C + t * 3 * C + h * hs;
        float* preatt_p = preatt + b*NH*T*T + h*T*T + t*T;
        float* att_p = att + b*NH*T*T + h*T*T + t*T;

        float maxval = 1e-5f;
        #pragma omp simd reduction(max:maxval)
        for(int t2=0;t2<=t;t2++){
            float* key = qkv + b * T * 3 * C + t2 * 3 * C + h*hs + C;
            float val = 0.0f;
            #pragma omp simd reduction(+:val)
            for(int i = 0;i<hs;i++){
                val += query[i] * key[i];
            }
            val *= scale;
            if(val>maxval){
                maxval = val;
            }
            preatt_p[t2] = val;
        }
        //softmax
        float sum = 0.0f;
        #pragma omp simd reduction(+:sum)
        for(int t2=0;t2<=t;t2++){
            float val = expf(preatt_p[t2] - maxval);
            att_p[t2] = val;
            sum += val;
        }
        float expinv = (sum==0.0f) ? 0.0f : 1.0f/sum;
        #pragma omp simd
        for(int t2=0;t2<T;t2++){
            if(t2<=t){
                att_p[t2] *= expinv;
            }
            else{
                att_p[t2] = 0.0f;
            }
        }   
        //accumulating
        float* out_p = out + b*T*C + t*C + h*hs;
        #pragma omp simd
        for(int t2=0;t2<hs;t2++){
            float val = 0.0f;
            #pragma omp simd reduction(+:val)
            for(int i = 0;i<T;i++){
                float value = qkv[b*T*3*C + i*3*C + 2*C + h*hs + t2];
                val += att_p[i] * value;
            }
            out_p[t2] = val;
        }
    }
}
void attention_forward_gpu(float* out, float* preatt, float* att, float* qkv, int B, int T, int C, int NH){
    dim3 threads(32, 32);
    dim3 blocks((B+31)/32, (T+31)/32);
    attention_forward_kernel<<<blocks, threads>>>(out, preatt, att, qkv, B, T, C, NH);
    cudaDeviceSynchronize();
}

void rand_init(float* arr, int size){
    for(int i = 0;i<size;i++){
        arr[i] = (float)rand() / RAND_MAX;
    }
}

int main(){

    int mul = 4;
    int B = 4*mul;
    int T = 128*mul;
    int C = 128*mul;
    int OC = 128*2*mul;
    int NH = 8*mul;

    float *preatt, *att, *qkv, *out;
    preatt = (float*)malloc(B*NH*T*T*sizeof(float));
    att = (float*)malloc(B*NH*T*T*sizeof(float));
    qkv = (float*)malloc(B*T*3*C*sizeof(float));
    out = (float*)malloc(B*T*C*sizeof(float));
    rand_init(preatt, B*NH*T*T);
    rand_init(att, B*NH*T*T);
    rand_init(qkv, B*T*3*C);

    srand(time(NULL));
    clock_t start, end;
    double time_used;
    start = clock();
    attention_forward(out, preatt, att, qkv, B, T, C, NH);
    end = clock();
    time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Time: %f\n", time_used);

    //gpu
    float *d_preatt, *d_att, *d_qkv, *d_out;
    cudaMalloc(&d_preatt, B*NH*T*T*sizeof(float));
    cudaMalloc(&d_att, B*NH*T*T*sizeof(float));
    cudaMalloc(&d_qkv, B*T*3*C*sizeof(float));
    cudaMalloc(&d_out, B*T*C*sizeof(float));

    cudaMemcpy(d_preatt, preatt, B*NH*T*T*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_att, att, B*NH*T*T*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qkv, qkv, B*T*3*C*sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    start = clock();
    attention_forward_gpu(d_out, d_preatt, d_att, d_qkv, B, T, C, NH);
    end = clock();
    time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Time: %f\n", time_used);

    float* check_out;
    check_out = (float*)malloc(B*T*C*sizeof(float));
    cudaMemcpy(check_out, d_out, B*T*C*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for(int i=0;i<B*T*C;i++){
        if(abs(out[i] - check_out[i]) > 1e-3f){
            printf("Incorrect output Try Again!\n");
            return 0;
        }
    }
    printf("Correct output Yay!\n");

    return 0;
}