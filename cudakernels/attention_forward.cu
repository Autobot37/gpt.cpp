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
__global__ void preatt_kernel(float* preatt,float* qkv, int B, int T, int C, int NH){
    int bt = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int t2 = blockIdx.z * blockDim.z + threadIdx.z;
    int b = bt / T;
    int t = bt % T;
    if(b>=B || t>=T || h>=NH || t2>=T){
        return;
    }
    int hs = C/NH;
    float scale = 1.0 / sqrtf(hs);
    float* query = qkv + b * T * 3 * C + t * 3 * C + h * hs;
    float* preatt_p = preatt + b*NH*T*T + h*T*T + t*T;
    float* key = qkv + b * T * 3 * C + t2 * 3 * C + h*hs + C;

    if(t2>t){
        preatt_p[t2] = 0.0f;
    }
    //key @ query 
    float val = 0.0f;
    for(int i = 0;i<hs;i++){
        val += query[i] * key[i];
    }
    val *= scale;
    preatt_p[t2] = val;
}

void preatt_gpu(float* preatt, float* qkv, int B, int T, int C, int NH){
    dim3 threads(8,8,8);
    dim3 blocks;
    blocks.x = (B*T+7)/8;
    blocks.y = (NH+7)/8;
    blocks.z = (T+7)/8;
    preatt_kernel<<<blocks, threads>>>(preatt, qkv, B, T, C, NH);
}

__global__ void softmax_kernel(float* att, float* preatt, int B, int T, int C, int NH){
    int bt = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int t2 = blockIdx.z * blockDim.z + threadIdx.z;
    int b = bt / T;
    int t = bt % T;
    if(b>=B || t>=T || h>=NH || t2>=T){
        return;
    }
    float* preatt_p = preatt + b*NH*T*T + h*T*T + t*T;
    float* att_p = att + b*NH*T*T + h*T*T + t*T;
    if(t2>t){
        att_p[t2] = 0.0f;
    }

    float maxval = 0.0f;
    float sum = 0.0f;
    
    __shared__ float sdata_max[1024];
    __shared__ float sdata_sum[1024];
    
    sdata_max[threadIdx.x] = preatt_p[t2];
    __syncthreads();

    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (threadIdx.x < s) {
            sdata_max[threadIdx.x] = fmaxf(sdata_max[threadIdx.x], sdata_max[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        maxval = sdata_max[0];
    }
    __syncthreads();
    
    sdata_sum[threadIdx.x] = att_p[t2];
    __syncthreads();

    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (threadIdx.x < s) {
            sdata_sum[threadIdx.x] += sdata_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        sum = sdata_sum[0];
    }
    __syncthreads();

    float expinv = (sum == 0.0f) ? 0.0f : 1.0f / sum;
    att_p[t2] *= expinv;
}

void softmax_gpu(float* preatt, float* att, int B, int T, int C, int NH){
    dim3 threads(8,8,8);
    dim3 blocks;
    blocks.x = (B*T+7)/8;
    blocks.y = (NH+7)/8;
    blocks.z = (T+7)/8;
    softmax_kernel<<<blocks, threads>>>(att, preatt, B, T, C, NH);
}

__global__ void accumulate_kernel(float* out, float* qkv, float* att, int B, int T, int C, int NH) {
    int bt = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int t = blockIdx.z * blockDim.z + threadIdx.z;
    int b = bt / (T);
    int t2 = bt % T;
    if(b >= B || t2 >= T || h >= NH || t >= T) {
        return;
    }
    float* out_p = out + b * T * C + t * C + h * (C / NH);
    float val = 0.0f;
    #pragma unroll
    for(int i = 0; i < T; i++) {
        float value = qkv[b * T * 3 * C + i * 3 * C + 2 * C + h * (C / NH) + t2];
        val += att[b * NH * T * T + h * T * T + t * T + i] * value;
    }
    out_p[t2] = val;
}

void accumulate_gpu(float* out, float* qkv, float* att, int B, int T, int C, int NH){
    dim3 threads(8, 8, 8);
    dim3 blocks;
    blocks.x = (B * T + 7) / 8;
    blocks.y = (NH + 7) / 8;
    blocks.z = (T + 7) / 8;
    accumulate_kernel<<<blocks, threads>>>(out, qkv, att, B, T, C, NH);
}

__global__ void attention_forward_kernel(float* out, float* preatt, float* att, float* qkv, int B, int T, int C, int NH){
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y * blockDim.y + threadIdx.y;
    int h = blockIdx.z * blockDim.z + threadIdx.z;
    if(b>=B || t>=T || h>=NH){
        return;
    }
    int hs = C/NH;
    float scale = 1.0 / sqrtf(hs);

    float* query = qkv + b * T * 3 * C + t * 3 * C + h * hs;
    float* preatt_p = preatt + b*NH*T*T + h*T*T + t*T;
    float* att_p = att + b*NH*T*T + h*T*T + t*T;
    //key @ query 
    float maxval = 1e-5f;
    for(int t2=0;t2<=t;t2++){
        float* key = qkv + b * T * 3 * C + t2 * 3 * C + h*hs + C;
        float val = 0.0f;
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
    for(int t2=0;t2<=t;t2++){
        float val = expf(preatt_p[t2] - maxval);
        att_p[t2] = val;
        sum += val;
    }
    float expinv = (sum==0.0f) ? 0.0f : 1.0f/sum;
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
    for(int t2=0;t2<hs;t2++){
        float val = 0.0f;
        for(int i = 0;i<T;i++){
            float value = qkv[b*T*3*C + i*3*C + 2*C + h*hs + t2];
            val += att_p[i] * value;
        }
        out_p[t2] = val;
    }
}
void attention_forward_gpu(float* out, float* preatt, float* att, float* qkv, int B, int T, int C, int NH){
    dim3 threads(8,8,8);
    dim3 blocks((B+7)/8, (T+7)/8, (NH+7)/8);
    attention_forward_kernel<<<blocks, threads>>>(out, preatt, att, qkv, B, T, C, NH);
}
void attention_forward_gpu2(float* out, float* preatt, float* att, float* qkv, int B, int T, int C, int NH){
    dim3 threads_preatt(8,8,8);
    dim3 blocks_preatt;
    blocks_preatt.x = (B*T+7)/8;
    blocks_preatt.y = (NH+7)/8;
    blocks_preatt.z = (T+7)/8;
    preatt_kernel<<<blocks_preatt, threads_preatt>>>(preatt, qkv, B, T, C, NH);

    dim3 threads_softmax(8,8,8);
    dim3 blocks_softmax;
    blocks_softmax.x = (B*T+7)/8;
    blocks_softmax.y = (NH+7)/8;
    blocks_softmax.z = (T+7)/8;
    softmax_kernel<<<blocks_softmax, threads_softmax>>>(att, preatt, B, T, C, NH);

    dim3 threads_accumulate(8, 8, 8);
    dim3 blocks_accumulate;
    blocks_accumulate.x = (B * T + 7) / 8;
    blocks_accumulate.y = (NH + 7) / 8;
    blocks_accumulate.z = (T + 7) / 8;
    accumulate_kernel<<<blocks_accumulate, threads_accumulate>>>(out, qkv, att, B, T, C, NH);
}

void rand_init(float* arr, int size){
    for(int i = 0;i<size;i++){
        arr[i] = (float)rand() / RAND_MAX;
    }
}
void print_2d(float* arr, int r, int c){
    for(int i=0;i<r;i++){
        for(int j=0;j<c;j++){
            printf("%f ", arr[i*c+j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(){

    int mul = 4;
    int B = 1*mul;
    int T = 128*mul;
    int C = 128*mul;
    int OC = 128*mul;
    int NH = 4*mul;

    float *preatt, *att, *qkv, *out;
    preatt = (float*)malloc(B*NH*T*T*sizeof(float));
    att = (float*)malloc(B*NH*T*T*sizeof(float));
    qkv = (float*)malloc(B*T*3*C*sizeof(float));
    out = (float*)malloc(B*T*C*sizeof(float));
    rand_init(preatt, B*NH*T*T);
    rand_init(att, B*NH*T*T);
    rand_init(qkv, B*T*3*C);

    attention_forward(out, preatt, att, qkv, B, T, C, NH);
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

    attention_forward_gpu(d_out, d_preatt, d_att, d_qkv, B, T, C, NH);
    
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

    //preatt kernel
    float* another_d_preatt;
    cudaMalloc(&another_d_preatt, B*NH*T*T*sizeof(float));

    preatt_gpu(another_d_preatt, d_qkv, B, T, C, NH);
    cudaDeviceSynchronize();

    float* preatt_check1;
    preatt_check1 = (float*)malloc(B*NH*T*T*sizeof(float));
    cudaMemcpy(preatt_check1, d_preatt, B*NH*T*T*sizeof(float), cudaMemcpyDeviceToHost);
    float* preatt_check2;
    preatt_check2 = (float*)malloc(B*NH*T*T*sizeof(float));
    cudaMemcpy(preatt_check2, another_d_preatt, B*NH*T*T*sizeof(float), cudaMemcpyDeviceToHost);

    for(int b=0;b<B;b++){
        for(int h=0;h<NH;h++){
            for(int t=0;t<T;t++){
                for(int t2=0;t2<=t;t2++){
                    if(abs(preatt_check1[b*NH*T*T + h*T*T + t*T + t2] - preatt_check2[b*NH*T*T + h*T*T + t*T + t2]) > 1e-3f){
                        printf("Incorrect preatt output Try Again!\n");
                        return 0;
                    }
                }
            }
        }
    }
    printf("Correct preatt output Yay!\n");

    //softmax
    //
    float* softmax_att;
    cudaMalloc(&softmax_att, B*NH*T*T*sizeof(float));
    softmax_gpu(softmax_att, another_d_preatt, B, T, C, NH);

    float* softmax_check;
    softmax_check = (float*)malloc(B*NH*T*T*sizeof(float));
    cudaMemcpy(softmax_check, d_att, B*NH*T*T*sizeof(float), cudaMemcpyDeviceToHost);
    for(int i=0;i<B*NH*T*T;i++){
        if(abs(softmax_check[i] - att[i]) > 1e-3f){
            printf("Incorrect softmax output Try Again!\n");
            return 0;
        }
    }
    printf("Correct softmax output Yay!\n");


    //accumulate
    float* accumulate_out;
    cudaMalloc(&accumulate_out, B*T*C*sizeof(float));
    accumulate_gpu(accumulate_out, d_qkv, softmax_att, B, T, C, NH);

    float* accumulate_check;
    accumulate_check = (float*)malloc(B*T*C*sizeof(float));
    cudaMemcpy(accumulate_check, d_out, B*T*C*sizeof(float), cudaMemcpyDeviceToHost);
    int i = 0;
    for(int i=0;i<B*T*C;i++){
        if(abs(accumulate_check[i] - out[i]) > 1e-3f){
            i+=1;
        }
    }
    printf("Incorrect output at %d times out of %d times\n",i,B*T*C);
    printf("Correct accumulate output Yay!\n");

    //attention forward
    float* final_out;
    cudaMalloc(&final_out, B*T*C*sizeof(float));
    
    attention_forward_gpu2(final_out, d_preatt, d_att, d_qkv, B, T, C, NH);

    float* final_check;
    final_check = (float*)malloc(B*T*C*sizeof(float));
    cudaMemcpy(final_check, d_out, B*T*C*sizeof(float), cudaMemcpyDeviceToHost);
    int ii = 0;
    for(int i=0;i<B*T*C;i++){
        if(abs(accumulate_check[i] - out[i]) > 1e-3f){
            ii+=1;
        }
    }
    printf("Incorrect output at %d times out of %d times\n",ii,B*T*C);
    printf("Correct accumulate output Yay!\n");
    printf("Correct output Yay!\n");

    return 0;
}