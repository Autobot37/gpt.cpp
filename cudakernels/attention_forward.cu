#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

void attention_forward(float* out, float* preatt, float* att, float* qkv, int B, int T, int C, int NH){
    int hs = C/NH;
    float scale = 1.0;
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
                for(int t2=0;t2<T;t2++){
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

__global__ void softmax_kernel(float* att, float* preatt, int B, int T, int C, int NH){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= B * T * NH){
        return;
    }
    int b = idx / (T * NH);
    int t = (idx / NH) % T;
    int h = idx % NH;

    float* preatt_p = preatt + b*NH*T*T + h*T*T + t*T;
    float* att_p = att + b*NH*T*T + h*T*T + t*T;
    
    float maxval = -INFINITY;
    for(int t2=0;t2<=t;t2++){
        maxval = fmaxf(maxval, preatt_p[t2]);
    }

    float expsum = 0.0f;
    
    for(int t2=0;t2<=t;t2++){
        float expv = expf(preatt_p[t2] - maxval);
        att_p[t2] = expv;
        expsum += expv;
    }
    float expinv = (expsum == 0.0f) ? 0.0f : 1.0f / expsum;

    for(int t2=0;t2<T;t2++){
        if(t2 <= t){
            att_p[t2] *= expinv;
        }
        else{
            att_p[t2] = 0.0f;
        }
    }
}

void softmax_gpu(float* preatt, float* att, int B, int T, int C, int NH){
    int numThreads = 1024;
    int numBlocks = (B * NH * T + numThreads - 1) / numThreads;
    softmax_kernel<<<numBlocks, numThreads>>>(att, preatt, B, T, C, NH);
}

__global__
void permute_kernel(float* q, float* k, float* v, float* inp, int B, int N, int NH, int d){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int inp_idx = \
            (b * N * 3 * NH * d)
            +   (n * 3 * NH * d)
            +       (0 * NH * d)
            +          (nh_ * d)
            +                d_;

        q[idx] = inp[inp_idx];
        k[idx] = inp[inp_idx + NH * d];
        v[idx] = inp[inp_idx + 2 * (NH * d)];
    }
}

__global__
void unpermute_kernel(float* inp, float* out, int B, int N, int NH, int d){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
        out[other_idx] = inp[idx];
    }
}

void attention_forward3(float* out, float* vaccum, float* qkvr, float* preatt, float* att,
                        float* inp, int B, int T, int C, int NH,
                        int block_size){

    int HS = C/NH;
    float* q, *k, *v;
    q = qkvr;
    k = qkvr + B * T * C;
    v = qkvr + B * T * C * 2;
    int total_threads = B * NH * T * HS;
    int num_blocks = (total_threads + block_size - 1) / block_size;
    permute_kernel<<<num_blocks, block_size>>>(q, k, v, inp, B, T, NH, HS);

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "permute_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return;
    }

    cublasHandle_t handle;
    cublasStatus_t cublasStatus = cublasCreate(&handle);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasStatus = cublasSgemmStridedBatched(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        T, T, HS,//transposed m n k
        &alpha,
        k, HS, T * HS,
        q, HS, T * HS,//columns
        &beta,
        preatt, T, T * T,
        B * NH
    );
    if(cublasStatus != CUBLAS_STATUS_SUCCESS){
        fprintf(stderr, "cublasSgemmStridedBatched failed\n");
        return;
    }

    int numThreads = 1024;
    int numBlocks = (B * NH * T + numThreads - 1) / numThreads;
    softmax_kernel<<<numBlocks, numThreads>>>(att, preatt, B, T, C, NH);

    cublasStatus = cublasSgemmStridedBatched(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        HS, T, T,
        &alpha,
        v, HS, T * HS,
        att, T, T * T,
        &beta,
        vaccum, HS, T * HS,
        B * NH
    );
    if(cublasStatus != CUBLAS_STATUS_SUCCESS){
        fprintf(stderr, "cublasSgemmStridedBatched failed\n");
        return;
    }

    num_blocks = (B * T * C + numThreads - 1) / numThreads;
    unpermute_kernel<<<num_blocks, numThreads>>>(vaccum, out, B, T, NH, HS);
}

//preatt -> 
//q = [B,T,NH,hs]
//k = [B,T,NH,hs]
//q = [B,NH,T,hs]
//k = [B,NH,T,hs]
//k.T = [B,NH,hs,T]
//preatt = q @ k.T = [B,NH,T,T]


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

void check(float* a, float * b, int n){
    for(int i=0;i<n;i++){
        if(fabs(a[i] - b[i]) > 1e-5){
            printf("Error at %d: %f %f\n", i, a[i], b[i]);
            return;
        }
    }
    printf("Correct\n");
}

int main(){

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device Name: %s\n", prop.name);
    
    int B = 4;
    int T = 1024;
    int C = 1600;
    int OC = 4800;
    int NH = 25;

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
    float *d_out, *d_preatt, *d_qkvr, *d_vaccum, *d_att, *d_inp;

    cudaMalloc(&d_out, B*T*C*sizeof(float));
    cudaMalloc(&d_preatt, B*NH*T*T*sizeof(float));
    cudaMalloc(&d_qkvr, B*T*3*C*sizeof(float));
    cudaMalloc(&d_vaccum, B*T*C*sizeof(float));
    cudaMalloc(&d_att, B*NH*T*T*sizeof(float));
    cudaMalloc(&d_inp, B*T*3*C*sizeof(float));

    cudaMemcpy(d_inp, qkv, B*T*3*C*sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 256;
    attention_forward3(d_out, d_vaccum, d_qkvr, d_preatt, d_att, d_inp, B, T, C, NH, block_size);

    float* out_preatt, *out_att, *out_out;
    out_preatt = (float*)malloc(B*NH*T*T*sizeof(float));
    out_att = (float*)malloc(B*NH*T*T*sizeof(float));
    out_out = (float*)malloc(B*T*C*sizeof(float));

    cudaMemcpy(out_preatt, d_preatt, B*NH*T*T*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_att, d_att, B*NH*T*T*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_out, d_out, B*T*C*sizeof(float), cudaMemcpyDeviceToHost);

    check(preatt, out_preatt, B*NH*T*T);
    check(att, out_att, B*NH*T*T);
    check(out, out_out, B*T*C);

    return 0;
}