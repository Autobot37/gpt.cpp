#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void encoder_forward_kernel(float* out, int* inp, float* wte, float* wpe, int B, int T, int C){
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y * blockDim.y + threadIdx.y;

    if (b < B && t < T) {
        float* out_p = out + b * T * C + t * C;
        float* wte_p = wte + inp[b * T + t] * C;
        float* wpe_p = wpe + t * C;
        for(int i = 0; i < C; i++) {
            out_p[i] = wte_p[i] + wpe_p[i];
        }
    }
}

void encoder_forward(float* out, int* inp, float* wte, float* wpe, int B, int T, int C){
    dim3 blockDim(32, 32); 
    dim3 gridDim((B + blockDim.x - 1) / blockDim.x, (T + blockDim.y - 1) / blockDim.y); // Adjust grid size
    encoder_forward_kernel<<<gridDim, blockDim>>>(out, inp, wte, wpe, B, T, C);
    cudaDeviceSynchronize();
    
}
void encoder_backward(float* d_wpe, float* d_wte, int* inp, float* d_out, int B, int T, int C){
    for(int b=0;b<B;b++){
        for(int t=0;t<T;t++){
            float* d_wte_p = d_wte + inp[b * T + t] * C;
            float* d_wpe_p = d_wpe + t * C;
            for(int i = 0;i<C;i++){
                d_wte_p[i] += d_out[i];
                d_wpe_p[i] += d_out[i];
            }
        }
    }
}

//--------
__global__ void layernorm_forward_kernel(float* out, float* mean, float* std_dev, float* inp, float* weight, float* bias, int B, int T, int C){
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y * blockDim.y + threadIdx.y;
    if(b>=B || t>=T){
        return;
    }
    float eps = 1e-5f;
    float* inp_p = inp + b*T*C + t*C;
    float* out_p = out + b*T*C + t*C;
                
    float m = 0.0f;
    for(int i=0;i<C;i++){
        m += inp_p[i];
    }
    m = m/C;
    mean[b*T + t] = m;

    float v = 0.0f;
    for(int i = 0;i<C;i++){
        float diff = inp_p[i] - m;
        v += diff * diff;
    }
    v = v / C;
    float s = 1.0f/sqrtf(v + eps);
    std_dev[b*T + t] = s;

    for(int i = 0;i<C;i++){
        out_p[i] = ((inp_p[i] - m) * s) * weight[i] + bias[i];            
    }
}

void layernorm_forward(float* out, float* mean, float* std_dev, float* inp, float* weight, float* bias, int B, int T, int C){
    dim3 threadsPerBlock(4, 256);
    dim3 numBlocks((B + threadsPerBlock.x - 1)/threadsPerBlock.x, (T + threadsPerBlock.y - 1)/threadsPerBlock.y);
    layernorm_forward_kernel<<<numBlocks, threadsPerBlock>>>(out, mean, std_dev, inp, weight, bias, B, T, C);
    cudaDeviceSynchronize();
}
//------------------------------------------
__global__ void matmul_gpu_kernel(float* out, float* inp, float* weight, float* bias, int B, int T, int C, int OC){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int bt = bx * blockDim.x + tx;
    int oc = by * blockDim.y + ty;

    if(bt<B*T && oc<OC){
        float* inp_p = inp + bt * C;
        float* out_p = out + bt * OC;
        float* weight_p = weight + oc * C;
        float val = (bias != NULL) ? bias[oc] : 0.0f;
        for(int i = 0;i<C;i++){
            val += inp_p[i] * weight_p[i];
        }
        out_p[oc] = val;
    }
    
}
void matmul_forward(float* out, float* inp, float* weight, float* bias, int B, int T, int C, int OC){
    dim3 block_size(32, 32);
    dim3 grid_size;
    grid_size.x = (B*T + block_size.x - 1) / block_size.x;
    grid_size.y = (OC + block_size.y - 1) / block_size.y;
    matmul_gpu_kernel<<<grid_size, block_size>>>(out, inp, weight, bias, B, T, C, OC);
}

//-------------------------------------------
__global__ void residual_forward_kernel(float* out, float* inp, float* skip, int dim){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<dim){
        out[i] = inp[i] + skip[i];
    }
}

void residual_forward(float* out, float* inp, float* skip, int dim){
    dim3 threads(256);
    dim3 blocks((dim + 255)/256);
    residual_forward_kernel<<<blocks, threads>>>(out, inp, skip, dim);
}
//-------------------------------------------

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
void attention_forward(float* out, float* preatt, float* att, float* qkv, int B, int T, int C, int NH){
    dim3 threads(32, 32);
    dim3 blocks((B+31)/32, (T+31)/32);
    attention_forward_kernel<<<blocks, threads>>>(out, preatt, att, qkv, B, T, C, NH);
    cudaDeviceSynchronize();
}

//---------------------------------------------
__global__ void gelu_forward_kernel(float* out, float* inp, int dim){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<dim){
        float x = inp[i];
        float cdf = 0.5f * (1.0f + tanhf((sqrtf(2.0f/M_PI) * (x + 0.044715f * x * x * x)) / (1.0f + 0.044715f * x * x)));
        out[i] = x * cdf;
    }
}

void gelu_forward(float* out, float* inp, int dim){
    dim3 threads(256);
    dim3 blocks((dim + 255)/256);
    gelu_forward_kernel<<<blocks, threads>>>(out, inp, dim);
}

//-------------------------------------------
__global__ void softmax_forward_kernel(float* out, float* inp, int B, int T, int V){
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y * blockDim.y + threadIdx.y;
    if(b>=B || t>=T){
        return;
    }
    float* inp_p = inp + b * T * V + t * V;
    float* out_p = out + b * T * V + t * V;
    float max_val = -1e-5f;
    for(int i=0;i<V;i++){
        if(inp_p[i] > max_val){
            max_val = inp_p[i];
        }
    }
    float sum = 0.0f;
    for(int i=0;i<V;i++){
        out_p[i] = expf(inp_p[i] - max_val);
        sum += out_p[i];
    }
    for(int i=0;i<V;i++){
        out_p[i] = out_p[i] / sum;
    }
}

void softmax_forward(float* out, float* inp, int B, int T, int V){
    dim3 threads(4,256);
    dim3 blocks((B + threads.x-1)/threads.x, (T + threads.y-1)/threads.y);
    softmax_forward_kernel<<<blocks, threads>>>(out, inp, B, T, V);
}
//-------------------------------------------
void crossentropy_forward(float* losses, float* probs, int * target, int B, int T, int V){
    for(int b = 0;b<B;b++){
        for(int t = 0;t<T;t++){
            float* probs_p = probs + b * T * V+ t * V;
            int ix = target[b * T + t];
            losses[b * T + t] = -logf(probs_p[ix]);
        }
    }
}
//-------------------------------------------
void print_3d(float *arr, int dim1, int dim2, int dim3) {
    printf("[\n");
    for (int i = 0; i < dim1; i++) {
        printf("  [\n");
        for (int j = 0; j < dim2; j++) {
            printf("    [ ");
            for (int k = 0; k < dim3; k++) {
                printf("%.2f", *(arr + i * dim2 * dim3 + j * dim3 + k));
                if (k < dim3 - 1) {
                    printf(", ");
                }
            }
            printf(" ]\n");
        }
        printf("  ]\n");
    }
    printf("]\n");
}

void print_2d_int(int *arr, int dim1, int dim2) {
    printf("[\n");
    for (int i = 0; i < dim1; i++) {
        printf("  [ ");
        for (int j = 0; j < dim2; j++) {
            printf("%.2d", *(arr + i * dim2 + j));
            if (j < dim2 - 1) {
                printf(", ");
            }
        }
        printf(" ]\n");
    }
    printf("]\n");
}

void print_2d(float *arr, int dim1, int dim2) {
    printf("[\n");
    for (int i = 0; i < dim1; i++) {
        printf("  [ ");
        for (int j = 0; j < dim2; j++) {
            printf("%.2f", *(arr + i * dim2 + j));
            if (j < dim2 - 1) {
                printf(", ");
            }
        }
        printf(" ]\n");
    }
    printf("]\n");
}