cublasHandle_t handle;


__global__
void embed_kernel(float* x, float* wte, float* wpe, int token, int pos, int C){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < C){
        x[i] = wte[token*C + i] + wpe[pos*C + i];
    }
}

void embed_gpu(float* x, float* wte, float* wpe, int token, int pos, int C){
    int num_threads = 1024;
    int num_blocks = (C + num_threads - 1) / num_threads;
    embed_kernel<<<num_blocks, num_threads>>>(x, wte, wpe, token, pos, C);
}
//-----------------------------------------------------------------------------------------------
__global__
void layernorm_kernel(float* out, float* x, float* w, float* b, int C){
    int idx = threadIdx.x;
    float mean = 0;
    __shared__ float s_mean[1024];
    __shared__ float s_var[1024];
    s_mean[idx] = 0.0f;
    s_var[idx] = 0.0f;
    __syncthreads();

    for(int i = idx;i<C;i+=blockDim.x){
        s_mean[idx] += x[i];
    }
    __syncthreads();
    if(idx == 0){
        float m = 0;
        for(int i = 0;i<blockDim.x;i++){
            m += s_mean[i];
        }
        m /= C;
        s_mean[0] = m;
    }
    __syncthreads();
    mean = s_mean[0];

    for(int i = idx;i<C;i+=blockDim.x){
        float diff = x[i] - mean;
        s_var[idx] += diff * diff;
    }
    __syncthreads();
    if(idx == 0){
        float v = 0;
        for(int i = 0;i<blockDim.x;i++){
            v += s_var[i];
        }
        v /= C;
        s_var[0] = v;
    }
    __syncthreads();
    float var = s_var[0];
    float scale = 1.0 / sqrt(var + 1e-6);
    for(int i = idx;i<C;i+=blockDim.x){
        out[i] = (x[i] - mean) * scale * w[i] + b[i];
    }
}

void layernorm_gpu(float* out, float* x, float* w, float* b, int C){
    int numThreads = 1024;
    int block = 1;
    layernorm_kernel<<<block,numThreads>>>(out,x,w,b,C);
}
//-----------------------------------------------------------------------------------------------

__global__ void add_bias(float* out, float* b, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N){
        out[i] += b[i];
    }
}
void gemm(float* out, float* in, float* w, float* b, int N, int D) {

    float alpha = 1.0;
    int lda = D;
    int incx = 1;
    float beta = 0.0;
    int incy = 1;

    cublasStatus_t status =  cublasSgemv(handle, CUBLAS_OP_T, D, N, &alpha, w, lda, in, incx, &beta, out, incy);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cublasSgemv failed\n");
    }
    if(b != NULL)
    add_bias<<<(N + 1023) / 1024, 1024>>>(out, b, N);
}
//-----------------------------------------------------------------------------------------------
__global__ void residual_kernel(float* out, float* in, int C){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < C){
        out[i] += in[i];
    }
}

void residual_gpu(float* out, float* in, int C){
    int num_threads = 1024;
    int num_blocks = (C + num_threads - 1) / num_threads;
    residual_kernel<<<num_blocks, num_threads>>>(out, in, C);
}

//-----------------------------------------------------------------------------------------------
__global__ void gelu_kernel(float* x, int C){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < C){
        float u = x[i];
        x[i] = 0.5 * u * (1 + tanh(sqrt(2.0/M_PI) * (u + 0.044715 * u * u * u)));
    }
}

void gelu_gpu(float* x, int C){
    int num_threads = 1024;
    int num_blocks = (C + num_threads - 1) / num_threads;
    gelu_kernel<<<num_blocks, num_threads>>>(x, C);
}
//-----------------------------------------------------------------------------------------------
__global__ void softmax_kernel(float*x, int N){
    int idx = threadIdx.x;
    __shared__ float smax[1024];
    __shared__ float ssum[1024];
    smax[idx] = 0.0f;
    ssum[idx] = 0.0f;
    __syncthreads();

    for(int i=idx;i<N;i+=blockDim.x){
        smax[idx] = fmaxf(smax[idx], x[i]);
    }
    __syncthreads();
    if(idx == 0){
        float maxval = -INFINITY;
        for(int i = 0;i<blockDim.x;i++){
            maxval = fmaxf(maxval, smax[i]);
        }
        smax[0] = maxval;
    }
    __syncthreads();
    float maxval = smax[0];
    for(int i=idx;i<N;i+=blockDim.x){
        x[i] = exp(x[i] - maxval);
        ssum[idx] += x[i];
    }
    __syncthreads();
    if(idx == 0){
        float sum = 0;
        for(int i = 0;i<blockDim.x;i++){
            sum += ssum[i];
        }
        ssum[0] = sum;
    }
    __syncthreads();
    float sum = ssum[0];
    for(int i=idx;i<N;i+=blockDim.x){
        x[i] /= sum;
    }
}

void softmax_gpu(float*x, int N){
    int numThreads = 1024;
    softmax_kernel<<<1, numThreads>>>(x, N);
}

//-----------------------------------------------------------------------------------------------


__device__ void softmaxg(float* x, int N){
    float max = x[0];
    for(int i = 1;i<N;i++){
        if(x[i] > max){
            max = x[i];
        }
    }
    float sum = 0;
    for(int i = 0;i<N;i++){
        x[i] = exp(x[i] - max);
        sum += x[i];
    }
    for(int i = 0;i<N;i++){
        x[i] /= sum;
    }
}

__global__
void attention_kernel(float* out, float* att, float* qkv, float* key_cache, float* value_cache, int l, int pos, int C, int NH, int head_size, int T){

    int h = threadIdx.x + blockIdx.x * blockDim.x;
    if(h >= NH){
        return;
    }

    float* q = qkv;
    float scale = 1.0 / sqrt(head_size);

    float* k = key_cache + l * C * T;
    float* v = value_cache + l * C * T;


    float* qh = q + h * head_size;
    float* atth = att + h * T;

    for(int t = 0;t<=pos;t++){
        float* kh = k + t * C + h * head_size;
        float score = 0.0f;
        for(int i = 0;i<head_size;i++){
            score += qh[i] * kh[i];
        }
        score *= scale;
        atth[t] = score;
    }
    for(int t=pos+1;t<T;t++){
        atth[t] = -INFINITY;
    }

    softmaxg(atth, T);

    float* outh = out + h * head_size;
    for(int t = 0;t<=pos;t++){
        float* vh = v + t * C + h * head_size;
        float score = atth[t];
        for(int i = 0;i<head_size;i++){
            outh[i] += score * vh[i];
        }
    }
}

void attention_gpu(float* out, float* att, float* qkv, float* key_cache, float* value_cache, int l, int pos, int C, int NH, int head_size, int T){
    int numThreads = 1024;
    int blocks = (NH + numThreads - 1) / numThreads;
    cudaMemcpy(key_cache + l * C * T + pos * C, qkv + C, C * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(value_cache + l * C * T + pos * C, qkv + 2*C, C * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemset(out, 0, C * sizeof(float));
    attention_kernel<<<blocks, numThreads>>>(out, att, qkv, key_cache, value_cache, l, pos, C, NH, head_size, T);
}
//-----------------------------------------------------------------------------------------------