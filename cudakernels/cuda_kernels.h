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
    if(b != NULL) {
        add_bias<<<(N + 1023) / 1024, 1024>>>(out, b, N);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("add_bias kernel launch failed: %s\n", cudaGetErrorString(err));
        }
    }

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

__global__
void fill_cache(float* key_cache, float* value_cache, float* arr, int l, int pos, int C, int NH, int T){
    //arr to cache
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < C){
        int head_size = C / NH;
        int nh = idx / head_size;
        int h = idx % head_size;
        float* key_cache_l = key_cache + l * T * NH * head_size;
        float* value_cache_l = value_cache + l * T * NH * head_size;
        //from-> NH HS 
        //to -> NH T HS
        int from = nh * head_size + h;
        int to = nh * T * head_size + pos * head_size + h;
        key_cache_l[to] = arr[from + C];
        value_cache_l[to] = arr[from + 2*C];
    }
}

__global__ 
void unfill_cache(float* cache, float* arr, int l, int pos, int C, int NH, int T){
    //cache to arr
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < C){
        int head_size = C / NH;
        int nh = idx / head_size;
        int h = idx % head_size;
        float* cache_l = cache + l * T * NH * head_size;
        //from-> NH T HS 
        //to -> NH HS
        int to = nh * head_size + h;
        int from = nh * T * head_size + pos * head_size + h;
        arr[to] = cache_l[from];
    }
}
__global__ 
void softmax_kernel2(float* x, int T, int NH, int pos, float scale){
    //requires launch <<<NH, 1024>>>
    int row = blockIdx.x;
    int idx = threadIdx.x;
    __shared__ float row_max[1024];
    __shared__ float row_sum[1024];
    row_max[threadIdx.x] = -FLT_MAX;
    row_sum[threadIdx.x] = 0;
    __syncthreads();
    

    for(int i = idx;i<pos;i+=blockDim.x){
        x[row * T + i] *= scale;
        float val = x[row * T + i];
        row_max[idx] = fmaxf(row_max[idx], val);
    }
    __syncthreads();
    if(idx==0){
        float maxval = -FLT_MAX;
        for(int i = 0;i<min(pos, blockDim.x);i++){
            maxval = fmaxf(maxval, row_max[i]);
        }
        row_max[0] = maxval;
    }
    __syncthreads();
    float maxval = row_max[0];
    
    for(int i = idx;i<pos;i+=blockDim.x){
        x[row * T + i] = expf(x[row * T + i] - maxval);
        row_sum[idx] += x[row * T + i];
    }
    __syncthreads();
    if(idx==0){
        float sum = 0;
        for(int i = 0;i<min(pos,blockDim.x);i++){
            sum += row_sum[i];
        }
        row_sum[0] = sum;
    }
    __syncthreads();
    float sum = row_sum[0];
    for(int i = idx;i<T;i+=blockDim.x){
        if(i<pos){
            x[row * T + i] /= sum;
        }
        else{
            x[row * T + i] = 0;
        }
    }
}

__device__ float warpReduceMax(float val){
    for(int offset = 16; offset > 0; offset /= 2){
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}
__device__ float warpReduceSum(float val){
    for(int offset = 16; offset > 0; offset /= 2){
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__
void softmax_kernel3(float* x, int T, int NH, int pos, float scale){
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = tid / 32;
    int laneId = tid % 32;

    int warpsPerBlock = blockDim.x / 32;

    extern __shared__ float shared[];
    float* maxvals = shared;
    float* sumvals = &shared[warpsPerBlock];

    float* x_h = x + idx * T;
    float maxval = -FLT_MAX;
    for(int i= tid;i<pos;i+=blockDim.x){
        x_h[i] *= scale;
        maxval = fmaxf(maxval, x_h[i]);
    }
    maxval = warpReduceMax(maxval);
    if(laneId == 0){
        maxvals[warpId] = maxval;
    }
    __syncthreads();
    if(tid==0){
        float val = -FLT_MAX;
        for(int i = 0;i<warpsPerBlock;i++){
            val = fmaxf(val, maxvals[i]);
        }
        maxvals[0] = val;
    }
    __syncthreads();
    float offset = maxvals[0];

    for(int i = tid;i<pos;i+=blockDim.x){
        x_h[i] = expf(x_h[i] - offset);
    }
    //sum
    float sumval = 0.0f;
    for(int i = tid;i<pos;i+=blockDim.x){
        sumval += x_h[i];
    }
    sumval = warpReduceSum(sumval);
    if(laneId == 0){
        sumvals[warpId] = sumval;
    }
    __syncthreads();
    if(tid==0){
        float val = 0;
        for(int i = 0;i<warpsPerBlock;i++){
            val += sumvals[i];
        }
        sumvals[0] = val;
    }
    __syncthreads();
    float sum = sumvals[0];
    for(int i = tid;i<T;i+=blockDim.x){
        x_h[i] = (i<pos) ? x_h[i] / sum : 0;
    }
}

void attention_blas(float* out, float* att, float* qkv, float* key_cache, float* value_cache, int l, int pos, int C, int NH, int T){

    int head_size = C / NH;
    int numThreads = 1024;
    int blocks = (C + numThreads - 1) / numThreads;
    fill_cache<<<blocks, numThreads>>>(key_cache, value_cache, qkv, l, pos, C, NH, T);

    float* q = qkv;
    float* k = key_cache + l * C * T;
    float* v = value_cache + l * C * T;
    //performing attention
    //q [NH, 1, HS] @ K [NH, T, HS].T -> [NH, 1, T]
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0;
    float beta = 0.0;
    cublasStatus_t status = cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, T, 1, head_size, &alpha, k, head_size, head_size * T, q, head_size, head_size, &beta, att, T, T, NH);
    if(status!=CUBLAS_STATUS_SUCCESS){
        cout << "CUBLAS error" << endl;
    }
    float scale = 1.0 / sqrt(head_size);    
    size_t memory = 2 * 512 * sizeof(float) / 32;
    softmax_kernel3<<<NH, 512, memory>>>(att, T, NH, pos+1, scale);

    status = cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, head_size, 1, T, &alpha, v, head_size, head_size * T, att, T, T, &beta, out, head_size, head_size, NH);
    if(status!=CUBLAS_STATUS_SUCCESS){
        cout << "CUBLAS error" << endl;
    }
    cublasDestroy(handle);
}
//-----------------------------------------------------------------------------------------------

void softmax_gpu(float* x, int N) {
    size_t memory = 2 * 512 * sizeof(float) / 32;
    softmax_kernel3<<<1, 512, memory>>>(x, N, 1, N, 1.0f);
}

