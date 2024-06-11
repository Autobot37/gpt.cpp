void matmul(float* out, float* in, float* w, float* b, int N ,int D){
    //in is D, w is N,D, b is N, out is N
    int i;
    #pragma omp parallel for private(i)
    for(i = 0;i<N;i++){
        float sum = (b!=NULL) ? b[i] : 0;
        for(int j = 0;j<D;j++){
            sum += in[j] * w[i*D + j];
        }
        out[i] = sum;
    } 
}

__global__
void matmul_kernel(float* out, float* in, float* w, float* b, int N ,int D){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N){
        float sum = (b!=NULL) ? b[i] : 0;
        for(int j = 0;j<D;j++){
            sum += in[j] * w[i*D + j];
        }
        out[i] = sum;
    }
}

void matmul_gpu(float* out, float* in, float* w, float* b, int N ,int D){
    int num_threads = 1024;
    int num_blocks = (N + num_threads - 1) / num_threads;
    matmul_kernel<<<num_blocks, num_threads>>>(out, in, w, b, N, D);
}