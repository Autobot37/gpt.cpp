void embed(float* x, float* wte, float* wpe, int token, int pos, int C){
    for(int i = 0;i<C;i++){
        x[i] = wte[token*C + i] + wpe[pos*C + i];
    }
}

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