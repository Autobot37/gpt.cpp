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