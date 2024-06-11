void residual(float* out, float* in, int C){
    for(int i = 0;i<C;i++){
        out[i] += in[i];
    }
}