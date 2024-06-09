void matmul(float* out, float* x, float* w, int n, int d){
    for(int i = 0;i<d;i++){
        float val = 0.0f;
        for(int j = 0;j<n;j++){
            val += x[j]*w[i*n+j];
        }
        out[i] = val;
    }
}