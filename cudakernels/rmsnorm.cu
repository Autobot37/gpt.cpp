void rmsnorm(float* out, float* x, float* w, int dim){
    float ss = 0.0f;
    for(int j = 0;j<dim;j++){
        ss += x[j]*x[j];
    }
    ss /= dim;
    ss += 1e-6f;
    ss = 1.0f/sqrtf(ss);
    for(int j = 0;j<dim;j++){
        out[j] = w[j] * (ss*x[j]);
    }
}