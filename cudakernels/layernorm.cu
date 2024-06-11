void layernorm(float* out, float* x, float* w, float* b, int C){
    float mean = 0;
    float var = 0;
    for(int i = 0;i<C;i++){
        mean += x[i];
    }
    mean /= C;
    for(int i = 0;i<C;i++){
        float diff = x[i] - mean;
        var += diff * diff;
    }
    var /= C;
    float scale = 1.0 / sqrt(var + 1e-6);
    for(int i = 0;i<C;i++){
        out[i] = (x[i] - mean) * scale * w[i] + b[i];
    }
}