void encoder(float* out, float* wte, int token,int n_embd){
    for(int j = 0;j<n_embd;j++){
        out[j] = wte[token*n_embd+j];
    }
}