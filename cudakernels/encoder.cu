void embed(float* x, float* wte, float* wpe, int token, int pos, int C){
    for(int i = 0;i<C;i++){
        x[i] = wte[token*C + i] + wpe[pos*C + i];
    }
}