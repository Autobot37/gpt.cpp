void softmax(float* x, int size){
    float max = x[0];
    for(int i = 1;i<size;i++){
        if(x[i]>max){
            max = x[i];
        }
    }
    float sum = 0.0f;
    for(int i = 0;i<size;i++){
        x[i] = expf(x[i]-max);
        sum += x[i];
    }
    for(int i = 0;i<size;i++){
        x[i] /= sum;
    }
}