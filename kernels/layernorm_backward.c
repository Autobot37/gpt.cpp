#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void layernorm_backward(float* d_inp, float* d_weight, float* d_bias, float* d_out, 
                        float* mean, float* std_dev, float* inp, float* weight, float* bias,
                        int B, int T, int C){
    float eps = 1e-5f;
    for(int b = 0;b<B;b++){
        for(int t = 0;t<T;t++){
            float* inp_p = inp + b*T*C + t*C;
            float* out_p = d_out + b*T*C + t*C;
            float* d_inp_p = d_inp + b*T*C + t*C;
            float mean_p = mean[b*T + t];
            float std_dev_p = std_dev[b*T + t];
            for(int i = 0;i<C;i++){
                d_weight[i] += out_p[i] * ((inp_p[i] - mean_p) * std_dev_p);
                d_bias[i] += out_p[i];
                d_inp_p[i] += out_p[i] * weight[i] * std_dev_p;
            }
        }
    }
}

int main(){
    return 0;
}