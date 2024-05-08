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
            float* d_inp_p = d_inp + b * T * C + t * C;
            float* d_out_p = d_out + b * T * C + t * C;
            float* inp_p = inp + b * T * C + t * C;
            float mean_p = mean[b * T + t];
            float rstd_p = std_dev[b * T + t];
            float* weight_p = weight + t * C;
            float* bias_p = bias + t * C;

            float dnorm_mean = 0.0f;
            float norm_dnorm = 0.0f;
            
            for(int i = 0;i<C;i++){
                float dnorm = d_out_p[i] * weight_p[i];
                float norm = (inp_p[i] - mean_p) * rstd_p;
                dnorm_mean += dnorm;
                norm_dnorm += norm * (dnorm * norm);
            }
            dnorm_mean = dnorm_mean / C;
            norm_dnorm = norm_dnorm / C;

            for(int i = 0;i<C;i++){
                d_weight[i] += d_out_p[i] * ((inp_p[i] - mean_p) * rstd_p);
                d_bias[i] += d_out_p[i];
                //for inputs
                float dnorm = d_out_p[i] * weight_p[i];
                d_inp_p[i] = dnorm - dnorm_mean - dnorm_mean;
                d_inp_p[i] *= rstd_p;
            }
        }
    }
}

int main(){
    return 0;
}