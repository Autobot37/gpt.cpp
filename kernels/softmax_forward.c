#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void softmax_forward(float* out, float* inp, int B, int T, int V){
    for(int b=0;b<B;b++){
        for(int t=0;t<T;t++){
            float* inp_p = inp + b * T * V + t * V;
            float* out_p = out + b * T * V + t * V;
            float max_val = -1e-5f;
            for(int i=0;i<V;i++){
                if(inp_p[i] > max_val){
                    max_val = inp_p[i];
                }
            }
            float sum = 0.0f;
            for(int i=0;i<V;i++){
                out_p[i] = expf(inp_p[i] - max_val);
                sum += out_p[i];
            }
            for(int i=0;i<V;i++){
                out_p[i] = out_p[i] / sum;
            }
        }
    }
}
int main(){
    
    return 0;
}