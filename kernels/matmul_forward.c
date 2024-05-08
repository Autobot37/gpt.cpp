#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

//inp(B,T,C) @  weight(3*C, C).T -> out(B,T,3*C)
void matmul_forward(float* out, float* inp, float* weight, float* bias, int B, int T, int C){
    for(int b = 0;b<B;b++){
        for(int t = 0;t<T;t++){
            float* inp_p = inp + b * T * C + t * C;
            float* out_p = out + b * T * 3 * C + t * 3 * C;
            for(int t2=0;t2<3*C;t2++){
                float* weight_p = weight + t2* C;
                float val = (bias != NULL) ? bias[t2] : 0.0f;
                for(int i = 0;i<C;i++){
                    val += inp_p[i] * weight_p[i];
                }
                out_p[t2] = val;
            }
        }
    }
}

int main(){
    
    return 0;
}