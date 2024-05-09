#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void crossentropy_softmax_backward(float* d_logits,
    float* dlosses, float* probs, int * target, int B, int T, int V){
    for(int b = 0;b<B;b++){
        for(int t = 0;t<T;t++){
            float* probs_p = probs + b * T * V + t * V;
            float* dlogits_p = d_logits + b * T * V + t * V;
            float dloss_p = dlosses[b * T + t];
            int ix = target[b * T + t];
            for(int i = 0;i<V;i++){
                float p = probs_p[i];
                float indicator = i==ix ? 1.0f : 0.0f;
                dlogits_p[i] = (p - indicator) * dloss_p;
            }
        }
    }
}

int main(){
    
    return 0;
}