#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void crossentropy_forward(float* losses, float* probs, int * target, int B, int T, int V){
    for(int b = 0;b<B;b++){
        for(int t = 0;t<T;t++){
            float* probs_p = probs + b * T * V + t * V;
            int ix = target[b * T + t];
            losses[b * T + t] = -logf(probs_p[ix]);
        }
    }
}

int main(){
    
    return 0;
}