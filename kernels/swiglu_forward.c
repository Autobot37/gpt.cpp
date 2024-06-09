#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void gelu_forward(float* out, float* inp, int dim){
    for(int i = 0;i<dim;i++){
        float val = inp[i];
        out[i] = 0.5f * val * (1.0f + tanhf(0.7978845608f * (val + 0.044715f * val * val * val)));
    }
}

int main(){
    
    return 0;
}