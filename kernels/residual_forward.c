#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void residual_forward(float* out, float* inp, float* skip, int dim){
    for(int i = 0;i<dim;i++){
        out[i] = inp[i] + skip[i];
    }
}

int main(){
    
    return 0;
}