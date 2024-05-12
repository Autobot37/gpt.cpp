#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "../utils.h"

void encoder_forward(float* out, int* inp, float* wte, float* wpe, int B, int T, int C){
    for(int b = 0;b<B;b++){
        for(int t = 0;t<T;t++){
            float* out_p = out + b * T * C + t * C;
            float* wte_p = wte + inp[b * T + t] * C;
            float* wpe_p = wpe + t * C;
            for(int i = 0;i<C;i++){
                out_p[i] = wte_p[i] + wpe_p[i];
            }
        }
    };
}

int main(){

    int mul = 8;
    int C = 32*mul;
    int NH = 4*mul;
    int T = 128*mul;
    int V = 4096*mul;
    int L = 4*mul;
    int B = 1*mul;
    int* inp = (int*)mallocCheck(sizeof(int) * B * T);
    float* out = (float*)mallocCheck(sizeof(int) * B * T * C);
    float* wpe = (float*)mallocCheck(sizeof(int) * T * C);
    float* wte = (float*)mallocCheck(sizeof(int) * V * C);
    clock_t start, end, end2;
    double time_used;
    start = clock();

    encoder_forward(out, inp, wte, wpe, B,T,C);

    end = clock();
    time_used = ((double)(end - start)) / CLOCKS_PER_SEC;  
    printf("Time Used CPU: %lf seconds\n", time_used);
    
    free(inp);
    free(out);
    free(wpe);
    free(wte);

    return 0;
}