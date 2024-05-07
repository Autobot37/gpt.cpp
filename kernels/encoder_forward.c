#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void print_mat(float* matrix, int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        printf("Matrix %d:\n", b);
        for (int t = 0; t < T; t++) {
            for (int c = 0; c < C; c++) {
                printf("%f ", matrix[b * T * C + t * C + c]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

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

    int B = 8;
    int T = 8;
    int C = 8;
    int V = 8;
    float* wte = (float*)malloc(V*C*sizeof(float));
    float* wpe = (float*)malloc(T*C*sizeof(float));
    int* inp = (int*)malloc(B*T*sizeof(int));
    float* out = (float*)malloc(B*T*C*sizeof(float));
    encoder_forward(out, inp, wte, wpe, B, T, C);
    print_mat(out, B, T, C);

    return 0;
}