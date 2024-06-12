#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tokenizer.h"

void rand_init(float *arr, int n){
    for(int i = 0; i < n; i++){
        arr[i] = (float)rand() / RAND_MAX;
    }
}

int main(){
    srand(time(NULL));

    int max_len = 8;

    float* logits = (float*)malloc(503024 * sizeof(float));

    for(int i = 0;i<max_len;i++){
        rand_init(logits, 503024);

        
    }




    return 0;
}