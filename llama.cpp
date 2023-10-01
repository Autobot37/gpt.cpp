#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
typedef struct{
    

} Config;

typedef struct {

} Weights;

typedef struct {

} Runstate;

typedef struct {
    Config config;
    Weights weights;
    Runstate runstate;
} Transformer;

void malloc_runstate(Runstate*S, Config* p){

}

void free_run_state(Runstate* s){

}

float* forward(Transformer* transformer, int token, int pos){
    Config* p = &transformer->config;
    Weights* w = &transformer->weights;
    Runstate* s = &transformer->runstate;
    
}


int main(){

    printf("Fucking _Fast");
    return 0;
}