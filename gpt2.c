#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "kernels/kernels.h"

typedef struct Config {
    int block_size;
    int vocab_size;
    int n_layer;
    int n_head;
    int n_embd;
}Config;

typedef struct TransformerWeights {

    float* wte;
    float* wpe;
    float* ln_1_w;
    float* ln_1_b;    
    float* c_attn_w;
    float* c_attn_b;
    float* c_proj_w;
    float* c_proj_b;
    float* ln_2_w;
    float* ln_2_b;
    float* mlp_c_fc_w;
    float* mlp_c_fc_b;
    float* mlp_c_proj_w;
    float* mlp_c_proj_b;
    float* ln_f_w;
    float* ln_f_b;
}Weights;

#define NUM_TENSORS 16

void fill_param_sizes(int* param_sizes,  Config config){
    int V = config.vocab_size;
    int C = config.n_embd;
    int T  = config.block_size;
    int L = config.n_layer;
    param_sizes[0] = V * C;
    param_sizes[1] = T * C;
    param_sizes[2] = L * C;
    param_sizes[3] = L * C;
    param_sizes[4] = L * (3*C) * C;
    param_sizes[5] = L * (3*C);
    param_sizes[6] = L * C * C;
    param_sizes[7] = L * C;
    param_sizes[8] = L * C;
    param_sizes[9] = L * C;
    param_sizes[10] = L* ( 4 * C) * C;
    param_sizes[11] = L * (4 * C);
    param_sizes[12] = L * C * (4 * C);
    param_sizes[13] = L * C;
    param_sizes[14] = C;
    param_sizes[15] = C;
}

float* alloc_weights(Weights* params, int* param_sizes){
    int num_params  =  0;
    for(int i = 0;i<NUM_TENSORS;i++){
        num_params += param_sizes[i];
    }
    float* params_memory = (float*)calloc(num_params , sizeof(float));
    float** ptrs[] = {
        &params->wte, &params->wpe, &params->ln_1_w, &params->ln_1_b,
        &params->c_attn_w, &params->c_attn_b, &params->c_proj_w, &params->c_proj_b,
        &params->ln_2_w, &params->ln_2_b, &params->mlp_c_fc_w, &params->mlp_c_fc_b,
        &params->mlp_c_proj_w, &params->mlp_c_proj_b, &params->ln_f_w, &params->ln_f_b
    };
    float* iter = params_memory;
    for(int i = 0;i<NUM_TENSORS;i++){
        *ptrs[i] = iter;
        iter += param_sizes[i];
    }
    return params_memory;
}

#define NUM_ACTIVATIONS 5
typedef struct Activations {
    float* encoded;
    float* ln_1;
    float* ln_1_mean;
    float* ln_1_rstd;
    float* qkv;
    float* preatt;
    float* att;
    float* atty;
}activations;

void fill_act_sizes(int* act_sizes, Config config, int B){
    int V = config.vocab_size;
    int C = config.n_embd;
    int T  = config.block_size;
    int L = config.n_layer;
    int NH = config.n_head;
    act_sizes[0] = B * T * C;
    act_sizes[1] = L * B * T * C;
    act_sizes[2] = L * B * T;
    act_sizes[3] = L * B * T;
    act_sizes[4] = L * B * T * 3 * C;
    act_sizes[5] = L * B * NH * T * T;
    act_sizes[6] = L * B * NH * T * T;
    act_sizes[7] = L * B * T * C;
}

float* alloc_activations(activations* act, int* act_sizes){
    int num_activations = 0;
    for(int i = 0;i<NUM_ACTIVATIONS;i++){
        num_activations += act_sizes[i];
    }
    float* act_memory = (float*)calloc(num_activations , sizeof(float));
    if(act_memory == NULL){
        printf("Activation memory assign failed\n");
    }
    float** ptrs[] = {
        &act->encoded, &act->ln_1, &act->ln_1_mean, &act->ln_1_rstd, &act->qkv,
        &act->preatt, &act->att, &act->atty
    };
    float* iter = act_memory;
    for(int i = 0;i<NUM_ACTIVATIONS;i++){
        *ptrs[i] = iter;
        iter += act_sizes[i];
    }

    return act_memory;
}

int main(){

    Weights weights;
    Config config;
    int B, T, C, V, L, NH;
    B = 4;
    config.vocab_size = V =  4;
    config.block_size = T = 4;
    config.n_layer = L = 2;
    config.n_head = NH = 2;
    config.n_embd = C = 4;
   
    activations act;
    
    int wt_sizes[NUM_TENSORS];
    fill_param_sizes(wt_sizes, config);
    alloc_weights(&weights, wt_sizes);
    int act_sizes[NUM_ACTIVATIONS];
    fill_act_sizes(act_sizes, config, B);
    alloc_activations(&act, act_sizes);
    int head_size = C / config.n_head;      

    int* input = (int*)calloc(B * T , sizeof(int));

    encoder_forward(act.encoded, input, weights.wte, weights.wpe, B, T, C);
    for(int l=0; l<L; l++){
        //weights for layer l
        float* ln_l_w = weights.ln_1_w + l * C;
        float* ln_l_b = weights.ln_1_b + l * C;
        float* c_attn_w = weights.c_attn_w + l * 3 * C * C;
        float* c_attn_b = weights.c_attn_b + l * 3 * C;
        float* c_proj_w = weights.c_proj_w + l * C * C;
        float* c_proj_b = weights.c_proj_b + l * C;
        //activations for layer l
        float* ln_l_out = act.ln_1 + l * B * T * C;
        float* ln_l1_mean = act.ln_1_mean + l * B * T;
        float* ln_l1_rstd = act.ln_1_rstd + l * B * T;
        float* qkv = act.qkv + l * B * T * 3 * C;
        float* preatt = act.preatt + l * B * NH * T * T;
        float * att = act.att + l * B * NH * T * T;
        float* atty = act.atty + l * B * T * C;

        layernorm_forward(ln_l_out, ln_l1_mean, ln_l1_rstd, act.encoded, ln_l_w, ln_l_b, B, T, C);
        matmul_forward(qkv, ln_l_out, c_attn_w, c_attn_b, B, T, C);
        attention_forward(atty, preatt, att, qkv, B, T, C, NH);
    }
    printf("wet pants\n");

    return 0;
}