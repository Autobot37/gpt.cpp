#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>
#include <cblas.h>
#include <cblas-openblas.h>
#include <cfloat>

#include "kernels/kernels.h"
#include "tokenizer.h"

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

#define NUM_ACTIVATIONS 22
typedef struct Activations {
    float* encoded;
    float* ln_1;
    float* ln_1_mean;
    float* ln_1_rstd;
    float* qkv;
    float* preatt;
    float* att;
    float* atty;
    float* attproj;
    float* residual2;
    float* ln_2 ;
    float* ln_2_mean;
    float* ln_2_rstd;
    float* c_fc;
    float* fc_gelu;
    float* c_proj;
    float* residual3;
    float* ln_f;
    float* ln_f_mean;
    float* ln_f_rstd;
    float* logits;
    float* probs;
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
    act_sizes[8] = L * B * T * C;
    act_sizes[9] = L * B * T * C;
    act_sizes[10] = L * B * T * C;
    act_sizes[11] = L * B * T;
    act_sizes[12] = L * B * T;
    act_sizes[13] = L * B * T * 4 * C;
    act_sizes[14] = L * B * T * 4 * C;
    act_sizes[15] = L * B * T * C;
    act_sizes[16] = L * B * T * C;
    act_sizes[17] = B * T * C;
    act_sizes[18] = B * T;
    act_sizes[19] = B * T;
    act_sizes[20] = B * T * V;
    act_sizes[21] = B * T * V;
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
        &act->preatt, &act->att, &act->atty, &act->attproj,
        &act->residual2, &act->ln_2, &act->ln_2_mean, &act->ln_2_rstd,
        &act->c_fc, &act->fc_gelu, &act->c_proj,&act->residual3,
        &act->ln_f, &act->ln_f_mean, &act->ln_f_rstd, &act->logits,
        &act->probs
    };
    float* iter = act_memory;
    for(int i = 0;i<NUM_ACTIVATIONS;i++){
        *ptrs[i] = iter;
        iter += act_sizes[i];
    }

    return act_memory;
}

typedef struct Transformer {
    Weights weights;
    activations act;
    Config config;
    int params_sizes[NUM_TENSORS];
    int act_sizes[NUM_ACTIVATIONS];
    float* params_memory;
}GPT;


void gpt_build(GPT* model, const char* path){
    FILE* model_file = fopen(path, "rb");
    if(model_file == NULL){
        printf("Model file not found\n");
        exit(1);
    }   
    int model_header[8];
    fread(model_header, sizeof(int), 8, model_file);
    if(model_header[0] != 3737){
        printf("Model file not compatible\n");
        exit(1);
    }
    int T, V, L, C, NH;
    model->config.block_size = T = model_header[2];
    model->config.vocab_size = V = model_header[3];
    model->config.n_layer = L = model_header[4];
    model->config.n_head = NH = model_header[5];
    model->config.n_embd = C = model_header[6];

    printf("[GPT-2]\n");
    printf("Block size: %d\n", T);
    printf("Vocab size: %d\n", V);
    printf("Number of layers: %d\n", L);
    printf("Number of heads: %d\n", NH);
    printf("Embedding size: %d\n", C);

    fill_param_sizes(model->params_sizes, model->config);

    int num_params = 0;
    for(int i = 0;i<NUM_TENSORS;i++){
        num_params += model->params_sizes[i];
    }
    model->params_memory = alloc_weights(&model->weights, model->params_sizes);
    fread(model->params_memory, sizeof(float), num_params, model_file);
    fclose(model_file);
}

void gpt_forward(GPT* model, int* inputs, int B){

    int T = model->config.block_size;
    int V = model->config.vocab_size;
    int NH = model->config.n_head;
    int C = model->config.n_embd;
    int L = model->config.n_layer;
    int head_size = C / NH;      

    Weights weights = model->weights;
    activations act = model->act;

    float* residual = NULL;
    encoder_forward(act.encoded, inputs, weights.wte, weights.wpe, B, T, C);
    
    for(int l=0; l<L; l++){

        residual = (l==0) ? act.encoded : act.residual3 + (l-1) * B * T * C;
        //weights for layer l
        float* ln_l1_w = weights.ln_1_w + l * C;
        float* ln_l1_b = weights.ln_1_b + l * C;
        float* c_attn_w = weights.c_attn_w + l * 3 * C * C;
        float* c_attn_b = weights.c_attn_b + l * 3 * C;
        float* c_proj_w = weights.c_proj_w + l * C * C;
        float* c_proj_b = weights.c_proj_b + l * C;
        float* ln_l2_w = weights.ln_2_w + l * C;
        float* ln_l2_b = weights.ln_2_b + l * C;
        float* c_fc_w = weights.mlp_c_fc_w + l * 4 * C * C;
        float* c_fc_b = weights.mlp_c_fc_b + l * 4 * C;
        float* mlp_proj_w = weights.mlp_c_proj_w + l * 4 * C * C;
        float* mlp_proj_b = weights.mlp_c_proj_b + l * C;
        //activations for layer l
        float* ln_l1_out = act.ln_1 + l * B * T * C;
        float* ln_l1_mean = act.ln_1_mean + l * B * T;
        float* ln_l1_rstd = act.ln_1_rstd + l * B * T;
        float* qkv = act.qkv + l * B * T * 3 * C;
        float* preatt = act.preatt + l * B * NH * T * T;
        float * att = act.att + l * B * NH * T * T;
        float* atty = act.atty + l * B * T * C;
        float* attproj = act.attproj + l * B * T * C;
        float* residual2 = act.residual2 + l* B * T * C;
        float* ln_l2_out = act.ln_2 + l * B * T * C;
        float* ln_l2_mean = act.ln_2_mean + l * B * T;
        float* ln_l2_rstd = act.ln_2_rstd + l * B * T;
        float* c_fc = act.c_fc + l * B * T * 4 * C;
        float* fc_gelu = act.fc_gelu + l * B * T * 4 * C;
        float* c_proj = act.c_proj + l * B * T * C;
        float* residual3 = act.residual3 + l * B * T * C;

        layernorm_forward(ln_l1_out, ln_l1_mean, ln_l1_rstd, act.encoded, ln_l1_w, ln_l1_b, B, T, C);
        matmul_forward_blas(qkv, ln_l1_out, c_attn_w, c_attn_b, B, T, C, C);
        attention_forward_blas(atty, preatt, att, qkv, B, T, C, NH);
        matmul_forward_blas(attproj, atty, c_proj_w, c_proj_b, B, T, C, C);
        residual_forward(residual2, residual, attproj, B*T*C);
        layernorm_forward(ln_l2_out, ln_l2_mean, ln_l2_rstd, residual2, ln_l2_w, ln_l2_b, B, T, C);
        matmul_forward_blas(c_fc, ln_l2_out, c_fc_w, c_fc_b, B, T, C, 4*C);
        gelu_forward(fc_gelu, c_fc, B*T*4*C);
        matmul_forward_blas(c_proj, fc_gelu, mlp_proj_w, mlp_proj_b, B, T, 4*C, C);
        residual_forward(residual3, residual2, c_proj, B*T*C);
    }
    residual = act.residual3 + (L-1) * B * T * C;
    layernorm_forward(act.ln_f, act.ln_f_mean, act.ln_f_rstd, residual, weights.ln_f_w, weights.ln_f_b, B, T, C);
    matmul_forward_blas(act.logits, act.ln_f, weights.wte, NULL, B, T, C, V);
    softmax_forward_blas(act.probs, act.logits, B, T, V);
}

int main(){
    srand(time(NULL));

    double bstart, bend;
    bstart = omp_get_wtime();

    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "dictionary.bin");
    int vocab_size = tokenizer.vocab_size;
    
    GPT model;
    
    gpt_build(&model, "pythonscripts/params.bin");
    int T = model.config.block_size;
    int V = model.config.vocab_size;
    int NH = model.config.n_head;
    int C = model.config.n_embd;
    int L = model.config.n_layer;
    int B = 1;
    fill_act_sizes(model.act_sizes, model.config, B);
    alloc_activations(&model.act, model.act_sizes);
    int max_tokens = 4;
    int* inputs = (int*)mallocCheck((T+max_tokens)  * sizeof(int));
    for(int i = 0;i<T;i++){
        inputs[i] = 1;
    }

    bend = omp_get_wtime();
    double bcpu_time_used = bend - bstart;
    printf("Build time: %f seconds\n", bcpu_time_used);

    double start, end;
    start = omp_get_wtime();

    for(int i = 0;i<max_tokens;i++){
        gpt_forward(&model, inputs, B);
        float* probs = model.act.probs + (T-1) * V;
        int max_ind = 0;
        float max_val = -1.0f;
        for(int j = 0;j<V;j++){
            if(probs[j] > max_val){
                max_val = probs[j];
                max_ind = j;
            }
        }
        printf("%s ", tokenizer_decode(&tokenizer, max_ind));
        fflush(stdout);
        inputs[T] = max_ind;
        inputs = inputs + 1;
    }
    printf("\n");

    end = omp_get_wtime();
    double cpu_time_used = end - start;
    printf("Inference time: %f seconds\n", cpu_time_used);
    
    float tok_per_sec = (float)max_tokens / cpu_time_used;
    printf("Tokens per second: %f\n", tok_per_sec);

    tokenizer_free(&tokenizer);

    return 0;
}