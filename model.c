#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "tokenizer.h"

void rand_init(float* x, int N){
    for(int i = 0;i<N;i++){
        x[i] = (float)rand() / RAND_MAX;
    }
}

void print(float* x, int N){
    for(int i = 0;i<N;i++){
        printf("%.4f ", x[i]);
    }
    printf("\n");
    printf("----------------------\n");
}

typedef struct Config {
    int block_size;
    int vocab_size;
    int n_layer;
    int n_head;
    int n_embd;
} Config;

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

void fill_param_sizes(int* param_sizes,  Config* config){
    int V = config->vocab_size;
    int C = config->n_embd;
    int T  = config->block_size;
    int L = config->n_layer;
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

    printf("Number of parameters: %.2fM\n", num_params/1e6);
    long long size = num_params * sizeof(float);
    printf("Size of parameters: %lldMB\n", size/1024/1024);

    float* params_memory = (float*)calloc(num_params , sizeof(float));

    if(params_memory == NULL){
        printf("Error allocating memory\n");
        exit(1);
    }

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

#define NUM_ACTIVATIONS 11

typedef struct Activations {
    float* x;
    float* qkv;
    float* att;
    float* atty;
    float* attproj;
    float* c_fc;
    float* key_cache;
    float* value_cache;
    float* logits;
    float* residual1;
    float* residual2;
} Activations;

void fill_activation_sizes(int* activation_sizes, Config* config){
    int C = config->n_embd;
    int T = config->block_size;
    int L = config->n_layer;
    int NH = config->n_head;
    int V = config->vocab_size;
    activation_sizes[0] = C;
    activation_sizes[1] = 3*C;
    activation_sizes[2] = NH * T;
    activation_sizes[3] = C;
    activation_sizes[4] = C;
    activation_sizes[5] = 4*C;
    activation_sizes[6] = L * T * C;
    activation_sizes[7] = L * T * C;
    activation_sizes[8] = V;
    activation_sizes[9] = C;
    activation_sizes[10] = C;
}

float* alloc_activations(Activations* activations, int* activation_sizes){
    int num_activations = 0;
    for(int i = 0;i<NUM_ACTIVATIONS;i++){
        num_activations += activation_sizes[i];
    }

    long long size = num_activations * sizeof(float);
    printf("Size of activations: %lldMB\n", size/1024/1024);
    
    float* activations_memory = (float*)calloc(num_activations, sizeof(float));
    if(activations_memory == NULL){
        printf("Error allocating memory\n");
        exit(1);
    }
    float** ptrs[] = {
        &activations->x, &activations->qkv, &activations->att, &activations->atty,
        &activations->attproj, &activations->c_fc, &activations->key_cache, &activations->value_cache,
        &activations->logits, &activations->residual1, &activations->residual2
    };
    float* iter = activations_memory;
    for(int i = 0;i<NUM_ACTIVATIONS;i++){
        *ptrs[i] = iter;
        iter += activation_sizes[i];
    }
    return activations_memory;
}

typedef struct Model {
    Weights weights;
    Activations activations;
    Config config;
    int params_sizes[NUM_TENSORS];
    int activation_sizes[NUM_ACTIVATIONS];
    float* params_memory; 
} Model;

void create_model(Model* model, const char* path){
    FILE* model_file = fopen(path, "rb");
    if(model_file == NULL){
        printf("Error opening file\n");
        exit(1);
    }
    int model_header[8];
    size_t ret = fread(model_header, sizeof(int), 8, model_file);
    if(model_header[0] != 3737 || ret != 8){
        printf("Invalid model file\n");
        exit(1);
    }
    int T,V,L,C,NH;
    model->config.block_size = T = model_header[2];
    model->config.vocab_size = V = model_header[3];
    model->config.n_layer = L = model_header[4];
    model->config.n_head = NH = model_header[5];
    model->config.n_embd = C = model_header[6];

    printf("[Configuration]\n");
    printf("Block size: %d\n", T);
    printf("Vocab size: %d\n", V);
    printf("Number of layers: %d\n", L);
    printf("Embedding size: %d\n", C);
    printf("Number of heads: %d\n", NH);

    fill_param_sizes(model->params_sizes, &model->config);

    size_t num_params = 0;
    for(int i = 0;i<NUM_TENSORS;i++){
        num_params += model->params_sizes[i];
    }
    model->params_memory = alloc_weights(&model->weights, model->params_sizes);
    ret = fread(model->params_memory, sizeof(float), num_params, model_file);
    if(ret != num_params){
        printf("Error reading model file\n");
        exit(1);
    }
    fclose(model_file);
}

void embed(float* x, float* wte, float* wpe, int token, int pos, int C){
    for(int i = 0;i<C;i++){
        x[i] = wte[token*C + i] + wpe[pos*C + i];
    }
}

void layernorm(float* out, float* x, float* w, float* b, int C){
    float mean = 0;
    float var = 0;
    for(int i = 0;i<C;i++){
        mean += x[i];
    }
    mean /= C;
    for(int i = 0;i<C;i++){
        float diff = x[i] - mean;
        var += diff * diff;
    }
    var /= C;
    float scale = 1.0 / sqrt(var + 1e-6);
    for(int i = 0;i<C;i++){
        out[i] = (x[i] - mean) * scale * w[i] + b[i];
    }
}

void matmul(float* out, float* in, float* w, float* b, int N ,int D){
    //in is D, w is N,D, b is N, out is N
    int i;
    #pragma omp parallel for private(i)
    for(i = 0;i<N;i++){
        float sum = (b!=NULL) ? b[i] : 0;
        for(int j = 0;j<D;j++){
            sum += in[j] * w[i*D + j];
        }
        out[i] = sum;
    } 
}
void softmax(float* x, int N){
    float max = x[0];
    for(int i = 1;i<N;i++){
        if(x[i] > max){
            max = x[i];
        }
    }
    float sum = 0;
    for(int i = 0;i<N;i++){
        x[i] = exp(x[i] - max);
        sum += x[i];
    }
    for(int i = 0;i<N;i++){
        x[i] /= sum;
    }
}

void residual(float* out, float* in, int C){
    for(int i = 0;i<C;i++){
        out[i] += in[i];
    }
}

void gelu(float* x, int C){
    for(int i = 0;i<C;i++){
        float u = x[i];
        x[i] = 0.5 * u * (1 + tanh(sqrt(2.0/M_PI) * (u + 0.044715 * u * u * u)));
    }
}

void attention(float* out, float* att, float* qkv, float* key_cache, float* value_cache, int l, int pos, int C, int NH, int head_size, int T){

    float* q = qkv;
    memcpy(key_cache + l * C * T + pos * C, qkv + C, C * sizeof(float));
    memcpy(value_cache + l * C * T + pos * C, qkv + 2*C, C * sizeof(float));

    float scale = 1.0 / sqrt(head_size);

    float* k = key_cache + l * C * T;
    float* v = value_cache + l * C * T;

    int h;
    #pragma omp parallel for private(h)
    for(h = 0;h<NH;h++){

        float* qh = q + h * head_size;
        float* atth = att + h * T;

        for(int t = 0;t<=pos;t++){
            float* kh = k + t * C + h * head_size;
            float score = 0.0f;
            for(int i = 0;i<head_size;i++){
                score += qh[i] * kh[i];
            }
            score *= scale;
            atth[t] = score;
        }
        for(int t=pos+1;t<T;t++){
            atth[t] = -INFINITY;
        }

        softmax(atth, T);

        float* outh = out + h * head_size;
        memset(outh, 0, head_size * sizeof(float));
        for(int t = 0;t<=pos;t++){
            float* vh = v + t * C + h * head_size;
            float score = atth[t];
            for(int i = 0;i<head_size;i++){
                outh[i] += score * vh[i];
            }
        }
    }
}
float* forward(Model* model, int token, int pos){

    Config* c = &model->config;

    int V = c->vocab_size;
    int L = c->n_layer;
    int C = c->n_embd;
    int NH = c->n_head;
    int T = c->block_size;
    int head_size = C / NH;

    Weights* w = &model->weights;
    Activations* a = &model->activations;

    embed(a->x, w->wte, w->wpe, token, pos, C);

    for(int l=0;l<L;l++){

        layernorm(a->residual1, a->x, w->ln_1_w + l*C, w->ln_1_b + l*C, C);
        matmul(a->qkv, a->residual1, w->c_attn_w + l*3*C*C, w->c_attn_b + l*3*C, 3*C, C);
        attention(a->atty, a->att, a->qkv, a->key_cache, a->value_cache, l, pos, C, NH, head_size, T);
        matmul(a->attproj, a->atty, w->c_proj_w + l*C*C, w->c_proj_b + l*C, C, C);
        residual(a->x, a->attproj, C);
        layernorm(a->residual2, a->x, w->ln_2_w + l*C, w->ln_2_b + l*C, C);
        matmul(a->c_fc, a->residual2, w->mlp_c_fc_w + l*4*C*C, w->mlp_c_fc_b + l*4*C, 4*C, C);
        gelu(a->c_fc, 4*C);
        matmul(a->residual2, a->c_fc, w->mlp_c_proj_w + l*C*4*C, w->mlp_c_proj_b + l*C, C, 4*C);
        residual(a->x, a->residual2, C);
    }
    layernorm(a->x, a->x, w->ln_f_w, w->ln_f_b, C);
    matmul(a->logits, a->x, w->wte, NULL, V, C);
    return a->logits;
}

void generate(Model* model, Tokenizer* tokenizer, int max_tokens){

    int token = 50256;
    int pos = 0;

    clock_t start, end;
    start = clock();

    for(int i = 0;i<max_tokens;i++){
        float* logits = forward(model, token, pos);
        softmax(logits, model->config.vocab_size);
        int next = 0;
        float max = 0;
        for(int i = 0;i<model->config.vocab_size;i++){
            if(logits[i] > max){
                max = logits[i];
                next = i;
            }
        }
        const char* piece = tokenizer_decode(tokenizer, next);
        // safe_printf(piece);
        printf("%d ", next);
        fflush(stdout);
        token = next;
        pos++;
    }
    printf("\n");
    end = clock();
    double time_taken = ((double)end - start) / CLOCKS_PER_SEC;
    double one_token = (time_taken / max_tokens) * 1000;
    printf("One token took %.6f ms\n", one_token);
}

int main(){

    Model model;
    create_model(&model, "params.bin");

    fill_activation_sizes(model.activation_sizes, &model.config);
    alloc_activations(&model.activations, model.activation_sizes);
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "tokenizer.bin");

    generate(&model, &tokenizer, 32);
  
    return 0;

}