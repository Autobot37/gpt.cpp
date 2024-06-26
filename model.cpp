#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <float.h>
#include "tokenizer.h"
#include "kernels/kernels.h"

void rand_init(float* x, int N){
    for(int i = 0;i<N;i++){
        x[i] = (float)rand() / RAND_MAX;
    }
}

void print(float* x, int N){
    for(int i = 0;i<N;i++){
        printf("%.6f ", x[i]);
    }
    printf("\n");
    printf("----------------------\n");
}
void isequal(float* a, float* b, int n){
    float maxval = -INFINITY;
    for(int i = 0;i<n;i++){
        maxval = fmaxf(maxval, fmaxf(a[i], b[i]));
    }
    float eps = 1e-4;
    for(int i = 0;i<n;i++){
        if(fabs(a[i] - b[i]) > eps * (maxval + 1)){
            printf("Mismatch at index %d CPU: %.6f GPU: %.6f\n", i, a[i], b[i]);
        }
    }
}

bool CHECK = false;

typedef struct {
    int vocab_size;
    float temprature;
} Sampler;

void build_sampler(Sampler* sampler, int vocab_size, float temprature){
    sampler->vocab_size = vocab_size;
    sampler->temprature = temprature;
}

int sample_multi(float* probabilities, int n, float coin){
    float cdf = 0.0f;
    for(int i = 0;i<n;i++){
        cdf += probabilities[i];
        if(cdf > coin){
            return i;
        }
    }
    return n-1;
}
int sample_argmax(float* probabilities, int n){
    int max_idx = 0;
    float max_val = -FLT_MAX;
    for(int i = 0;i<n;i++){
        if(probabilities[i] > max_val){
            max_val = probabilities[i];
            max_idx = i;
        }
    }
    return max_idx;
}

int sample(Sampler* sampler, float* logits){
    int next;
    for(int i = 0;i<sampler->vocab_size;i++){
        logits[i] /= sampler->temprature;
    }
    softmax(logits, sampler->vocab_size);
    float coin = (float)rand() / RAND_MAX;
    if(CHECK){
        next = sample_argmax(logits, sampler->vocab_size);
    }
    else{
        next = sample_multi(logits, sampler->vocab_size, coin);
    }
    return next;
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
#define COLOR_RESET "\x1b[0m"
#define COLOR_GREEN "\x1b[32m"
#define STYLE_BOLD "\x1b[1m"
#define STYLE_UNDERLINE "\x1b[4m"

int* generate(Model* model, Tokenizer* tokenizer, int max_tokens, vector<int>& tokens, Sampler* sampler, int num_tokens){
    int* generated_tokens = (int*)malloc((max_tokens + num_tokens) * sizeof(int));
    float* logits;

    int token = tokens[0];
    int pos = 0;
    int next;
    clock_t start, end;
    #pragma omp master
    {
        start = clock();
    }
    if(!CHECK)printf(COLOR_GREEN STYLE_BOLD "%s" COLOR_RESET, tokenizer->decode(token, token).c_str());
    generated_tokens[pos] = token;
    while(pos < max_tokens + num_tokens - 1){
        float* logits = forward(model, token, pos);

        if(pos < num_tokens - 1){
            next = tokens[pos + 1];
        }
        else{
            next = sample(sampler, logits);
        }
        pos++;
        token = next;
        // printf(" %d", token);
        string piece = tokenizer->decode(next, token);
        piece = safe_printf(piece);
        if(!CHECK)printf(COLOR_GREEN STYLE_BOLD "%s" COLOR_RESET, piece.c_str());
        fflush(stdout);
        generated_tokens[pos] = token;
        if(!CHECK && token==1)break;
    }
    printf("\n");
    #pragma omp master
    {
        end = clock();
        double time_taken = ((double)end - start) / CLOCKS_PER_SEC;
        double one_token = (time_taken / max_tokens) * 1000;
        printf("One token took %.6f ms\n", one_token);
    }

    return generated_tokens;
}

void completion(Model* model, Tokenizer* tokenizer, Sampler* sampler, char* prompt, int max_tokens){
    vector<int>tokens = tokenizer->encode(prompt);
    int num_tokens = tokens.size();
    printf("Number of tokens: %d\n", num_tokens);
    generate(model, tokenizer, max_tokens, tokens, sampler, num_tokens);
}

void chat(Model* model, Tokenizer* tokenizer, Sampler* sampler) {
    char prompt[512];
    char context[512] = ""; 
    while (true) {
        printf("You: ");
        fgets(prompt, 512, stdin);
        int len = strlen(prompt);
        if (len > 0 && prompt[len - 1] == '\n') {
            prompt[len - 1] = '\0';
        }

        if (strcmp(prompt, "exit") == 0) {
            break;
        }

        char combined_prompt[1024];
        snprintf(combined_prompt, sizeof(combined_prompt), "%s %s", context, prompt);
        completion(model, tokenizer, sampler, combined_prompt, 128);
        strncpy(context, combined_prompt, sizeof(context) - 1);
        context[sizeof(context) - 1] = '\0'; 
    }
}
void check_output(char* path, Model* model, Sampler* sampler, Tokenizer* tokenizer){
    FILE* file = fopen(path, "r");
    if(file == NULL){
        printf("Error opening file\n");
        exit(1);
    }
    int header[8];
    size_t ret = fread(header, sizeof(int), 8, file);
    if(header[0] != 3737){
        printf("Invalid file\n");
        exit(1);
    }
    int vocab_size = header[1];
    int max_length = header[2];
    int num_tokens = header[3];

    int file_tokens[num_tokens];
    size_t num_read = fread(file_tokens, sizeof(int), num_tokens, file);
    if(num_read != num_tokens){
        printf("Invalid number of tokens\n");
        exit(1);
    }

    int file_next_tokens[max_length + num_tokens];
    size_t length = fread(file_next_tokens, sizeof(int), max_length, file);
    if(length != max_length){
        printf("Invalid number of tokens\n");
        exit(1);
    }
    fclose(file);

    vector<int> v_file_tokens(file_tokens, file_tokens + num_tokens);
    int* gen_tokens = generate(model, tokenizer, max_length, v_file_tokens, sampler, num_tokens);
    for(int i = 0;i<max_length;i++){
        assert(gen_tokens[i] == file_next_tokens[i]);
    }
}


int main(){

    srand(time(NULL));

    Model model;
    create_model(&model, "params.bin");

    fill_activation_sizes(model.activation_sizes, &model.config);
    alloc_activations(&model.activations, model.activation_sizes);
    Tokenizer tokenizer;
    tokenizer.init("tokenizer.bin");
    
    Sampler sampler;
    build_sampler(&sampler, model.config.vocab_size, 0.7);
    CHECK = true;
    check_output("debug.bin", &model, &sampler, &tokenizer);
    printf("All tests passed\n");
    printf("-----------------------------\n");
    CHECK = false;

    printf("\033[H\033[J");
    completion(&model, &tokenizer, &sampler, "Let me sing a song for you", 128);
    chat(&model, &tokenizer, &sampler);


    return 0;

}