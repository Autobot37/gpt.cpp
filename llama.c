#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <time.h>

typedef struct {
    int n_embd;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
} Config;

typedef struct {

    float* wte;
    float* rms_attn_weight;
    float* rms_ffn_weight;
    float* wq;
    float* wk;
    float* wv;
    float* wo;
    float* w1;
    float* w2;
    float* w3;
    float* rms_final_weight;
    float* wcls;
} Weights;

typedef struct {
    float* x;
    float* xb;
    float* q;
    float* k;
    float* v;
    float* att;
    float* xb2;
    float* hb;
    float* hb2;
    float* logits;
    float* key_cache;
    float* value_cache;
} Activations;

typedef struct {
    Config config;
    Weights weights;
    Activations activations;
    int fd;
    float* data;
    ssize_t size;
} Model;

void alloc_activations(Activations* s, Config* config){
    int kv_dim = config->n_kv_heads * config->n_embd / config->n_heads;
    s->x = (float*)malloc(config->n_embd*sizeof(float));
    s->xb = (float*)malloc(config->n_embd*sizeof(float));
    s->xb2 = (float*)malloc(config->n_embd*sizeof(float));
    s->hb = (float*)malloc(config->hidden_dim*sizeof(float));
    s->hb2 = (float*)malloc(config->hidden_dim*sizeof(float));
    s->q = (float*)malloc(config->n_embd*sizeof(float));
    s->key_cache = (float*)malloc(config->n_layers*config->seq_len*kv_dim*sizeof(float));
    s->value_cache = (float*)malloc(config->n_layers*config->seq_len*kv_dim*sizeof(float));
    s->att = (float*)malloc(config->n_heads * config->seq_len*sizeof(float));
    s->logits = (float*)malloc(config->vocab_size*sizeof(float));
}

void memory_map_weights(Weights* w, float* data, Config* config, int shared_weights){
    int head_size = config->n_embd / config->n_heads;
    int n_layers = config->n_layers;

    ssize_t num_params = 0;
    long long x = 0;

    w->wte = data;
    x = config->vocab_size * config->n_embd;
    data += x;
    num_params += x;

    w->rms_attn_weight = data;
    x = n_layers * config->n_embd;
    data += x;
    num_params += x;

    w->wq = data;
    x = n_layers * config->n_embd * (config->n_heads * head_size);
    data += x;
    num_params += x;

    w->wk = data;
    x = n_layers * config->n_embd * (config->n_kv_heads * head_size);
    data += x;
    num_params += x;

    w->wv = data;
    x = n_layers * config->n_embd * (config->n_kv_heads * head_size);
    data += x;
    num_params += x;

    w->wo = data;
    x = n_layers * config->n_embd * config->n_embd;
    data += x;
    num_params += x;

    w->rms_ffn_weight = data;
    x = n_layers * config->n_embd;
    data += x;
    num_params += x;

    w->w1 = data;
    x = n_layers * config->n_embd * config->hidden_dim;
    data += x;
    num_params += x;

    w->w2 = data;
    x = n_layers * config->hidden_dim * config->n_embd;
    data += x;
    num_params += x;

    w->w3 = data;
    x = n_layers * config->hidden_dim * config->n_embd;
    data += x;
    num_params += x;

    w->rms_final_weight = data;
    x = config->n_embd;
    data += x;
    num_params += x;

    data += config->seq_len * head_size / 2;
    data += config->seq_len * head_size / 2;

    w->wcls = shared_weights ? w->wte : data;
    if(shared_weights){
        num_params += 0;
    }
    else{
        x = config->n_embd * config->vocab_size;
        data += x;
        num_params += x;
    }
    printf("Number of parameters: %ld\n", num_params);
}

void read_checkpoint(char* filename, Config* config, Weights* weights, 
                    int* fd, float** data, ssize_t* size){

    FILE* file = fopen(filename, "rb");
    if(file == NULL){
        printf("Error opening file\n");
        exit(1);
    }
    if(fread(config, sizeof(Config), 1, file) != 1){
        printf("Error reading config\n");
        exit(1);
    }
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    fseek(file, 0, SEEK_END);
    *size = ftell(file);

    printf("Size in mb: %f\n", *size/1024.0/1024.0);

    if(shared_weights){
        printf("Using shared weights\n");
    }
    else{
        printf("Using separate weights\n");
    }
    fclose(file);
    *fd = open(filename, O_RDONLY);
    if(*fd == -1){
        printf("Error opening file\n");
        exit(1);
    }

    *data = (float*)mmap(NULL, *size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if(*data == MAP_FAILED){
        printf("Error mapping file\n");
        exit(1);
    }
    float* ptr = *data + sizeof(Config)/sizeof(float);
    memory_map_weights(weights, ptr, config, shared_weights);
}

void encoder(float* out, float* wte, int token,int n_embd){
    for(int j = 0;j<n_embd;j++){
        out[j] = wte[token*n_embd+j];
    }
}

void rmsnorm(float* out, float* x, float* w, int dim){
    float ss = 0.0f;
    for(int j = 0;j<dim;j++){
        ss += x[j]*x[j];
    }
    ss /= dim;
    ss += 1e-6f;
    ss = 1.0f/sqrtf(ss);
    for(int j = 0;j<dim;j++){
        out[j] = w[j] * (ss*x[j]);
    }
}

void matmul(float* out, float* x, float* w, int n, int d){
    for(int i = 0;i<d;i++){
        float val = 0.0f;
        for(int j = 0;j<n;j++){
            val += x[j]*w[i*n+j];
        }
        out[i] = val;
    }
}

void softmax(float* x, int size){
    float max = x[0];
    for(int i = 1;i<size;i++){
        if(x[i]>max){
            max = x[i];
        }
    }
    float sum = 0.0f;
    for(int i = 0;i<size;i++){
        x[i] = expf(x[i]-max);
        sum += x[i];
    }
    for(int i = 0;i<size;i++){
        x[i] /= sum;
    }
}

void apply_rotemb(float* k,float* q, int pos, int dim, int head_size, int kv_dim){
    for(int i = 0;i<dim;i+=2){
        int head_dim = i % head_size;
        float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
        float val = pos * freq;
        float fcr = cosf(val);
        float fci = sinf(val);
        int rotn = i < kv_dim ? 2 : 1;
        for(int v = 0;v<rotn;v++){
            float* vec = (v==0)? q : k;
            float v0 = vec[i];
            float v1 = vec[i+1];
            vec[i] = v0*fcr - v1*fci;
            vec[i+1] = v0*fci + v1*fcr;
        }
    }
}

void residual(float* out, float* x, int dim){
    for(int i = 0;i<dim;i++){
        out[i] += x[i];
    }
}

void swiglu(float* hb, float* hb2, int dim){
    for(int i = 0;i<dim;i++){
        float val = hb[i];
        val *= (1.0f / (1.0f + expf(-val)));
        val *= hb2[i];
        hb[i] = val;
    }
}

void print(float* x, int size){
    for(int i = 0;i<size;i++){
        printf("%f ", x[i]);
    }
    printf("\n");
}

float* forward(Model* model, int token, int pos){

    Activations* acts = &model->activations;
    Config* config = &model->config;
    Weights* weights = &model->weights;
    int dim = config->n_embd;
    int head_size = dim/config->n_heads;
    int kv_dim = config->n_kv_heads * dim / config->n_heads;
    int kv_mul = config->n_heads / config->n_kv_heads;

    encoder(acts->x, weights->wte, token, config->n_embd);
    for(int L = 0;L<config->n_layers;L++){
    
        rmsnorm(acts->xb, acts->x, weights->rms_attn_weight + L*dim, dim);
        matmul(acts->q, acts->xb, weights->wq + L*dim*dim, dim, dim);
        int skip = L * config->seq_len * kv_dim;
        acts->k = acts->key_cache + skip + pos*kv_dim;
        acts->v = acts->value_cache + skip + pos*kv_dim;
        matmul(acts->k, acts->xb, weights->wk + L*dim*kv_dim, dim, kv_dim);
        matmul(acts->v, acts->xb, weights->wv + L*dim*kv_dim, dim, kv_dim);

        apply_rotemb(acts->k, acts->q, pos, dim, head_size, kv_dim);

        for(int h =0;h < config->n_heads; h++){

            float* q = acts->q + h*head_size;
            float* att = acts->att + h* config->seq_len;
            
            for(int t= 0;t<=pos;t++){
                float* k = acts->key_cache + skip + t*kv_dim + (h/kv_mul)*head_size;

                float score = 0.0f;
                for(int j = 0;j<head_size;j++){
                    score += q[j]*k[j];
                }
                score /= sqrtf(head_size);
                att[t] = score;
            }

            softmax(att, pos + 1);

            float* xb = acts->xb + h*head_size;
            memset(xb, 0, head_size*sizeof(float));
            for(int t = 0;t<=pos;t++){
                float* v = acts->value_cache + skip + t*kv_dim + (h/kv_mul)*head_size;
                float score = att[t];
                for(int j = 0;j<head_size;j++){
                    xb[j] += score*v[j];
                }
            }

        }
        matmul(acts->xb2, acts->xb, weights->wo + L * dim * dim, dim, dim);

        residual(acts->x, acts->xb2, dim);

        rmsnorm(acts->xb, acts->x, weights->rms_ffn_weight + L*dim, dim);

        matmul(acts->hb, acts->xb, weights->w1 + L*dim*config->hidden_dim, dim, config->hidden_dim);
        matmul(acts->hb2, acts->hb, weights->w3 + L*config->hidden_dim*dim, dim, config->hidden_dim);

        swiglu(acts->hb, acts->hb2, config->hidden_dim);

        matmul(acts->xb, acts->hb, weights->w2 + L * dim*config->hidden_dim, config->hidden_dim, dim);

        residual(acts->x, acts->xb, dim);
    }
    rmsnorm(acts->x, acts->x, weights->rms_final_weight, dim);

    matmul(acts->logits, acts->x, weights->wcls, dim, config->vocab_size);
    return acts->logits;
}

//--------------------------------
typedef struct {
    char** vocab;
    float* vocab_scores;
    int vocab_size;
    unsigned int max_token_length;
} Tokenizer;

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size){
    t->vocab_size = vocab_size;
    t->vocab = (char**)malloc(vocab_size*sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size*sizeof(float));
    FILE* file = fopen(tokenizer_path, "r");
    if(file == NULL){
        printf("Error opening tokenizer file\n");
        exit(1);
    }
    if(fread(&t->max_token_length, sizeof(int), 1, file) != 1){
        printf("Error reading max token length\n");
        exit(1);
    }
    int len;
    for(int i = 0;i<vocab_size;i++){
        if(fread(t->vocab_scores + i, sizeof(float), 1, file) != 1){
            printf("Error reading token score\n");
            exit(1);
        }
        if(fread(&len, sizeof(int), 1, file) != 1){
            printf("Error reading token length\n");
            exit(1);
        }
        t->vocab[i] = (char*)malloc(len+1);
        if(fread(t->vocab[i], sizeof(char), len, file) != len){
            printf("Error reading token\n");
            exit(1);
        }
        t->vocab[i][len] = '\0';
    }
    fclose(file);
}

char* decode(Tokenizer* t, int token){
    char* piece = t->vocab[token];
    return piece;
}

//--------------------------------
typedef struct{
    float prob;
    int index;
}ProbIndex;

typedef struct{
    int vocab_size;
    ProbIndex* probindex;
    float temprature;
    float topp;
} Sampler;

int sample_argmax(float* probs, int n){
    float max = probs[0];
    int argmax = 0;
    for(int i = 1;i<n;i++){
        if(probs[i]>max){
            max = probs[i];
            argmax = i;
        }
    }
    return argmax;
}

void build_sampler(Sampler* s, int vocab_size, float temprature, float topp){
    s->vocab_size = vocab_size;
    s->temprature = temprature;
    s->topp = topp;
    s->probindex = (ProbIndex*)malloc(vocab_size*sizeof(ProbIndex));
}

int sample(Sampler* sampler, float* logits){
    for(int q = 0;q<sampler->vocab_size;q++){
        logits[q] /= sampler->temprature;
    }
    softmax(logits, sampler->vocab_size);
    int next = sample_argmax(logits, sampler->vocab_size);
    return next;
}

void generate(Model* model, Tokenizer* tokenizer, Sampler* sampler, int max_tokens){
    int next;

    int* tokens = (int*)malloc(5*sizeof(int));
    tokens[0] = 9038; 
    tokens[1] = 701;
    tokens[2] = 29876;
    tokens[3] = 263;
    tokens[4] = 931;

    int token = tokens[0];
    int pos = 0;

    __clock_t start, end;
    start = clock();

    while(pos < max_tokens){
        float* logits = forward(model, token, pos);

        if(pos < 5){
            next = tokens[pos];
        }
        else{
            next = sample(sampler, logits);
        }

        pos++;
        token = next;
        char* piece = decode(tokenizer, token);
        printf("%s ", piece);
        fflush(stdout);
    }
    printf("\n");
    
    end = clock();
    double time_taken = ((double)end - start)/CLOCKS_PER_SEC;
    double per_token = time_taken/max_tokens;
    printf(" A Token took: %f secs\n", per_token);

}

int main(){
    Model model;
    
    read_checkpoint("stories110M.bin", &model.config, &model.weights, &model.fd, &model.data, &model.size);
    Config config = model.config;
    printf("[Llama2 Config]\n");
    printf("n_embd: %d\n", config.n_embd);
    printf("n_layers: %d\n", config.n_layers);
    printf("n_heads: %d\n", config.n_heads);
    printf("n_kv_heads: %d\n", config.n_kv_heads);
    printf("vocab_size: %d\n", config.vocab_size);
    printf("seq_len: %d\n", config.seq_len);
    printf("hidden_dim: %d\n", config.hidden_dim);

    alloc_activations(&model.activations, &model.config);

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, "tokenizer.bin", config.vocab_size);

    Sampler sampler;
    build_sampler(&sampler, config.vocab_size, 0.75f, 0.0f);

    generate(&model, &tokenizer, &sampler, 64);


    return 0;
}