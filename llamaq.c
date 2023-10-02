#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <omp.h>

#define max(a,b) a>b?a:b

typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;//max_seq_len although it is 1 it is still used for pos emb and cache storage
} Config;

typedef struct {
    float* tok_embedding_table;
    float* rms_ffn_weight;
    float* rms_attn_weight;
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

void memmap_weights(Weights* w,Config* p,float* ptr){
    int vocab_size = p->vocab_size;
    int dim = p->dim;
    int n_layers = p->n_layers;
    int n_heads = p->n_heads;
    int head_size = dim/n_heads;

    w->tok_embedding_table = ptr;
    ptr += vocab_size * dim;
    w->rms_attn_weight = ptr;
    ptr += n_layers * dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers  * dim;
    w->wq = ptr;
    ptr += n_layers * dim * (n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * dim * (n_heads * head_size);
    w->w1 = ptr;
    ptr += n_layers* dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * dim * p->hidden_dim;
    w->w3 = ptr;
    ptr += n_layers * dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += dim;
    /////take care while encoding weights
    w->wcls = ptr;
    ptr += vocab_size*dim;
}

typedef struct {
    float* x;
    float* xb;
    float* xb2;
    float* q;
    float* k;
    float* v;
    float* key_cache;
    float* value_cache;
    float* att;
    float* hb;
    float* hb2;
    float* logits;
} Runstate;

void calloc_runstate(Runstate* s, Config* p){
    int kv_dim = (p->dim*p->n_kv_heads)/p->n_heads;
    s->x = (float*)calloc(p->dim, sizeof(float));
    s->xb = (float*)calloc(p->dim, sizeof(float));
    s->xb2 = (float*)calloc(p->dim, sizeof(float));
    s->hb = (float*)calloc(p->hidden_dim, sizeof(float));
    s->hb2 = (float*)calloc(p->hidden_dim, sizeof(float));
    s->q = (float*)calloc(p->dim, sizeof(float));
    s->k = (float*)calloc(kv_dim, sizeof(float));
    s->v = (float*)calloc(kv_dim, sizeof(float));  
    s->att = (float*)calloc(p->n_heads*p->seq_len, sizeof(float));
    s->logits = (float*)calloc(p->vocab_size,sizeof(float));
    s->key_cache = (float*)calloc(p->n_layers*p->seq_len*kv_dim, sizeof(float));
    s->value_cache = (float*)calloc(p->n_layers*p->seq_len*kv_dim, sizeof(float));
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->k || !s->v || !s->att || !s->logits || !s->key_cache
     || !s->value_cache) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_runstate(Runstate *s){
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

typedef struct {
    Config config;
    Weights weights;
    Runstate runstate;
} Transformer;

void RMSNorm(float* out, float* in, float* w, int dim){
    float sm = 1e-6;
    for(int i = 0;i<dim;i++){
        sm += in[i]*in[i];
    }
    sm /= dim;
    sm = 1.0f/sqrtf(sm);
    for(int i = 0;i<dim;i++){
        out[i] = w[i]*(sm*in[i]);
    }
}

void matmul(float* out, float* in, float* w, int n, int d){
    //  w d,n @ in n, -> out d,
    int i;
    #pragma omp parallel for private(i)
    for(i = 0;i<d;i++){
        float val = 0.0f;
        for(int j = 0;j<n;j++){
            val += in[j]*w[i*n+j];
        }
        out[i] = val;
    } 
}

void softmax(float*x,int d){
    float max_val = x[0];
    for(int i = 1;i<d;i++)max_val = max(x[i],max_val);
    float sum = 0.0f;
    for(int i = 0;i<d;i++){
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for(int i = 0;i<d;i++){
        x[i] /= sum;
    }
}

float* forward(Transformer* transformer,int token, int pos){
    Config* p = &transformer->config;
    Weights* w = &transformer->weights;
    Runstate* s = &transformer->runstate;
    printf("ok");
    float* x = s->x;
    int dim = p->dim;
    int head_size = dim/p->n_heads;
    int kv_dim = (p->n_kv_heads*dim)/p->n_heads;
    int rep = p->n_heads/p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    printf("ok");

    float* content_row = w->tok_embedding_table + token * dim;
    memcpy(x, content_row, dim*sizeof(*content_row));

    for(int l = 0; l< p->n_layers;l++){
        RMSNorm(s->xb,x,w->rms_attn_weight + l*dim, dim);
        matmul(s->q,s->xb,w->wq+l*dim*dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);

        for(int i = 0;i<dim/2;i++){
            int head_dim = i%head_size;
            float freq = 1.0f/powf(10000.0f,head_dim/(float)head_size);
            float val = pos*freq;
            float fcos = cosf(val);
            float fsin = sinf(val);
            int rotn = i<kv_dim?2:1;
            for(int r= 0;r<rotn;r++){
                float* vec = r==0?s->q:s->k;
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i] = v0*fcos - v1*fsin;
                vec[i+1] = v0*fsin + v1*fcos;
            }
        }

        float* key_row = s->key_cache + l*p->seq_len*kv_dim + pos*kv_dim; 
        float* value_row = s->value_cache + l*p->seq_len*kv_dim + pos*kv_dim;
        memcpy(key_row,s->k,kv_dim*(sizeof(*key_row)));
        memcpy(value_row,s->v,kv_dim*(sizeof(*value_row)));

        for(int h = 0;h<p->n_heads;h++){
            float* q = s->q + h*head_size;
            float* att = s->att + h*p->seq_len;
            for(int t = 0;t<=pos;t++){
                float* k = s->key_cache + l*p->seq_len*kv_dim + t*kv_dim + (h/rep)*head_size;
                float score = 0.0f;
                for(int i = 0;i<head_size;i++){
                    score += q[i]*k[i];
                }
                score /= sqrtf(head_size);
                att[t] = score;
            }
        
            softmax(s->att,pos+1);
            float* xb = s->xb + h*head_size;
            memset(xb,0,head_size*sizeof(float));
            for(int t=0;t<=pos;t++){
                float* v = s->key_cache + l*p->seq_len*kv_dim + t*kv_dim + (h/rep)*head_size;
                float a = att[t];
                for(int i=0;i<head_size;i++){
                    xb[i] = a*v[i];
                }
            }
        }
        matmul(s->xb2, s->xb, w->wo+l*dim*dim,dim,dim);
        //and here matrix gymnastics ends;
        for(int i = 0;i<dim;i++){
            x[i] +=  s->xb2[i];
        }
        RMSNorm(s->xb, x, w->rms_ffn_weight + l*dim,dim);
        matmul(s->hb,s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        matmul(s->hb2,s->xb,w->w3+l*dim*hidden_dim, dim, hidden_dim);

        for(int i = 0;i<hidden_dim;i++){
            float val = s->hb[i];
            val *= (1.0f/1.0f+exp(-val));
            val *= s->hb2[i];
            s->hb[i] = val;
        }
        matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);
        for(int i = 0;i<dim;i++){
            x[i] += s-> xb[i];
        }
    }
    RMSNorm(x,x,w->rms_final_weight,dim);
    matmul(s->logits,x,w->wcls,p->dim,p->vocab_size);
    return s->logits;
}

void run(Config* p){
    Transformer model;
    model.config.dim = 2048;
    model.config.hidden_dim = 4096;
    model.config.n_heads = 32;
    model.config.n_kv_heads = 4;
    model.config.n_layers = 32;
    model.config.seq_len = 32000;
    model.config.vocab_size = 8092*4;

    int vocab_size = model.config.vocab_size;
    int dim = model.config.dim;
    int hidden_dim = model.config.hidden_dim;
    int n_heads = model.config.n_heads;
    int n_kv_heads = model.config.n_kv_heads;
    int n_layers = model.config.n_layers;
    int seq_len = model.config.seq_len;
    int head_size = model.config.dim/model.config.n_heads;

    int total_memory = 0;
    total_memory += vocab_size * dim ; // tok_embedding_table
    total_memory += 2 * n_layers * dim ;   // rms_ffn_weight and rms_attn_weight
    total_memory += n_layers * dim * n_heads * head_size ; // wq,
    total_memory += 2* n_layers * dim * n_kv_heads * head_size ; // wk, wv for kv_heads
    total_memory += n_layers * dim * n_heads * head_size; // wo 
    total_memory += 3 * n_layers * hidden_dim * dim ;
    total_memory += dim ; // rms_final_weight
    total_memory += vocab_size * dim ; // wcls

    calloc_runstate(&model.runstate,&model.config);

    float* mem = (float*)calloc(total_memory, sizeof(float));
    if( mem == NULL)printf("no mem");
    memmap_weights(&model.weights,&model.config,mem);
    printf("total_size:%.4fmb\n",(float)total_memory/(float)(1024.0*1024.0));
    printf("paranoia\n");

    int token = 31;
    int pos = 1;

    clock_t start = clock();
    float* logits = forward(&model,token,pos);
    clock_t endlogits = clock();
    clock_t startp = clock();
    // for(int i = 0;i<vocab_size;i++){
    //     printf("%f",logits[i]);
    // }
    clock_t endp = clock();

    float logtime = (double)(endlogits - start) / CLOCKS_PER_SEC;
    printf("\n");

    printf("Forward takes:%.4f seconds",logtime);
    printf("\n");
    free_runstate(&model.runstate);
    free(mem);
}

int main(){

    Config p;
    p.dim = 4;
    p.hidden_dim = 2;
    p.n_heads = 4;
    p.n_kv_heads = 2;
    p.n_layers = 2;
    p.seq_len = 4;
    p.vocab_size = 8092;

    run(&p);
    
    return 0;
}