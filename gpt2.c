#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// #define M_PI 3.14159

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
    float* ln_1_weight;
    float* ln_1_bias;    
    float* c_attn;
    float* c_proj;
    float* ln_2_weight;
    float* ln_2_bias;
    float* mlp_c_fc;
    float* mlp_c_proj;
    float* ln_f_weight;
    float* ln_f_bias;
    float* lm_head;
}Weights;

void rand_init(float* x, int dim){
    srand(time(NULL));
    for(int i = 0;i<dim;i++){
        x[i] = (float)rand() / RAND_MAX;
    }
}

void rand_init_int(int* x, int dim, int low, int high){
    srand(time(NULL));
    for(int i = 0;i<dim;i++){
        x[i] = rand() % (high - low + 1) + low; 
    }
}

void alloc_weights(Weights* w, Config* config){

    w->wte = (float*)calloc(config->vocab_size * config->n_embd, sizeof(float));
    w->wpe = (float*)calloc(config->block_size * config->n_embd, sizeof(float));
    w->ln_1_weight = (float*)calloc(config->n_layer * config->n_embd, sizeof(float));
    w->ln_1_bias   = (float*)calloc(config->n_layer * config->n_embd, sizeof(float));

    //attn
    w->c_attn = (float*)calloc(config->n_layer * config->n_embd * 3 * config->n_embd, sizeof(float));
    w->c_proj = (float*)calloc(config->n_layer * config->n_embd * config->n_embd, sizeof(float));
    //
    w->ln_2_weight = (float*)calloc(config->n_layer * config->n_embd, sizeof(float));
    w->ln_2_bias   = (float*)calloc(config->n_layer * config->n_embd, sizeof(float));
    w->mlp_c_fc    = (float*)calloc(config->n_layer * config->n_embd * 4 * config->n_embd, sizeof(float));
    w->mlp_c_proj  = (float*)calloc(config->n_layer * 4 * config->n_embd * config->n_embd, sizeof(float));
    //
    w->ln_f_weight = (float*)calloc(config->n_layer * config->n_embd, sizeof(float));
    w->ln_f_bias   = (float*)calloc(config->n_layer * config->n_embd, sizeof(float));
    //
    w->lm_head = (float*)calloc(config->n_embd * config->vocab_size, sizeof(float));

    rand_init(w->wte, config->vocab_size * config->n_embd);
    rand_init(w->wpe, config->block_size * config->n_embd);
    rand_init(w->ln_1_weight, config->n_layer * config->n_embd);
    rand_init(w->ln_1_bias, config->n_layer * config->n_embd);
    rand_init(w->c_attn, config->n_layer * config->n_embd * 3 * config->n_embd);
    rand_init(w->c_proj, config->n_layer * config->n_embd * config->n_embd);
    rand_init(w->ln_2_weight, config->n_layer * config->n_embd);
    rand_init(w->ln_2_bias, config->n_layer * config->n_embd);
    rand_init(w->mlp_c_fc, config->n_layer * config->n_embd * 4 * config->n_embd);
    rand_init(w->mlp_c_proj, config->n_layer * 4 * config->n_embd * config->n_embd);
    rand_init(w->ln_f_weight, config->n_layer * config->n_embd);
    rand_init(w->ln_f_bias, config->n_layer * config->n_embd);
    rand_init(w->lm_head, config->n_embd * config->vocab_size);

}

typedef struct activations {

    int* input;
    float* emb;


}activations;

void print_3d(float *arr, int dim1, int dim2, int dim3) {
    printf("[\n");
    for (int i = 0; i < dim1; i++) {
        printf("  [\n");
        for (int j = 0; j < dim2; j++) {
            printf("    [ ");
            for (int k = 0; k < dim3; k++) {
                printf("%.2f", *(arr + i * dim2 * dim3 + j * dim3 + k));
                if (k < dim3 - 1) {
                    printf(", ");
                }
            }
            printf(" ]\n");
        }
        printf("  ]\n");
    }
    printf("]\n");
}

void print_2d_int(int *arr, int dim1, int dim2) {
    printf("[\n");
    for (int i = 0; i < dim1; i++) {
        printf("  [ ");
        for (int j = 0; j < dim2; j++) {
            printf("%.2d", *(arr + i * dim2 + j));
            if (j < dim2 - 1) {
                printf(", ");
            }
        }
        printf(" ]\n");
    }
    printf("]\n");
}

void print_2d(float *arr, int dim1, int dim2) {
    printf("[\n");
    for (int i = 0; i < dim1; i++) {
        printf("  [ ");
        for (int j = 0; j < dim2; j++) {
            printf("%.2f", *(arr + i * dim2 + j));
            if (j < dim2 - 1) {
                printf(", ");
            }
        }
        printf(" ]\n");
    }
    printf("]\n");
}


void copy(float* dest, float* src, int dim){
    for(int i = 0;i<dim;i++){
        dest[i] = src[i];
    }
}

void add(float* c, float * a,float* b, int dim){
    for(int i=0;i<dim;i++){
        c[i] = a[i]+b[i];
    }
}
//x.shape = batch_size, block_size, n_embd
void layernorm(float* out, float* x, float* weight, float* bias, int batch_size, int block_size, int n_embd){
    for(int bsz = 0;bsz<batch_size;bsz++){
        float* y = &x[bsz*block_size*n_embd];
        float* outy = &out[bsz*block_size*n_embd];
        //now y is block_size,n_embd
        float* mean = (float*)calloc(n_embd, sizeof(float));
        float* std_dev = (float*)calloc(n_embd, sizeof(float));

        for(int j =0;j<n_embd;j++){
            float sum = 0.0f;
            for(int i=0;i<block_size;i++){
                sum += y[i*n_embd + j];
            }
            mean[j] = sum/block_size;
        }

        for(int j = 0;j<n_embd;j++){
            float sum = 0.0f;
            for(int i = 0;i<block_size;i++){
                float diff = y[i*n_embd + j] - mean[j];
                sum += diff * diff;
            }
            std_dev[j] = sqrtf(sum / (block_size-1));
        }

        for(int i = 0;i<block_size;i++){
            for(int j = 0;j<n_embd;j++){
                outy[i * n_embd + j] = ((y[i * n_embd + j] - mean[j]) / std_dev[j]) * weight[j] + bias[j];            }
        }

        free(mean);
        free(std_dev);
    }
}

void matmul(float* dest, float* a, float* b, int n, int m, int k) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            float sum = 0.0;
            for (int l = 0; l < m; ++l) {
                sum += a[i * m + l] * b[l * k + j];
            }
            dest[i * k + j] = sum;
        }
    }
}
//for 2d matrix params-> float* x, original dim1, original dim2 -> returns x (dim2, dim1)
void transpose(float* x, int n, int m) {
    float* temp = (float*)malloc(n * m * sizeof(float));
    if (temp == NULL) {
        return;
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            temp[j * n + i] = x[i * m + j];
        }
    }
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            x[i * n + j] = temp[i * n + j];
        }
    }
    free(temp);
}

void softmax(float* x, int n, int m) {
    for (int i = 0; i < n; ++i) {
        float max_val = x[i * m];
        for (int j = 1; j < m; ++j) {
            if (x[i * m + j] > max_val) {
                max_val = x[i * m + j];
            }
        }
        float sum_exp = 0.0;
        for (int j = 0; j < m; ++j) {
            x[i * m + j] -= max_val;
            sum_exp += exp(x[i * m + j]);
        }
        for (int j = 0; j < m; ++j) {
            x[i * m + j] = exp(x[i * m + j]) / sum_exp;
        }
    }
}

void gelu(float* x, int dim){
    for(int i = 0; i < dim; i++){
        float input = x[i];
        x[i] = 0.5f * input * (1.0f + tanh(sqrt(2.0f / M_PI) * (input + 0.044715f * pow(input, 3.0f))));
    }
}

void build_(Weights* weights, Config* config, char* path){
    FILE* model_file = fopen(path, "rb");
    if(model_file == NULL){printf("error opening file");exit(1);}
    int model_header[8];
    fread(model_header, sizeof(int), 8, model_file);
    if (model_header[0]!=3737){printf("another file");exit(1);}
    if (model_header[1] != 1){printf("another version"); exit(1); }

    int maxT, V, L, NH, C;
    config->block_size = maxT = model_header[2];
    config->vocab_size = V = model_header[3];
    config->n_layer = L = model_header[4];
    config->n_head = NH = model_header[5];
    config->n_embd = C = model_header[6];
    printf("[GPT-2]\n");
    printf("max_seq_len: %d\n", maxT);
    printf("vocab_size: %d\n", V);
    printf("num_layers: %d\n", L);
    printf("num_heads: %d\n", NH);
    printf("channels: %d\n", C);


    alloc_weights(weights,config);
    printf("weights allocated");

    //
    fread(weights->wte, sizeof(float), config->vocab_size * config->n_embd, model_file);
    fread(weights->wpe, sizeof(float), config->block_size * config->n_embd, model_file);

    for (int i = 0; i < L; i++) {
        fread(weights->ln_1_weight + i * config->n_embd, sizeof(float), config->n_embd, model_file);
        fread(weights->ln_1_bias + i * config->n_embd, sizeof(float), config->n_embd, model_file);
        fread(weights->c_attn + i * config->n_embd * 3 * config->n_embd, sizeof(float), 3 * config->n_embd * config->n_embd, model_file);
        fread(weights->c_proj + i * config->n_embd * config->n_embd, sizeof(float), config->n_embd * config->n_embd, model_file);
        fread(weights->ln_2_weight + i * config->n_embd, sizeof(float), config->n_embd, model_file);
        fread(weights->ln_2_bias + i * config->n_embd, sizeof(float), config->n_embd, model_file);
        fread(weights->mlp_c_fc + i * config->n_embd * 4 * config->n_embd, sizeof(float), 4 * config->n_embd * config->n_embd, model_file);
        fread(weights->mlp_c_proj + i * 4 * config->n_embd * config->n_embd, sizeof(float), 4 * config->n_embd * config->n_embd, model_file);
        fread(weights->ln_f_weight + i * config->n_embd, sizeof(float), config->n_embd, model_file);
        fread(weights->ln_f_bias + i * config->n_embd, sizeof(float), config->n_embd, model_file);
    }
    
    fclose(model_file);

}
int main(){

    Weights weights;
    Config config;
    int batch_size = 2;

    activations act;
    char* path = "params.bin";
    build_(&weights, &config, path);
    int head_size = config.n_embd / config.n_head;


    //tril buffer
    float* bias = (float*)calloc(config.block_size * config.block_size , sizeof(float));
    rand_init(bias, config.block_size * config.block_size);
    for(int i = 0;i<config.block_size;i++){
        for(int j = 0;j<i;j++){
            bias[i * config.block_size + j] = 0.0;
        }
    }
    //input.shape  =  batch_size, block_size
    act.input = (int*)calloc(batch_size * config.block_size , sizeof(int));
    rand_init_int(act.input, batch_size * config.block_size,0,config.vocab_size);

    float* wte = (float*)calloc(batch_size * config.block_size * config.n_embd, sizeof(float));
    for(int i = 0;i<batch_size;i++){
        for(int j = 0;j<config.block_size;j++){
            copy(&wte[i * config.block_size * config.n_embd + j * config.n_embd] , &weights.wte[act.input[i * config.block_size + j]], config.n_embd);
        }
    };

    float* wpe = (float*)calloc(batch_size * config.block_size * config.n_embd, sizeof(float));
    for(int i = 0;i<batch_size;i++){
        for(int j = 0;j<config.block_size;j++){
            copy(&wpe[i * config.block_size * config.n_embd + j * config.n_embd] , &weights.wpe[j], config.n_embd);
        }
    };

    float* x = (float*)calloc(batch_size * config.block_size * config.n_embd, sizeof(float));
    add(x,wte,wpe,batch_size * config.block_size * config.n_embd);

    //attention blocks
    for(int layer = 0;layer<config.n_layer;layer++){
        float* ln_1x = (float*)calloc(batch_size * config.block_size * config.n_embd, sizeof(float));
        float* ln_w1 = &weights.ln_1_weight[layer * config.n_embd];
        float* ln_b1 = &weights.ln_1_bias[layer * config.n_embd];
        layernorm(ln_1x, x, ln_w1, ln_b1, batch_size, config.block_size, config.n_embd);

        //c_attn(x): block_size,block_size , n_embd @ n_embd , 3*n_embd -> batch_size, block_size, 3 * n_embd
        float* qkv = (float*)calloc(batch_size * config.block_size * 3 * config.n_embd, sizeof(float));
        for(int bsz = 0;bsz<batch_size;bsz++){
            float* a = &ln_1x[bsz * config.block_size * config.n_embd];
            float* b = &weights.c_attn[layer * config.n_embd * 3 * config.n_embd];
            float* qkvout = &qkv[bsz * config.block_size * 3 * config.n_embd];
            //y->block_size,n_embd @ w->c_attn->n_embd , 3 * n_embd
            matmul(qkvout, a, b, config.block_size, config.n_embd, 3 * config.n_embd);
        }

        float* q = (float*)calloc(batch_size * config.block_size * config.n_embd, sizeof(float));
        float* k = (float*)calloc(batch_size * config.block_size * config.n_embd, sizeof(float));
        float* v = (float*)calloc(batch_size * config.block_size * config.n_embd, sizeof(float));
        for(int bsz = 0;bsz<batch_size;bsz++){
            for(int blk = 0;blk<config.block_size;blk++){
                copy(&q[bsz * config.block_size * config.n_embd + blk * config.n_embd], &qkv[bsz * config.block_size * 3 * config.n_embd + blk * 3 * config.n_embd + 0 * config.n_embd], config.n_embd);
                copy(&k[bsz * config.block_size * config.n_embd + blk * config.n_embd], &qkv[bsz * config.block_size * 3 * config.n_embd + blk * 3 * config.n_embd + 1 * config.n_embd], config.n_embd);
                copy(&v[bsz * config.block_size * config.n_embd + blk * config.n_embd], &qkv[bsz * config.block_size * 3 * config.n_embd + blk * 3 * config.n_embd + 2 * config.n_embd], config.n_embd);
            }
        }
        //splitting completed;
        float* att = (float*)calloc(batch_size * config.n_head * config.block_size * config.block_size, sizeof(float));

        for(int bsz = 0;bsz<batch_size;bsz++){
            for(int head = 0;head < config.n_head;head++){
                float* out = &att[bsz * config.n_head * config.block_size * config.block_size + head * config.block_size * config.block_size];
                float* a   = &q[bsz * config.n_head * config.block_size * head_size + head * config.block_size * head_size];
                float* b   = &k[bsz * config.n_head * config.block_size * head_size + head * config.block_size * head_size];
                transpose(b, config.block_size, head_size);
                matmul(out, a, b, config.block_size, head_size, config.block_size);
            }
        }
        for(int at = 0;at<batch_size * config.n_head * config.block_size * config.block_size;at++){
            att[at] *= (1.0 / sqrtf(head_size));
        }
        //att completed
        for(int bsz = 0;bsz<batch_size;bsz++){
            for(int head = 0;head<config.n_head;head++){
                float* y = &att[bsz * config.n_head * config.block_size * config.block_size + head * config.block_size * config.block_size];
                for(int _i = 0;_i<config.block_size;_i++){
                    for(int _j = 0;_j<config.block_size;_j++){
                        if(bias[_i*config.block_size + _j]==0.0){
                            y[_i*config.block_size + _j] = -1.0 * __LONG_LONG_MAX__;
                        }
                    }
                }
            }
        }
        //trilled
        for(int bsz = 0;bsz<batch_size;bsz++){
            for(int head=0;head<config.n_head;head++){
                float* y = &att[bsz * config.n_head * config.block_size * config.block_size + head * config.block_size * config.block_size];
                softmax(y, config.block_size, config.block_size);
            }
        }
        //softmaxed
        float* y = (float*)calloc(batch_size * config.n_head * config.block_size * head_size, sizeof(float));
        for(int bsz = 0;bsz<batch_size;bsz++){
            for(int head=0;head<config.n_head;head++){
                float* out = &y[bsz * config.n_head * config.block_size * head_size + head * config.block_size * head_size];
                float* a =   &att[bsz * config.n_head * config.block_size * config.block_size + head * config.block_size * config.block_size];
                float* b =   &v[bsz * config.n_head * config.block_size * head_size + head * config.block_size * head_size];
                matmul(out, a, b, config.block_size, config.block_size, head_size);
            }
        }
        //did att @ v
        ////now doing bsz,n_heads,block_size,head_size to bsz,block_size, n_heads,head_size
        float* temp = (float*)malloc(batch_size * config.block_size * config.n_head * head_size * sizeof(float));

        // Perform the transpose operation
        for (int bsz = 0; bsz < batch_size; ++bsz) {
            for (int blk = 0; blk < config.block_size; ++blk) {
                for (int nh = 0; nh < config.n_head; ++nh) {
                    for (int hs = 0; hs < head_size; ++hs) {
                        int index_input = bsz * config.n_head * config.block_size * head_size +
                                        nh * config.block_size * head_size +
                                        blk * head_size +
                                        hs;
                        int index_output = bsz * config.block_size * config.n_head * head_size +
                                        blk * config.n_head * head_size +
                                        nh * head_size +
                                        hs;
                        temp[index_output] = y[index_input];
                    }
                }
            }
        }
        for (int i = 0; i < batch_size * config.block_size * config.n_head * head_size; ++i) {
            y[i] = temp[i];
        }
        free(temp);

        ////ok ctgs

        float* attout = (float*)calloc(batch_size * config.block_size * config.n_embd, sizeof(float));
        for(int bsz = 0;bsz<batch_size;bsz++){
            float* y = &attout[bsz * config.block_size * config.n_embd];
            float* a = &y[bsz * config.block_size * config.n_embd];
            float* b = &weights.c_proj[layer * config.n_embd * config.n_embd];
            matmul(y, a , b, config.block_size, config.n_embd, config.n_embd);
        }

        //////attention completed
        add(x, attout, x, batch_size * config.block_size * config.n_embd);

        ///ln_2
        float* ln_2x = (float*)calloc(batch_size * config.block_size * config.n_embd, sizeof(float));
        float* ln_w2 = &weights.ln_2_weight[layer * config.n_embd];
        float* ln_b2 = &weights.ln_2_bias[layer * config.n_embd];
        layernorm(ln_2x, x, ln_w2, ln_b2, batch_size, config.block_size, config.n_embd);

        //mlping ln_2x
        float* mlpout1 = (float*)calloc(batch_size * config.block_size * 4 * config.n_embd, sizeof(float));
        for(int bsz=0;bsz<batch_size;bsz++){
            float* a   = &ln_2x[bsz * config.block_size * config.n_embd];
            float* b   = &weights.mlp_c_fc[layer * config.n_embd * 4 * config.n_embd];
            float* out = &mlpout1[bsz * config.block_size * 4 * config.n_embd];
            matmul(out, a, b, config.block_size, config.n_embd, 4 * config.n_embd);   
        } 
        //doing gelu
        gelu(mlpout1, batch_size * config.block_size * 4 * config.n_embd);

        float* mlpout = (float*)calloc(batch_size * config.block_size * config.n_embd, sizeof(float));
        for(int bsz=0;bsz<batch_size;bsz++){
            float* a   = &mlpout1[bsz * config.block_size * 4 * config.n_embd];
            float* b   = &weights.mlp_c_proj[layer * 4 * config.n_embd  * config.n_embd];
            float* out = &mlpout[bsz * config.block_size * config.n_embd];
            matmul(out, a, b, config.block_size, 4 * config.n_embd, config.n_embd);   
        } 

        add(x, mlpout, x, batch_size * config.block_size * config.n_embd);
    }


    float* out = (float*)calloc(batch_size * config.block_size * config.vocab_size, sizeof(float));
    for(int bsz = 0;bsz<batch_size;bsz++){
        float* y = &out[bsz * config.block_size * config.vocab_size];
        float* a = &x[bsz * config.block_size * config.n_embd];
        float* b = weights.lm_head;
        matmul(y, a, b, config.block_size, config.n_embd, config.vocab_size);
    }

    print_3d(out, batch_size, config.block_size, config.vocab_size);

    printf("wet pants\n");


    return 0;
}