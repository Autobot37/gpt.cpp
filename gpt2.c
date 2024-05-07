#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define M_PI 3.14159

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
}activations;

void fill_act_sizes(int* act_sizes, Config config, int B){
    int V = config.vocab_size;
    int C = config.n_embd;
    int T  = config.block_size;
    int L = config.n_layer;
    act_sizes[0] = B * T * C;
    act_sizes[1] = B * T * C;
    act_sizes[2] = B * T;
    act_sizes[3] = B * T;
    act_sizes[4] = B * T * 3 * C;
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
        &act->encoded, &act->ln_1, &act->ln_1_mean, &act->ln_1_rstd, &act->qkv
    };
    float* iter = act_memory;
    for(int i = 0;i<NUM_ACTIVATIONS;i++){
        *ptrs[i] = iter;
        iter += act_sizes[i];
    }

    return act_memory;
}


//x = wte(x) + wpe(x)
//x.shape == B,T
//out.shape == B,T,C,= wpe(B,T,C) + wte(B,T,C)
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
//dl/dwpe = dl/dout * dout/dwpe, o
void encoder_backward(float* d_wpe, float* d_wte, int* inp, float* d_out, int B, int T, int C){
    for(int b=0;b<B;b++){
        for(int t=0;t<T;t++){
            float* d_out_p = d_out + b * T * C + t * C;
            float* d_wte_p = d_wte + inp[b * T + t] * C;
            float* d_wpe_p = d_wpe + t * C;
            for(int i = 0;i<C;i++){
                d_wte_p[i] += d_out[i];
                d_wpe_p[i] += d_out[i];
            }
        }
    }
}

void layernorm_forward(float* out, float* mean, float* std_dev, float* inp, float* weight, float* bias, int B, int T, int C){
    float eps = 1e-5f;
    for(int b = 0;b<B;b++){
        for(int t = 0;t<T;t++){
            float* inp_p = inp + b*T*C + t*C;
            float* out_p = out + b*T*C + t*C;
                        
            float m = 0.0f;
            for(int i=0;i<C;i++){
                m += inp_p[i];
            }
            m = m/C;
            mean[b*T + t] = m;

            float v = 0.0f;
            for(int i = 0;i<C;i++){
                float diff = inp_p[i] - m;
                v += diff * diff;
            }
            v = v / C;
            float s = 1.0f/sqrtf(v + eps);
            std_dev[b*T + t] = s;

            for(int i = 0;i<C;i++){
                out_p[i] = ((inp_p[i] - m) * s) * weight[i] + bias[i];            
            }
        }    
    }
}
//(x-m) * s , m = sum(x), s = 1/(sum(x-m)^2)/C, 
void layernorm_backward(float* d_inp, float* d_weight, float* d_bias, float* d_out, float* mean, float* std_dev, float* inp, float* weight, float* bias, int B, int T, int C){
    float eps = 1e-5f;
    for(int b = 0;b<B;b++){
        for(int t = 0;t<T;t++){
            float* inp_p = inp + b*T*C + t*C;
            float* out_p = d_out + b*T*C + t*C;
            float* d_inp_p = d_inp + b*T*C + t*C;
            float mean_p = mean[b*T + t];
            float std_dev_p = std_dev[b*T + t];
            for(int i = 0;i<C;i++){
                d_weight[i] += out_p[i] * ((inp_p[i] - mean_p) * std_dev_p);
                d_bias[i] += out_p[i];
                d_inp_p[i] += out_p[i] * weight[i] * std_dev_p;
            }
        }
    }
}

void matmul_forward(float* out, float* inp, float* weight, float* bias, int B, int T, int C){
    for(int b = 0;b<B;b++){
        for(int t = 0;t<T;t++){
            float* inp_p = inp + b*T*C + t*C;
            float* out_p = out + b*T*3*C + t*3*C;
            for(int i = 0;i<3*C;i++){
                float val = (bias != NULL) ? bias[i] : 0.0f;
                float* row = weight + i * C;
                for(int j = 0;j<C;j++){
                    val += inp_p[j] * weight[j];
                }
                out_p[i] = val;
            }
        }
    }
}
void matmul_backward(float* d_inp, float* d_weight, float* d_bias, float* d_out, float* inp, float* weight, float* bias, int B, int T, int C){
    for(int b = 0;b<B;b++){
        for(int t = 0;t<T;t++){
            float* inp_p = inp + b*T*C + t*C;
            float* d_inp_p = d_inp + b*T*C + t*C;
            float* d_out_p = d_out + b*T*3*C + t*3*C;
            for(int i = 0;i<3*C;i++){
                float* d_weight_p = d_weight + i * C;
                float val = d_out_p[i];
                for(int j = 0;j<C;j++){
                    d_weight_p[j] += val * inp_p[j];
                    d_inp_p[j] += val * weight[j];
                }
                if(bias != NULL){
                    d_bias[i] += val;
                }
            }
        }
    }
}

void attention_forward(float* preatt,float* att, float* out, float* inp, int B, int T, int C, int NH){
    int C3 = 3*C;
    int hs = C / NH;
    float scale = 1.0/sqrtf(hs);
    for(int b=0;b<B;b++){
        for(int t = 0;t<T;t++){
            for(int h = 0;h<NH;h++){
                float* q = inp + b*T*C3 + t*C3 + h*hs;//T, NH, C//NH
                float* preatt_p = preatt + b*NH*T*T + h*T*T + t*T;
                float* att_p = att + b*T*NH*T + t*T*T + t*T;
                float maxval = preatt_p[0];
                for(int t2 = 0;t2<=t;t++){
                    float* k = inp + b*T*C3 + C + t2*C3 + h*hs;//T, NH, C//NH
                    float val = 0.0f;
                    for(int i = 0;i<hs;i++){
                        val += q[i] * k[i];
                    }
                    val *= scale;
                    if(val > maxval){
                        maxval = val;
                    }
                    preatt_p[t2] = val;
                }
                float expsum = 0.0f;
                for(int t2 = 0;t2<=t;t2++){
                    float exp = expf(preatt_p[t2] - maxval);
                    expsum += expsum;
                    att_p[t2] = exp;
                }
                float exp_inv = 1.0f/(1e-8 + expsum);
                for(int t2=0;t2<T;t2++){
                    if(t2 <= t)
                    att_p[t2] *= exp_inv;
                    else{
                        att_p[t2] = 0.0f;
                    }
                }
                float* out_p = out + b*T*C + t*C + h*hs;//T, C
                for(int t2=0;t2<=t;t2++){
                    float* v = inp + b*T*C3 + 2*C + t2*C3 + h*hs;//T, C 
                    float att_val = att_p[t2];
                    for(int i = 0;i<hs;i++){
                        out_p[i] += att_val * v[i];
                    }
                }

            }
        }
    }
}

void attention_backward(float* d_inp, float* d_out, float* d_att, float* preatt, float* att, float* inp, int B, int T, int C, int NH){
    int C3 = 3*C;
    int hs = C / NH;
    float scale = 1.0/sqrtf(hs);
    for(int b = 0;b<B;b++){
        for(int t = 0;t<T;t++){
            for(int h = 0;h<NH;h++){
                float* dquery = inp + b*T*C3 + t*C3 + h*hs;
                float* datt = d_att + b*T*NH*T + t*T*T + t*T;
                float* att = att + b*T*NH*T + t*T*T + t*T;

                for(int t2 = 0;t2<=t;t2++){
                    float* v = inp + b*T*C3 + 2*C + t2*C3 + h*hs;
                    float* dval = d_inp + b*T*C3 + 2*C + t2*C3 + h*hs;
                    for(int i = 0;i<hs;i++){
                        datt[t2] += d_out[i] * v[i];
                        dval[i] += att[t2] * d_out[i];
                    }   
                }
                //softmax
                float* dpreatt = preatt + b*NH*T*T + h*T*T + t*T;
                for(int t2 = 0;t2<=t;t2++){
                    for(int t3=0;t3<=t;t3++){
                        float indicator = (t2 == t3) ? 1.0f : 0.0f;
                        float deriv = att[t2] * (indicator - att[t3]);
                        dpreatt[t3] += datt[t2] * deriv;
                    }
                }
                float* query = inp + b*T*C3 + t*C3 + h*hs;
                for(int t2 = 0;t2<=t;t2++){
                    float* key = inp + b*T*C3 + C + t2*C3 + h*hs;
                    float* dkey = d_inp + b*T*C3 + C + t2*C3 + h*hs;
                    for(int i = 0;i<hs;i++){
                        dquery[i] += d_out[i] * key[i] * scale;
                        dkey[i] += d_out[i] * query[i] * scale;
                    }
                }

            }
        }
    }
}
void gelu_forward(float* out, float* inp, int N){
    for(int i = 0; i < N; i++){
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        out[i] = 0.5f * x * (1.0f + tanh(sqrt(2.0f / M_PI) * (x + cube)));
    }
}
void gelu_backward(float* dinp, float* dout, float* inp, int N){
    for(int i = 0;i<N;i++){
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        float tanh_val = tanh(sqrt(2.0f / M_PI) * (x + cube));
        float exp_val = exp(-0.5f * x * x) * (0.5f * (1.0f + tanh_val) * (1.0f + tanh_val) + 0.7978845608f * tanh_val);
        dout[i] = 0.5f * (1.0f + tanh_val) + x * exp_val;
    }
}

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




// void add(float* c, float * a,float* b, int dim){
//     for(int i=0;i<dim;i++){
//         c[i] = a[i]+b[i];
//     }
// }
// //x.shape = batch_size, block_size, n_embd
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

    //tril buffer
    // float* bias = (float*)calloc(config.block_size * config.block_size , sizeof(float));
    // rand_init(bias, config.block_size * config.block_size);
    // for(int i = 0;i<config.block_size;i++){
    //     for(int j = 0;j<i;j++){
    //         bias[i * config.block_size + j] = 0.0;
    //     }
    // }
    // //input.shape  =  batch_size, block_size
    int* input = calloc(B * T , sizeof(int));
    rand_init_int(input, B * T, 0, V);

    encoder_forward(act.encoded, input, weights.wte, weights.wpe, B, T, C);
    for(int l=0; l<L; l++){
        //weights for layer l
        float* ln_l_w = weights.ln_1_w + l * C;
        float* ln_l_b = weights.ln_1_b + l * C;
        float* qkv_w = weights.c_attn_w + l * 3 * C * C;
        float* qkv_b = weights.c_attn_b + l * 3 * C;
        float* proj_w = weights.c_proj_w + l * C * C;
        float* proj_b = weights.c_proj_w + l * C;
        //activations for layer l
        float* ln_l_out = act.ln_1 + l * B * T * C;
        float* ln_l1_mean = act.ln_1_mean + l * B * T;
        float* ln_l1_rstd = act.ln_1_rstd + l * B * T;
        float* qkv_out = act.qkv + l * B * T * 3 * C;
        //float* atty = act.atty + l * B * T * C;

        layernorm_forward(ln_l_out, ln_l1_mean, ln_l1_rstd, act.encoded, ln_l_w, ln_l_b, B, T, C);
        //matmul_forward(qkv_out, ln_l_out, qkv_w, qkv_b, B, T, C);
        //attention_forward(atty, qkv_out, B, T, C, NH);
        //layernorm_forward();






    }
    printf("wet pants\n");

    return 0;
}