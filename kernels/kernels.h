#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

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

//--------
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


//-------------------------------------------
void matmul_forward(float* out, float* inp, float* weight, float* bias, int B, int T, int C, int OC){
    for(int b = 0;b<B;b++){
        for(int t = 0;t<T;t++){
            float* inp_p = inp + b * T * C + t * C;
            float* out_p = out + b * T * OC + t * OC;
            for(int t2=0;t2<OC;t2++){
                float* weight_p = weight + t2* C;
                float val = (bias != NULL) ? bias[t2] : 0.0f;
                for(int i = 0;i<C;i++){
                    val += inp_p[i] * weight_p[i];
                }
                out_p[t2] = val;
            }
        }
    }
}

//-------------------------------------------

void residual_forward(float* out, float* inp, float* skip, int dim){
    for(int i = 0;i<dim;i++){
        out[i] = inp[i] + skip[i];
    }
}
//-------------------------------------------

void attention_forward(float* out, float* preatt, float* att, float* qkv, int B, int T, int C, int NH){
    int hs = C/NH;
    float scale = 1.0 / sqrtf(hs);
    for(int b = 0;b<B;b++){
        for(int t = 0;t<T;t++){
            for(int h = 0;h<NH;h++){
                
                //q @ k 
                float* query = qkv + b * T * 3 * C + t * 3 * C + h * hs;
                float* preatt_p = preatt + b*NH*T*T + h*T*T + t*T;
                float* att_p = att + b*NH*T*T + h*T*T + t*T;


                float maxval = 1e-5f;
                for(int t2=0;t2<=t;t2++){
                    float* key = qkv + b * T * 3 * C + t2 * 3 * C + h*hs + C;
                    float val = 0.0f;
                    for(int i = 0;i<hs;i++){
                        val += query[i] * key[i];
                    }
                    val *= scale;
                    if(val>maxval){
                        maxval = val;
                    }
                    preatt_p[t2] = val;
                }
                //softmax
                float sum = 0.0f;
                for(int t2=0;t2<=t;t2++){
                    float val = expf(preatt_p[t2] - maxval);
                    att_p[t2] = val;
                    sum += val;
                }
                float expinv = (sum==0.0f) ? 0.0f : 1.0f/sum;
                for(int t2=0;t2<T;t2++){
                    if(t2<=t){
                        att_p[t2] *= expinv;
                    }
                    else{
                        att_p[t2] = 0.0f;
                    }
                }   
                //accumulating
                float* out_p = out + b*T*C + t*C + h*hs;
                for(int t2=0;t2<hs;t2++){
                    float val = 0.0f;
                    for(int i = 0;i<T;i++){
                        float value = qkv[b*T*3*C + i*3*C + 2*C + h*hs + t2];
                        val += att_p[i] * value;
                    }
                    out_p[t2] = val;
                }
            }
        }
    }
} 

//---------------------------------------------
void gelu_forward(float* out, float* inp, int dim){
    for(int i = 0;i<dim;i++){
        float val = inp[i];
        out[i] = 0.5f * val * (1.0f + tanhf(0.7978845608f * (val + 0.044715f * val * val * val)));
    }
}

//-------------------------------------------
void softmax_forward(float* out, float* inp, int B, int T, int V){
    for(int b=0;b<B;b++){
        for(int t=0;t<T;t++){
            float* inp_p = inp + b * T * V + t * V;
            float* out_p = out + b * T * V + t * V;
            float max_val = -1e-5f;
            for(int i=0;i<V;i++){
                if(inp_p[i] > max_val){
                    max_val = inp_p[i];
                }
            }
            float sum = 0.0f;
            for(int i=0;i<V;i++){
                out_p[i] = expf(inp_p[i] - max_val);
                sum += out_p[i];
            }
            for(int i=0;i<V;i++){
                out_p[i] = out_p[i] / sum;
            }
        }
    }
}

//-------------------------------------------
void crossentropy_forward(float* losses, float* probs, int * target, int B, int T, int V){
    for(int b = 0;b<B;b++){
        for(int t = 0;t<T;t++){
            float* probs_p = probs + b * T * V+ t * V;
            int ix = target[b * T + t];
            losses[b * T + t] = -logf(probs_p[ix]);
        }
    }
}
//-------------------------------------------
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