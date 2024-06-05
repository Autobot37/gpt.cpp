#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cblas.h>
#include <cblas-openblas.h>
#include <cfloat>

void attention_forward(float* out, float* preatt, float* att, float* qkv, int B, int T, int C, int NH){
    int hs = C/NH;
    float scale = 1.0 / sqrtf(hs);
    #pragma omp parallel for collapse(3) schedule(dynamic)
    for(int b = 0;b<B;b++){
        for(int t = 0;t<T;t++){
            for(int h = 0;h<NH;h++){
                
                //q @ k 
                float* query = qkv + b * T * 3 * C + t * 3 * C + h * hs;
                float* preatt_p = preatt + b*NH*T*T + h*T*T + t*T;
                float* att_p = att + b*NH*T*T + h*T*T + t*T;

                float maxval = 1e-5f;
                #pragma omp simd reduction(max:maxval)
                for(int t2=0;t2<=t;t2++){
                    float* key = qkv + b * T * 3 * C + t2 * 3 * C + h*hs + C;
                    float val = 0.0f;
                    #pragma omp simd reduction(+:val)
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
                #pragma omp simd reduction(+:sum)
                for(int t2=0;t2<=t;t2++){
                    float val = expf(preatt_p[t2] - maxval);
                    att_p[t2] = val;
                    sum += val;
                }
                float expinv = (sum==0.0f) ? 0.0f : 1.0f/sum;
                #pragma omp simd
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
                #pragma omp simd
                for(int t2=0;t2<hs;t2++){
                    float val = 0.0f;
                    #pragma omp simd reduction(+:val)
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

void attention_forward_blas(float* out, float* preatt, float* att, float* qkv, int B, int T, int C, int NH) {
    int hs = C / NH;
    float scale = 1.0 / sqrtf(hs);

    #pragma omp parallel for collapse(3) schedule(dynamic)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                // Compute the index offsets
                int qkv_offset = b * T * 3 * C + t * 3 * C + h * hs;
                int preatt_offset = b * NH * T * T + h * T * T + t * T;
                int att_offset = b * NH * T * T + h * T * T + t * T;
                int out_offset = b * T * C + t * C + h * hs;

                // Compute q @ k
                cblas_sgemv(CblasRowMajor, CblasNoTrans, T, hs, scale, qkv + qkv_offset, C, qkv + qkv_offset + C, 1, 0.0f, preatt + preatt_offset, 1);

                // Compute softmax
                float maxval = -FLT_MAX;
                float sum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    maxval = fmaxf(maxval, preatt[preatt_offset + t2]);
                    att[att_offset + t2] = expf(preatt[preatt_offset + t2] - maxval);
                    sum += att[att_offset + t2];
                }
                float expinv = (sum == 0.0f) ? 0.0f : 1.0f / sum;
                cblas_sscal(T, expinv, att + att_offset, 1);

                // Accumulate
                cblas_sgemv(CblasRowMajor, CblasTrans, T, hs, 1.0f, qkv + qkv_offset + 2 * C, C, att + att_offset, 1, 0.0f, out + out_offset, 1);
            }
        }
    }
}


int main(){

    int mul = 4;
    int B = 4*mul;
    int T = 128*mul;
    int C = 128*mul;
    int OC = 128*2*mul;
    int NH = 8*mul;

    float *preatt, *att, *qkv, *out;
    preatt = (float*)malloc(B*NH*T*T*sizeof(float));
    att = (float*)malloc(B*NH*T*T*sizeof(float));
    qkv = (float*)malloc(B*T*3*C*sizeof(float));
    out = (float*)malloc(B*T*C*sizeof(float));

   
    attention_forward(out, preatt, att, qkv, B, T, C, NH);

    attention_forward_blas(out, preatt, att, qkv, B,T,C,NH);
     
    return 0;
}