
void embed(float* x, float* wte, float* wpe, int token, int pos, int C){
    for(int i = 0;i<C;i++){
        x[i] = wte[token*C + i] + wpe[pos*C + i];
    }
}
//-----------------------------------------------------------------------------------------------
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
//-----------------------------------------------------------------------------------------------

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

//-----------------------------------------------------------------------------------------------

void residual(float* out, float* in, int C){
    for(int i = 0;i<C;i++){
        out[i] += in[i];
    }
}

//-----------------------------------------------------------------------------------------------


void gelu(float* x, int C){
    for(int i = 0;i<C;i++){
        float u = x[i];
        x[i] = 0.5 * u * (1 + tanh(sqrt(2.0/M_PI) * (u + 0.044715 * u * u * u)));
    }
}

//-----------------------------------------------------------------------------------------------

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

//-----------------------------------------------------------------------------------------------

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

//-----------------------------------------------------------------------------------------------
