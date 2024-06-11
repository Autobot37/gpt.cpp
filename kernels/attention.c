
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