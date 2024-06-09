void attention(float* _att, float* _q, float* _key_cache, float* _value_cache, float* _xb, int pos, int dim, int head_size, int kv_dim, int kv_mul, int n_heads, int seq_len, int n_kv_heads, int n_embd, int L){
    int skip = L * seq_len * kv_dim;
    for(int h =0;h < n_heads; h++){

        float* q = _q + h*head_size;
        float* att = _att + h* seq_len;
        
        for(int t= 0;t<=pos;t++){
            float* k = _key_cache + skip + t*kv_dim + (h/kv_mul)*head_size;

            float score = 0.0f;
            for(int j = 0;j<head_size;j++){
                score += q[j]*k[j];
            }
            score /= sqrtf(head_size);
            att[t] = score;
        }

        softmax(att, pos + 1);

        float* xb = _xb + h*head_size;
        memset(xb, 0, head_size*sizeof(float));
        for(int t = 0;t<=pos;t++){
            float* v = _value_cache + skip + t*kv_dim + (h/kv_mul)*head_size;
            float score = att[t];
            for(int j = 0;j<head_size;j++){
                xb[j] += score*v[j];
            }
        }

    }
}