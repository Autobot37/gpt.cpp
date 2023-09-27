#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <random>
#include <curand_kernel.h>

using namespace std;

#define max(a,b) (a>b)?a:b
#define inf INFINITY;


__global__ void init_random_matrix_kernel(float* mat, int rows, int cols, unsigned int seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < rows && j < cols) {
        curandState state;
        curand_init(seed, i * cols + j, 0, &state);
        mat[i * cols + j] = curand_uniform(&state);
    }
}

void init_random_matrix(float* mat, int rows, int cols) {
    int threadsPerBlock = 256; // You can adjust this based on your GPU's capabilities
    dim3 blocksPerGrid((rows + threadsPerBlock - 1) / threadsPerBlock, (cols + threadsPerBlock - 1) / threadsPerBlock);
    unsigned int seed = time(NULL);

    init_random_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(mat, rows, cols, seed);
    cudaDeviceSynchronize(); 
}


void print_mat(float* mat, int rows, int cols) {
	printf("[%d, %d]\n",rows,cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.3f\t", mat[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

float gelu(float x) {
    return 0.5 * x * (1.0 + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
}

__global__ void add_kernel(float* a, float* b, int sz) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < sz) {
        a[idx] += b[idx];
    }
}

void add(float* a, float* b, int sz) {
    float* d_a;
    float* d_b;

    cudaMalloc((void**)&d_a, sz * sizeof(float));
    cudaMalloc((void**)&d_b, sz * sizeof(float));

    cudaMemcpy(d_a, a, sz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sz * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 256; // Adjust block size as needed
    int num_blocks = (sz + block_size - 1) / block_size;

    add_kernel<<<num_blocks, block_size>>>(d_a, d_b, sz);
    cudaMemcpy(a, d_a, sz * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
}

__global__ void add2d_kernel(float* a, float* b, int x, int y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < x && col < y) {
        a[row * y + col] += b[row * y + col] + col;
    }
}

void add2d(float* a, float* b, int x, int y) {
    float* d_a;
    float* d_b; 
    cudaMalloc((void**)&d_a, x * y * sizeof(float));
    cudaMalloc((void**)&d_b, x * y * sizeof(float));
    cudaMemcpy(d_a, a, x * y * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, x * y * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((x + blockDim.x - 1) / blockDim.x, (y + blockDim.y - 1) / blockDim.y);

    add2d_kernel<<<gridDim, blockDim>>>(d_a, d_b, x, y);
    cudaMemcpy(a, d_a, x * y * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
}


__global__ void c_matmul(float*c,float*a,float*b,int m,int n,int p){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row<m && col<p){
        float sum = 0.0f;
        for(int i = 0;i<n;i++){
            sum += a[n*row+i] * b[p*i + col];
        }
        c[row*p+col] = sum;
    }
}


void matmul(float* out, float* in, float* w, int m, int n, int p) {
	float*da,*db,*dc;
	cudaMalloc((void**)&da,m*n*sizeof(float));
  cudaMalloc((void**)&db,n*p*sizeof(float));
  cudaMalloc((void**)&dc,m*p*sizeof(float));

	cudaMemcpy(da,in, m*n*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(db,w,n*p*sizeof(float), cudaMemcpyHostToDevice);

	dim3 blockDim(256,256);
	dim3 gridDim((p+blockDim.x-1)/blockDim.x, (m+blockDim.y-1)/blockDim.y);

	c_matmul<<<gridDim, blockDim>>>(da,db,dc,m,n,p);
	cudaMemcpy(out,dc,m*p*sizeof(float),cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);

}

void LayerNorm(float* x, float* weight, float* bias, int bsz, int n_embd) {
    float eps = 1e-5;
    for (int i = 0; i < bsz; i++) {
        float mean = 0.0f;
        float var = 0.0f;
        for (int j = 0; j < n_embd; j++) {
            mean += x[i*n_embd+j];
        }
        mean /= n_embd;
        for (int j = 0; j < n_embd; j++) {
            float diff = x[i*n_embd+j] - mean;
            var += diff * diff;
        }
        var /= n_embd;
        for (int j = 0; j < n_embd; j++) {
            float x_hat = (x[i*n_embd+j] - mean) / sqrt(var + eps);
            x[i*n_embd+j] = x_hat * weight[j] + bias[j];
        }
    }
}



typedef struct {
	int block_size;
	int vocab_size;
	int n_layer;
	int n_head;
	int n_embd;
}Config;

typedef struct {
	float* token_embedding_table; //vocab_size,n_embd;
	float* pos_embedding_table;//block_size. n_embd;
	//init Layernorm
	float* layernorm_weight;//n_layer, n_embd
	float* layernorm_bias;//n_layer, n_embd
	float* layernorm2_weight;//n_layer, n_embd
	float* layernorm2_bias;//n_layer, n_embd
	//Attention
	float* c_attn;//n_layer,n_embd,3*n_embd
	float* c_proj;//n_layer,n_embd,n_embd
	//final layernorm
	float* fin_layernorm_weight;//n_embd
	float* fin_layernorm_bias;//n_embd
	//MLP
	float* mlp_c_fc;//n_layer, n_embd,4*n_embd
	float* mlp_c_proj;//n_layer,4*n_embd,n_embd
	//
	float* lm_head_w;//n_embd,vocab_size
}Weights;

void init_weights(Weights* w,Config* p, float* ptr){
	int n_layers = p->n_layer;

	w->token_embedding_table = ptr;
	ptr += p->vocab_size * p->n_embd;
	init_random_matrix(w->token_embedding_table,p->vocab_size,p->n_embd);
	w->pos_embedding_table = ptr;
	ptr += p->block_size * p->n_embd;
	init_random_matrix(w->pos_embedding_table, p->block_size, p->n_embd);
	w->layernorm_weight = ptr;
	ptr += p->n_layer * p->n_embd;
	init_random_matrix(w->layernorm_weight, n_layers, p->n_embd);

	w->layernorm_bias = ptr;
	ptr += p->n_layer * p->n_embd;
    init_random_matrix(w->layernorm_bias, n_layers, p->n_embd);

	w->c_attn = ptr;
	ptr += p->n_layer * p->n_embd * 3*p->n_embd;
    init_random_matrix(w->c_attn, n_layers, 3 * p->n_embd * p->n_embd);

	w->c_proj = ptr;
	ptr += p->n_layer * p->n_embd * p->n_embd;
    init_random_matrix(w->c_proj, n_layers, p->n_embd * p->n_embd);

	w->layernorm2_weight = ptr;
	ptr += p->n_layer * p->n_embd;
	init_random_matrix(w->layernorm2_weight, n_layers, p->n_embd);

	w->layernorm2_bias = ptr;
	ptr += p->n_layer * p->n_embd;
    init_random_matrix(w->layernorm2_bias, n_layers, p->n_embd);


	w->mlp_c_fc = ptr;
	ptr += p->n_layer * p->n_embd * 4*p->n_embd;
    init_random_matrix(w->mlp_c_fc, n_layers, 4 * p->n_embd * p->n_embd);

	w->mlp_c_proj = ptr;
	ptr += p->n_layer * 4*p->n_embd * p->n_embd;
	init_random_matrix(w->mlp_c_proj, n_layers, 4 * p->n_embd * p->n_embd);

	//layers end
	w->fin_layernorm_weight = ptr;
	ptr += p->n_embd;
    init_random_matrix(w->fin_layernorm_weight, 1, p->n_embd);

	w->fin_layernorm_bias = ptr;
	ptr += p->n_embd;
	init_random_matrix(w->fin_layernorm_bias, 1, p->n_embd);

	w->lm_head_w = ptr;
	ptr += p->n_embd * p->vocab_size;
    init_random_matrix(w->lm_head_w, p->n_embd, p->vocab_size);
}

typedef struct {
	float* emb;//block_size,n_embd
	float* res_emb;//block_size,n_embd
	float * qkv;//block_size,3*n_embd
	float* qv;//n_head,block_size,head_size
	float* kv;//n_head,head_size,block_size
	float* vv;//n_head,block_size,head_size
	float* att;//n_head,block_size,block_size
	float* y;//n_head,block_size,head_size
	float* out;//block_size,n_embd
	float* mlp1;//block_size,4*n_embd
	float* mlp2;//block_size,n_embd
	float* logits;//black_size, vocab_size
} Runstate;

void alloc_runstate(Runstate* s,Config *p){

	int head_size = p->n_embd/p->n_head;

	s->emb = (float*)calloc(p->block_size*p->n_embd, sizeof(float));
	s->res_emb = (float*)calloc(p->block_size*p->n_embd, sizeof(float));
	s->qkv = (float*)calloc(p->block_size*3*p->n_embd,sizeof(float));
	s->qv = (float*)calloc(p->n_head * p->block_size * head_size, sizeof(float));
    s->kv = (float*)calloc(p->n_head * head_size * p->block_size, sizeof(float));
    s->vv = (float*)calloc(p->n_head * p->block_size * head_size, sizeof(float));
    s->att = (float*)calloc(p->n_head * p->block_size * p->block_size, sizeof(float));

    s->y = (float*)calloc(p->n_head * p->block_size * head_size, sizeof(float));
    s->out = (float*)calloc(p->block_size * p->n_embd, sizeof(float));
    s->mlp1 = (float*)calloc(p->block_size * 4 * p->n_embd, sizeof(float));
    s->mlp2 = (float*)calloc(p->block_size * p->n_embd, sizeof(float));
    s->logits = (float*)calloc(p->block_size * p->vocab_size, sizeof(float));

    if (s->emb == NULL || s->res_emb == NULL || s->qkv == NULL ||
        s->qv == NULL || s->kv == NULL || s->vv == NULL || s->att == NULL ||
        s->y == NULL || s->out == NULL || s->mlp1 == NULL ||
        s->mlp2 == NULL || s->logits == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(1); // or handle the error as needed
    }

}

void free(Runstate* s){
	free(s->emb);
	free(s->res_emb);
	free(s->qkv);
	free(s->qv);
	free(s->kv);
	free(s->vv);
	free(s->att);
	free(s->y);
	free(s->out);
	free(s->mlp1);
	free(s->mlp2);
	free(s->logits);
}

typedef struct{
	Config config;
	Weights weights;
	Runstate runstate;
}GPT;



float* forward(GPT* gpt,int* tokens){//input have to be blockSzie

	Config* c = &gpt->config;
	Weights* w = &gpt->weights;
	Runstate *s = &gpt->runstate;


	int n_embd = c->n_embd;
	int head_size = n_embd/c->n_head;
	int n_head = c->n_head;
	int block_size = c->block_size;


	//initng bias tril

	float* bias = (float*)calloc(block_size*block_size,sizeof(float));

	for (int i=0;i<block_size;i++) {
    	for (int j=0;j<block_size;j++) {
        	if (i>=j) {
            	bias[i*block_size+j] = 1;
        } else {
            bias[i*block_size+j] = 0;
        }
    }

	//print_mat(bias,block_size,block_size);
}

	for(int i = 0;i< block_size;i++){
		float* row = w->token_embedding_table + tokens[i]*n_embd;
		memcpy(s->emb+i*n_embd, row, n_embd*sizeof(float));
	}
	//print_mat(s->emb,block_size,n_embd);

	for(int i=0;i<block_size;i++){
		float* row = w->pos_embedding_table + i * n_embd;
		add(s->emb+i*n_embd, row , n_embd);
	}
	//print_mat(s->emb,block_size,n_embd);

	memcpy(s->res_emb,s->emb,block_size*n_embd*sizeof(float)); // saving for residual

	//embedding complete neatly.

	float* x = s->emb;
	//print_mat(x,block_size,n_embd);
	// print_mat(s->emb,block_size,n_embd);

	for(int l = 0;l< c->n_layer;l++){

		//print_mat(x,block_size,n_embd);
		LayerNorm(x, w->layernorm_weight + l*n_embd, w->layernorm_bias + l*n_embd, block_size, n_embd);
		matmul(s->qkv,x,w->c_attn + l*(n_embd * 3*n_embd),block_size, n_embd, 3*n_embd);

		//print_mat(s->qkv,block_size,3*n_embd);

		//qv
		for (int i=0; i<n_head;i++) {

    		for (int j=0; j<block_size; j++) {
        		for (int k=0; k<head_size; k++) {
            		s->qv[i*block_size*head_size+ j*head_size + k] = s->qkv[j*(3*n_embd) + 3*k +  6*i];
        		}
    		}

			//print_mat(s->qv+i*block_size*head_size,block_size,head_size);

		//kv
    		for (int j=0; j<head_size; j++) {
        		for (int k=0; k<block_size; k++) {
            		s->kv[i*block_size*head_size+ j*block_size + k] = s->qkv[k*(3*n_embd) + 3*j + 1 + 6*i];
        		}
    		}

		//vv
    		for (int j=0; j<block_size; j++) {
        		for (int k=0; k<head_size; k++) {
            		s->vv[i*block_size*head_size+ j*head_size + k] = s->qkv[j*(3*n_embd) + 3*k + 2 + 6*i];
        		}
    		}


		// ///completed with ktranspose inbuilt

		// ///now they are block of nh, block_size, head_size
		// //k of n_head, head_size, block_size

		// ///att have to be in shape n_head,block_size, block_size


        	for (int j=0; j<block_size; j++) {
            	for (int k=0; k<block_size; k++) {
                	float sum = 0.0f;
                	for (int h=0; h<head_size; h++) {
                    	sum += s->qv[i*block_size*head_size + j*head_size + h] * s->kv[i*head_size * block_size + h*block_size + k];
                	}
                	s->att[i*block_size*block_size + j*block_size + k] = sum/sqrtf(head_size);
            	}
        	}

			//print_mat(s->att+i*block_size*block_size,block_size,block_size);



		////attention matrix obtained

        	for(int j=0; j<block_size;j++){
        		for(int k =0;k<block_size;k++){
        			if(bias[j*block_size+k]==0){
        				s->att[i*block_size*block_size + j*block_size + k] = -inf;
        			}
        		}
        	}

        	//print_mat(s->att,block_size,block_size);

        ////softmaxing of att of nh,block_size,block_size
        	for(int j=0;j<block_size;j++){

				float row_max = -inf;
        		//max_value
        		for (int k = 0; k < block_size; k++) {
            		float val = s->att[i*block_size*block_size + j*block_size + k];
            		row_max = fmaxf(row_max, val);
        		}

        		float row_sum = 0.0f;
        		//soft vals
        		for(int k=0;k<block_size;k++){
        			float val = s->att[i*block_size*block_size+j*block_size+k];
        			val = expf(val - row_max);
					assert(i*block_size*block_size+j*block_size+k<n_head*block_size*block_size);
        			s->att[i*block_size*block_size+j*block_size+k] = val;
        			row_sum += val;
        		}

        		for(int n=0;n<block_size;n++){
        			s->att[i*block_size*block_size+j*block_size+n] /= row_sum;
        		}
        	}

			//print_mat(s->att+i*block_size*block_size,block_size,block_size);
			//print_mat(s->vv+i*block_size*head_size,block_size,head_size);
			//print_mat(s->y+i*head_size*block_size,block_size,head_size);//4,2 | 4,4 @ 4,2

			int att_start = i * block_size * block_size;
     		int vv_start = i * block_size * head_size;
    		int y_start = i * block_size * head_size;

    		matmul(s->y + y_start, s->att + att_start, s->vv + vv_start, block_size, block_size, head_size);
			//print_mat(s->y + y_start, block_size, head_size);
        }
		// print_mat(s->y,block_size,n_embd);
        ///end of n_head loop
        float* temp = (float*)calloc(block_size * n_head * head_size ,sizeof(float));

        for (int i = 0; i < n_head; i++) {
        	for (int j = 0; j < block_size; j++) {
            	for (int k = 0; k < head_size; k++) {
               	 	temp[j * n_head * head_size + i * head_size + k] = s->y[i * block_size * head_size + j * head_size + k];
            	}
        	}
    	}

    	for(int i=0;i<n_head*block_size*head_size;i++){
    		s->y[i] = temp[i];
    	}

    	free(temp);

		//print_mat(s->y,block_size,n_embd);

    // 	// //now y can be eaccessed by [block_Sz,n_embd]

     	matmul(s->out, s->y, w->c_proj + l*n_embd*n_embd, block_size,n_embd,n_embd);

    	/////outed from causal attention

    	/////attention ends


    	add2d(s->out,s->res_emb,block_size,n_embd);
    	LayerNorm(s->out, w->layernorm2_weight + l*n_embd, w->layernorm2_bias + l*n_embd, block_size, n_embd);//block_size,n_embd

    	// ///MLP
    	matmul(s->mlp1,s->out, w->mlp_c_fc + l*n_embd*4*n_embd,block_size,n_embd,4*n_embd);
    	for(int i = 0;i<block_size;i++){
    		for(int j = 0;j<4*n_embd;j++){
    			s->mlp1[i*4*n_embd+j] = gelu(s->mlp1[i*4*n_embd+j]);
    		}
    	}
    	//print_mat(s->mlp1,block_size,4*n_embd);
    	matmul(s->mlp2,s->mlp1,w->mlp_c_proj + l*n_embd*4*n_embd,block_size,4*n_embd,n_embd);//block_size,4*n_embd@4*n_embd,n_embd
    	add2d(s->mlp2,s->res_emb,block_size,n_embd);

    	x = s->mlp2;

	}
    ///end all layers;
	//print_mat(x,block_size,n_embd);
	//print_mat(w->fin_layernorm_weight,1,n_embd);
	//print_mat(w->fin_layernorm_bias,1,n_embd);

	LayerNorm(x,w->fin_layernorm_weight,w->fin_layernorm_bias,block_size,n_embd);
	//print_mat(x,block_size,n_embd);

	matmul(s->logits, x, w->lm_head_w,block_size, n_embd, c->vocab_size);//block_size .vocab_size <- block_siw,n_embd @ n_embd,vocab_size
	free(bias);

    return s->logits;

}


int main(){
	clock_t start_time = clock();

	GPT model;

	model.config.block_size = 1024;
	model.config.vocab_size = 256*128;
	model.config.n_embd = 256;
	model.config.n_head = 8;
	model.config.n_layer = 4;

	int total_size =
    model.config.vocab_size * model.config.n_embd +
    model.config.block_size * model.config.n_embd +
    model.config.n_layer * model.config.n_embd * 4 + // layernorm_weight, layernorm_bias, layernorm2_weight, layernorm2_bias, c_attn, c_proj
    model.config.n_layer * model.config.n_embd * 4 * model.config.n_embd +
    model.config.n_layer * model.config.n_embd * 8 * model.config.n_embd + // mlp_c_fc, mlp_c_proj
	model.config.n_embd * 2 + // fin_layernorm_weight, fin_layernorm_bias
    model.config.n_embd * model.config.vocab_size;



	float* ptr = (float*)calloc(total_size , sizeof(float));

	printf("total_size:%d",total_size);

	if(ptr==NULL){
		printf("No memory");
	}

	init_weights(&model.weights,&model.config,ptr);
	alloc_runstate(&model.runstate, &model.config);


	int tokens[model.config.block_size];
	for(int i = 0;i<model.config.block_size;i++)tokens[i]=i;

	float* logits = forward(&model,tokens);

    //print_mat(logits,model.config.block_size,model.config.vocab_size);

	free(&model.runstate);
	free(ptr);

	clock_t end_time = clock();
	double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Elapsed time: %.2f seconds\n", elapsed_time);



	return 0;
}