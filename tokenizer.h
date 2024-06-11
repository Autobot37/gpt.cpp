#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>
#include <assert.h>
#include "utils.h"

#define MAX_KEY_LENGTH 100

typedef struct {
    int vocab_size;
    char** token_table;
    int eot_token;
} Tokenizer;

void safe_printf(const char* piece){
    if(piece==NULL){
        return;
    }
    if(piece[0]=='\0'){
        return;
    }
    if(piece[1]=='\0'){
        unsigned char byte_val = piece[0];
        if(!(isprint(byte_val) || isspace(byte_val))){
            return;
        }
    }
    printf("%s", piece);
}

void tokenizer_init(Tokenizer* tokenizer, const char* filename){
    FILE *file = fopenCheck(filename, "rb");
    int header[256];
    freadCheck(header, sizeof(int), 256, file);
    assert(header[0] == 20240328);
    tokenizer->vocab_size = header[2];
    tokenizer->eot_token = header[3];

    unsigned char length;
    tokenizer->token_table = (char**)mallocCheck(tokenizer->vocab_size * sizeof(char*));
    for(uint32_t i=0;i<tokenizer->vocab_size;i++){
        freadCheck(&length, sizeof(unsigned char), 1, file);
        assert(length>0);
        char* token_bytes = (char*)mallocCheck(length+1);
        freadCheck(token_bytes, sizeof(char), length, file);
        token_bytes[length] = '\0';
        tokenizer->token_table[i] = token_bytes;
    }
    fcloseCheck(file);
    printf("Tokenizer initialized\n");
}

const char* tokenizer_decode(Tokenizer* tokenizer, int token){
    if(token<0 || token>=tokenizer->vocab_size){
        return NULL;
    }
    return tokenizer->token_table[token];
}

// int main() {
    
//     Tokenizer tokenizer;
//     tokenizer_init(&tokenizer, "tokenizer.bin");
//     int token;
//     while(scanf("%d", &token) == 1){
//         const char* piece = tokenizer_decode(&tokenizer, token);
//         safe_printf(piece);
//         printf("\n");
//     }
//     return 0;
// }
