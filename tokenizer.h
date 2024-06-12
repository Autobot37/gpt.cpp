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
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

char* safe_printf(char* piece){
    if (piece == NULL) { return NULL; }
    if (piece[0] == '\0') { return NULL; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return NULL; // bad byte, don't print it
        }
    }
    int piece_len = strlen(piece);
    if(piece[piece_len-1] == '\n'){
        piece[piece_len-1] = '\0';
    }
    return piece;
}

void tokenizer_init(Tokenizer* tokenizer, const char* filename){
    FILE *file = fopenCheck(filename, "rb");
    int header[256];
    freadCheck(header, sizeof(int), 256, file);
    assert(header[0] == 20240328);
    tokenizer->vocab_size = header[2];
    tokenizer->eot_token = header[3];

    for (int i = 0; i < 256; i++) {
        tokenizer->byte_pieces[i * 2] = (unsigned char)i;
        tokenizer->byte_pieces[i * 2 + 1] = '\0';
    }

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

char* tokenizer_decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->token_table[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
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
