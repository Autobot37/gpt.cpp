#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cctype>
#include <cstring>
using namespace std;
#include "utils.h"

#define MAX_KEY_LENGTH 100

string safe_printf(const string& piece) {
    if (piece.empty()) {
        return ""; 
    }
    if (piece.length() == 1) {
        unsigned char byte_val = static_cast<unsigned char>(piece[0]);
        if (!isprint(byte_val) && !std::isspace(byte_val)) {
            return ""; 
        }
    }
    string result = piece;
    if (result.back() == '\n') {
        result.pop_back();
    }

    return result;
}

struct Decoder{
    int vocab_size;
    vector<string> token_table;
    int eot_token;
    vector<string> byte_pieces; // stores all single-byte strings
    Decoder();
    void init(const char* filename);
    string decode(int prev_token, int token);
};
Decoder::Decoder() {
    byte_pieces.resize(256);
    for (int i = 0; i < 256; i++) {
        byte_pieces[i] = string(1, static_cast<char>(i));
    }
}


void Decoder::init(const char* filename){
    ifstream file(filename, ios::binary);
    if(!file.is_open()){
        cerr << "Error: Failed to open bin file: " << filename << endl;
        exit(EXIT_FAILURE);
    }
    int header[256];
    file.read((char*)header, 256*sizeof(int));
    if(header[0] != 20240328){
        cerr << "Error: Invalid bin file: " << filename << endl;
        exit(EXIT_FAILURE);
    }
    vocab_size = header[2];
    eot_token = header[3];

    token_table.resize(vocab_size);
    for(int i = 0;i<vocab_size;i++){
        unsigned char length;
        file.read((char*)&length, sizeof(unsigned char));
        string token(length, '\0');
        file.read(&token[0], length);
        token_table[i] = token;
    }
    file.close();
    printf("Decoder initialized\n");
}

string Decoder::decode(int prev_token, int token) {
    string piece = token_table[token];
    if (prev_token == 1 && piece[0] == ' ') {
        piece.erase(0, 1);
    }
    unsigned char byte_val;
    if (sscanf(piece.c_str(), "<0x%02hhX>", &byte_val) == 1) {
        piece = byte_pieces[byte_val];
    }
    return piece;
}

struct TrieNode {
    unordered_map<char, TrieNode*> children;
    int token_length = -1;
    int token_id = -1;
    char character;

    TrieNode(char c = '\0');
    ~TrieNode();
    void init(const vector<string>& tokens);
    vector<int> encode(const string& prompt);
};

TrieNode::TrieNode(char c) : character(c) {}

TrieNode::~TrieNode() {
    for (auto& child : children) {
        delete child.second;
    }
}

void TrieNode::init(const vector<string>& tokens) {
    for (int i = 0; i < tokens.size(); i++) {
        const string& token = tokens[i];
        TrieNode* node = this;
        for (char c : token) {
            if (node->children.find(c) == node->children.end()) {
                node->children[c] = new TrieNode(c);
            }
            node = node->children[c];
        }
        node->token_length = token.length();
        node->token_id = i;
    }
    printf("TrieNode initialized\n");
}

vector<int> TrieNode::encode(const string& prompt) {
    vector<int> encoded;
    TrieNode* node = this;
    int i = 0;
    int last_token_id = -1;
    while (i < prompt.length()) {
        char key = prompt[i];
        if (node->children.find(key) == node->children.end()) {
            node = this;
            encoded.push_back(last_token_id);
            last_token_id = -1;
        } else {
            node = node->children[key];
            if (node->token_id != -1) {
                last_token_id = node->token_id;
            }
            i++;
        }
    }
    encoded.push_back(last_token_id);
    return encoded;
}

struct Tokenizer {
    Decoder decoder;
    TrieNode trie;
    Tokenizer();
    void init(const char* filename);
    vector<int> encode(const string& prompt);
    string decode(int prev_token, int token);
};

Tokenizer::Tokenizer() {
    decoder = Decoder();
    trie = TrieNode();
}

void Tokenizer::init(const char* filename) {
    decoder.init(filename);
    trie.init(decoder.token_table);
    printf("Tokenizer initialized\n");
}

vector<int> Tokenizer::encode(const string& prompt) {
    return trie.encode(prompt);
}

string Tokenizer::decode(int prev_token, int token) {
    return decoder.decode(prev_token, token);
}

// int main() {
    
//     Tokenizer tokenizer;
//     tokenizer.init("tokenizer.bin");
//     string prompt = "Hello, world!";
//     vector<int> encoded = tokenizer.encode(prompt);
//     int prev_token = -1;
//     for (int token : encoded) {
//         string piece = tokenizer.decode(prev_token, token);
//         piece = safe_printf(piece);
//         printf("%s", piece.c_str());
//         prev_token = token;
//     }
    

//     return 0;
// }
