#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define MAX_KEY_LENGTH 100

int main() {
    FILE *file = fopen("dictionary.bin", "rb");
    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }

    // Read the entire file into memory
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    uint8_t *buffer = (uint8_t *)malloc(file_size);
    if (buffer == NULL) {
        perror("Error allocating memory");
        fclose(file);
        return 1;
    }
    if (fread(buffer, 1, file_size, file) != file_size) {
        perror("Error reading file");
        free(buffer);
        fclose(file);
        return 1;
    }
    fclose(file);

    // Unpack the data
    int offset = 0;
    int magic_num = *(int *)(buffer + offset);
    offset += sizeof(int);
    int dict_size = *(int *)(buffer + offset);
    offset += sizeof(int);
    printf("Magic number: %d\n", magic_num);
    printf("Dictionary size: %d\n", dict_size);
    
    // Initialize a dictionary
    char **keys = (char **)malloc(dict_size * sizeof(char *));
    int *values = (int *)malloc(dict_size * sizeof(int));
    if (keys == NULL || values == NULL) {
        perror("Error allocating memory");
        free(buffer);
        return 1;
    }

    // Iterate over dictionary items and unpack key-value pairs
    for (int i = 0; i < dict_size; i++) {
        int key_size = *(int *)(buffer + offset);
        offset += sizeof(int);
        char *key = (char *)malloc((key_size + 1) * sizeof(char));
        if (key == NULL) {
            perror("Error allocating memory");
            free(buffer);
            return 1;
        }
        strncpy(key, (char *)(buffer + offset), key_size);
        key[key_size] = '\0';  // Null-terminate the string
        offset += key_size;
        int val = *(int *)(buffer + offset);
        offset += sizeof(int);

        // Store key-value pair in the dictionary
        keys[i] = key;
        values[i] = val;
    }

    printf("key: %s, value: %d\n", keys[0], values[0]);

    // Print the dictionary
    printf("Dictionary unpacked successfully:\n");
    for (int i = 0; i < dict_size; i++) {
        free(keys[i]);  // Free allocated memory for keys
    }
    free(keys);
    free(values);
    free(buffer);

    return 0;
}
