#include <stdio.h>
#include <stdlib.h>

//---------------------
void *malloc_check(size_t size, const char *file, int line) {
    void *ptr = malloc(size);
    if (ptr == NULL) {
        fprintf(stderr, "Error: Memory allocation failed at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        fprintf(stderr, "  Size: %zu bytes\n", size);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

#define mallocCheck(size) malloc_check(size, __FILE__, __LINE__)
//------------

FILE* fopen_check(const char *filename, const char *mode, const char *file, int line) {
    FILE *file_ptr = fopen(filename, mode);
    if (file_ptr == NULL) {
        fprintf(stderr, "Error: Failed to open file at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        fprintf(stderr, "  Filename: %s\n", filename);
        fprintf(stderr, "  Mode: %s\n", mode);
        exit(EXIT_FAILURE);
    }
    return file_ptr;
}

#define fopenCheck(filename, mode) fopen_check(filename, mode, __FILE__, __LINE__)

//------------
void fseek_check(FILE *stream, long offset, int whence, const char *file, int line) {
    if (fseek(stream, offset, whence) != 0) {
        fprintf(stderr, "Error: Failed to seek in file at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        exit(EXIT_FAILURE);
    }
}

#define fseekCheck(stream, offset, whence) fseek_check(stream, offset, whence, __FILE__, __LINE__)
//------------
void fread_check(void *ptr, size_t size, size_t nmemb, FILE *stream, const char *file, int line) {
    if (fread(ptr, size, nmemb, stream) != nmemb) {
        fprintf(stderr, "Error: Failed to read from file at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        exit(EXIT_FAILURE);
    }
}

#define freadCheck(ptr, size, nmemb, stream) fread_check(ptr, size, nmemb, stream, __FILE__, __LINE__)

//------------
void fclose_check(FILE *stream, const char *file, int line) {
    if (fclose(stream) != 0) {
        fprintf(stderr, "Error: Failed to close file at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        exit(EXIT_FAILURE);
    }
}

#define fcloseCheck(stream) fclose_check(stream, __FILE__, __LINE__)

//------------

