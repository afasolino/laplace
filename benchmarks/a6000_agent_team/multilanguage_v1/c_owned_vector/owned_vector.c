#include <stddef.h>
#include <stdlib.h>

int vector_push(int **data, size_t *size, size_t *capacity, int value) {
    int *replacement;
    size_t next_capacity;
    if (data == NULL || size == NULL || capacity == NULL || *size > *capacity) {
        return -1;
    }
    if (*size == *capacity) {
        next_capacity = *capacity == 0U ? 4U : *capacity * 2U;
        replacement = realloc(*data, next_capacity * sizeof **data);
        if (replacement == NULL) {
            return -2;
        }
        *data = replacement;
        *capacity = next_capacity;
    }
    (*data)[*size] = value;
    *size += 1U;
    return 0;
}

void vector_destroy(int **data, size_t *size, size_t *capacity) {
    if (data != NULL) {
        free(*data);
        *data = NULL;
    }
    if (size != NULL) {
        *size = 0U;
    }
    if (capacity != NULL) {
        *capacity = 0U;
    }
}
