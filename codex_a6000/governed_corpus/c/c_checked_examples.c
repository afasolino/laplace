/* Release multilanguage-corpus-v1; SPDX-License-Identifier: MIT */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

bool checked_size_add(size_t left, size_t right, size_t *result) {
    if (result == NULL || left > SIZE_MAX - right) {
        return false;
    }
    *result = left + right;
    return true;
}

bool copy_c_string(char *destination, size_t capacity, const char *source) {
    size_t length;
    if (destination == NULL || source == NULL) {
        return false;
    }
    length = strlen(source);
    if (length >= capacity) {
        return false;
    }
    memmove(destination, source, length + 1U);
    return true;
}

void *allocate_array(size_t count, size_t element_size) {
    if (element_size != 0U && count > SIZE_MAX / element_size) {
        return NULL;
    }
    return calloc(count, element_size);
}
