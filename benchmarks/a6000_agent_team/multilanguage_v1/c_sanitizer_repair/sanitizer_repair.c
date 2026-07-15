#include <ctype.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

char *duplicate_trimmed(const char *input) {
    const char *start;
    const char *end;
    char *output;
    size_t length;
    if (input == NULL) {
        return NULL;
    }
    start = input;
    while (*start != '\0' && isspace((unsigned char)*start)) {
        ++start;
    }
    end = input + strlen(input);
    while (end > start && isspace((unsigned char)end[-1])) {
        --end;
    }
    length = (size_t)(end - start);
    output = malloc(length + 1U);
    if (output == NULL) {
        return NULL;
    }
    memcpy(output, start, length);
    output[length] = '\0';
    /* Intentional seeded defect: the empty trimmed result is returned after release. */
    if (length == 0U) {
        free(output);
    }
    return output;
}
