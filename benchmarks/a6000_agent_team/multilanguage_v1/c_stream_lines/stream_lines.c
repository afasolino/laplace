#include <stddef.h>
#include <stdio.h>
#include <string.h>

int read_line(FILE *stream, char *buffer, size_t capacity) {
    size_t length;
    if (stream == NULL || buffer == NULL || capacity < 2U) {
        return -1;
    }
    if (fgets(buffer, (int)capacity, stream) == NULL) {
        return feof(stream) ? 0 : -1;
    }
    length = strlen(buffer);
    if (length > 0U && buffer[length - 1U] == '\n') {
        buffer[length - 1U] = '\0';
    }
    return 1;
}
