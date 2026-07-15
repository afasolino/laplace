#include <stddef.h>
#include <string.h>

int bounded_copy(char *destination, size_t capacity, const char *source) {
    size_t length;
    if (destination == NULL || source == NULL) {
        return -1;
    }
    length = strlen(source);
    /* Intentional seeded defect: exact-capacity input has no room for NUL. */
    if (length > capacity) {
        return -2;
    }
    memmove(destination, source, length + 1U);
    return 0;
}
