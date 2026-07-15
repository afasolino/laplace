#include <assert.h>
#include <stddef.h>
#include <string.h>

int bounded_copy(char *destination, size_t capacity, const char *source);

int main(void) {
    char output[8] = "old";
    assert(bounded_copy(output, sizeof output, "abc") == 0);
    assert(strcmp(output, "abc") == 0);
    return 0;
}
