#include <assert.h>
#include <stdlib.h>
#include <string.h>

char *duplicate_trimmed(const char *input);

int main(void) {
    char *value = duplicate_trimmed("  alpha  ");
    assert(value != NULL && strcmp(value, "alpha") == 0);
    free(value);
    return 0;
}
