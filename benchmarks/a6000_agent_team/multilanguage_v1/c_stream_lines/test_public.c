#include <assert.h>
#include <stdio.h>
#include <string.h>

int read_line(FILE *stream, char *buffer, size_t capacity);

int main(void) {
    FILE *stream = tmpfile();
    char buffer[16];
    assert(stream != NULL);
    assert(fputs("alpha\n", stream) >= 0);
    rewind(stream);
    assert(read_line(stream, buffer, sizeof buffer) == 1);
    assert(strcmp(buffer, "alpha") == 0);
    assert(fclose(stream) == 0);
    return 0;
}
