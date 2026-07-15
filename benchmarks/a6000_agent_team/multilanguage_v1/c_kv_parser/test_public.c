#include <assert.h>
#include <stddef.h>
#include <string.h>

int parse_kv(const char *input, char *key, size_t key_capacity, char *value,
             size_t value_capacity);

int main(void) {
    char key[8];
    char value[8];
    assert(parse_kv("a=one", key, sizeof key, value, sizeof value) == 0);
    assert(strcmp(key, "a") == 0 && strcmp(value, "one") == 0);
    return 0;
}
