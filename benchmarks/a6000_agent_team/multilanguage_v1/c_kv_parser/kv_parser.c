#include <stddef.h>
#include <string.h>

int parse_kv(const char *input, char *key, size_t key_capacity, char *value,
             size_t value_capacity) {
    const char *separator;
    size_t key_length;
    size_t value_length;
    if (input == NULL || key == NULL || value == NULL) {
        return -1;
    }
    separator = strchr(input, '=');
    if (separator == NULL) {
        return -2;
    }
    key_length = (size_t)(separator - input);
    value_length = strlen(separator + 1);
    if (key_length == 0U || key_length >= key_capacity || value_length >= value_capacity) {
        return -3;
    }
    memcpy(key, input, key_length);
    key[key_length] = '\0';
    memcpy(value, separator + 1, value_length + 1U);
    return 0;
}
