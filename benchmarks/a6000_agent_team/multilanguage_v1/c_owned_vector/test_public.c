#include <assert.h>
#include <stddef.h>

int vector_push(int **data, size_t *size, size_t *capacity, int value);
void vector_destroy(int **data, size_t *size, size_t *capacity);

int main(void) {
    int *data = NULL;
    size_t size = 0U;
    size_t capacity = 0U;
    assert(vector_push(&data, &size, &capacity, 9) == 0);
    assert(size == 1U && data[0] == 9);
    vector_destroy(&data, &size, &capacity);
    assert(data == NULL && size == 0U && capacity == 0U);
    return 0;
}
