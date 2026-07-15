#include <assert.h>
#include <stdint.h>

int checked_counter_add(int64_t *state, int64_t delta);

int main(void) {
    int64_t value = 10;
    assert(checked_counter_add(&value, -3) == 0);
    assert(value == 7);
    return 0;
}
