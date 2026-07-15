#include <stdint.h>

int checked_counter_add(int64_t *state, int64_t delta) {
    int64_t next;
    if (state == 0) {
        return -1;
    }
    next = *state + delta;
    *state = next;
    return 0;
}
