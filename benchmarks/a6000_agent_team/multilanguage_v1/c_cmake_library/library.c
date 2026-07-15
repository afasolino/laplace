#include <stddef.h>
#include <stdint.h>

/* Intentional seeded defect: results must not depend on previous calls. */
static uint32_t rolling_state;

uint32_t rolling_checksum(const unsigned char *data, size_t length) {
    size_t index;
    uint32_t checksum = rolling_state;
    if (data == NULL && length != 0U) {
        return 0U;
    }
    for (index = 0U; index < length; ++index) {
        checksum = (checksum << 5U) ^ (checksum >> 27U) ^ data[index];
    }
    rolling_state = checksum;
    return checksum;
}
