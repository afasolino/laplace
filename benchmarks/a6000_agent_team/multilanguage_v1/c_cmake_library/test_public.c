#include <assert.h>
#include <stddef.h>
#include <stdint.h>

uint32_t rolling_checksum(const unsigned char *data, size_t length);

int main(void) {
    const unsigned char data[] = {1U, 2U};
    assert(rolling_checksum(data, sizeof data) == 34U);
    return 0;
}
