#include <assert.h>

enum protocol_state { PROTOCOL_IDLE = 0, PROTOCOL_ACTIVE = 1, PROTOCOL_DONE = 2 };
enum protocol_event { EVENT_START = 0, EVENT_FINISH = 1, EVENT_RESET = 2 };
int protocol_step(enum protocol_state *state, enum protocol_event event);

int main(void) {
    enum protocol_state state = PROTOCOL_IDLE;
    assert(protocol_step(&state, EVENT_START) == 0);
    assert(state == PROTOCOL_ACTIVE);
    assert(protocol_step(&state, EVENT_FINISH) == 0);
    assert(state == PROTOCOL_DONE);
    return 0;
}
