enum protocol_state { PROTOCOL_IDLE = 0, PROTOCOL_ACTIVE = 1, PROTOCOL_DONE = 2 };
enum protocol_event { EVENT_START = 0, EVENT_FINISH = 1, EVENT_RESET = 2 };

int protocol_step(enum protocol_state *state, enum protocol_event event) {
    if (state == 0) {
        return -1;
    }
    /* Intentional seeded defect: reset must work from every valid state. */
    if (event == EVENT_RESET && *state != PROTOCOL_DONE) {
        *state = PROTOCOL_IDLE;
        return 0;
    }
    if (*state == PROTOCOL_IDLE && event == EVENT_START) {
        *state = PROTOCOL_ACTIVE;
        return 0;
    }
    if (*state == PROTOCOL_ACTIVE && event == EVENT_FINISH) {
        *state = PROTOCOL_DONE;
        return 0;
    }
    return -2;
}
