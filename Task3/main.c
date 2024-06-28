#include <stdio.h>
#include <stdlib.h>

typedef enum {
    OFF,
    IDLE,
    LOAD_SENSING,
    WATER_SELECTION,
    TEMP_SELECTION,
    WASH,
    RINSE,
    SPIN,
    END
} State;

typedef enum {
    POWER_BUTTON,
    START_PAUSE_BUTTON,
    CYCLE_SELECTION,
    WATER_LEVEL_SELECTION,
    TEMP_SELECTION_EVENT,
    CYCLE_COMPLETE
} Event;

int detect_load();
int select_water_level();
int select_temperature();
void run_timer(int seconds);

State state_machine(State current_state, Event event) {
    switch (current_state) {
        case OFF:
            if (event == POWER_BUTTON) return IDLE;
            break;
        case IDLE:
            if (event == START_PAUSE_BUTTON) return LOAD_SENSING;
            break;
        case LOAD_SENSING:
            if (detect_load()) return WATER_SELECTION;
            break;
        case WATER_SELECTION:
            if (select_water_level()) return TEMP_SELECTION;
            break;
        case TEMP_SELECTION:
            if (select_temperature()) return WASH;
            break;
        case WASH:
            run_timer(10); 
            return RINSE;
        case RINSE:
            run_timer(5); 
            return SPIN;
        case SPIN:
            run_timer(3); 
            return END;
        case END:
            if (event == POWER_BUTTON) return OFF;
            break;
        default:
            return OFF;
    }
    return current_state;
}

int detect_load() {
    return 1;  
}

int select_water_level() {
    return 1;  
}

int select_temperature() {
    return 1;  
}

void run_timer(int seconds) {
    int a = 100 + 100; // dummy instead of sleep()
}

void print_state(State state) {
    switch (state) {
        case OFF:
            printf("-> OFF\n");
            break;
        case IDLE:
            printf("-> IDLE\n");
            break;
        case LOAD_SENSING:
            printf("-> LOAD_SENSING\n");
            break;
        case WATER_SELECTION:
            printf("-> WATER_SELECTION\n");
            break;
        case TEMP_SELECTION:
            printf("-> TEMP_SELECTION\n");
            break;
        case WASH:
            printf("-> WASH\n");
            break;
        case RINSE:
            printf("-> RINSE\n");
            break;
        case SPIN:
            printf("-> SPIN\n");
            break;
        case END:
            printf("-> END\n");
            break;
        default:
            printf("-> FAILED!\n");
            break;
    }
}

void test_state_machine() {
    State current_state = OFF;

    Event events[] = {
        POWER_BUTTON, START_PAUSE_BUTTON, CYCLE_SELECTION, 
        WATER_LEVEL_SELECTION, TEMP_SELECTION_EVENT, CYCLE_COMPLETE
    };

    for (int i = 0; i < sizeof(events)/sizeof(events[0]); ++i) {
        current_state = state_machine(current_state, events[i]);
        print_state(current_state);
    }

    while (current_state != END) {
        current_state = state_machine(current_state, CYCLE_COMPLETE);
        print_state(current_state);
    }

    current_state = state_machine(current_state, POWER_BUTTON);
    print_state(current_state);
}

int main() {
    test_state_machine();
    return 0;
}
