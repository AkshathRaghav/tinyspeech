#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>  

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

// Each of the following functions uses sleep to simulate input 

int detect_load() {
    sleep(1);  
    return 1;  
}

int select_water_level() {
    sleep(1);  
    return 1;  
}

int select_temperature() {
    sleep(1);  
    return 1;  
}

void run_timer(int seconds) {
    sleep(seconds);  
}

int main() {
    State current_state = OFF;
    Event event;

    while (1) {
        event = POWER_BUTTON; // For testing purposes

        if (event == POWER_BUTTON && current_state == OFF) {
            current_state = state_machine(current_state, POWER_BUTTON);
        } else if (event == START_PAUSE_BUTTON && current_state == IDLE) {
            current_state = state_machine(current_state, START_PAUSE_BUTTON);
        } else {
            while (current_state != IDLE && current_state != OFF) {
                current_state = state_machine(current_state, event);
                printf("Current state: %d\n", current_state);
                if (current_state == END) {
                    break;
                }
            }
        }
    }

    return 0;
}
