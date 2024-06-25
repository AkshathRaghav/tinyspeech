# Task 2: Automated Washing Machine Scheduler 

The ![main.c](main.c) contains an implementation of a scheduler for the washing machine. It depends on user-inputs to switch and select between different states. 

Here are the states: 

```c
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
```

Here is our state machine: 

```c
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

```

We compile the same using `riscv64-unknown-elf-gcc -O1 -mabi=lp64 -march=rv64i -o main.o main.c`. This gives us our RISCV implementation of the C code. 

Here is our state_machine function: 

```
0000000000010184 <state_machine>:
   10184:	00800793          	li	a5,8
   10188:	06a7ea63          	bltu	a5,a0,101fc <state_machine+0x78>
   1018c:	00010737          	lui	a4,0x10
   10190:	00251793          	slli	a5,a0,0x2
   10194:	65470713          	addi	a4,a4,1620 # 10654 <__errno+0xc>
   10198:	00e787b3          	add	a5,a5,a4
   1019c:	0007a783          	lw	a5,0(a5)
   101a0:	00078067          	jr	a5
   101a4:	00300513          	li	a0,3
   101a8:	00008067          	ret
   101ac:	00400513          	li	a0,4
   101b0:	00008067          	ret
   101b4:	00500513          	li	a0,5
   101b8:	00008067          	ret
   101bc:	00600513          	li	a0,6
   101c0:	00008067          	ret
   101c4:	00700513          	li	a0,7
   101c8:	00008067          	ret
   101cc:	00800513          	li	a0,8
   101d0:	00008067          	ret
   101d4:	00b035b3          	snez	a1,a1
   101d8:	40b005b3          	neg	a1,a1
   101dc:	00b57533          	and	a0,a0,a1
   101e0:	00008067          	ret
   101e4:	0015b513          	seqz	a0,a1
   101e8:	00008067          	ret
   101ec:	00100793          	li	a5,1
   101f0:	faf59ce3          	bne	a1,a5,101a8 <state_machine+0x24>
   101f4:	00200513          	li	a0,2
   101f8:	00008067          	ret
   101fc:	00000513          	li	a0,0
   10200:	00008067          	ret
```



