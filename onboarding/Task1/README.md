## Task 1: C Code to RISCV 

Write a C program to add numbers together, and then generate assembly code for the same. 

0. Setting up the provided VDI file: 

![Untitled](https://github.com/AkshathRaghav/vsdsquadronmini/assets/75845563/2327e2aa-8bab-43cc-b190-c6e37e91ace5)


1. First, we wrote the C code in the sum1ton.c file. We compiled it using `gcc sum1ton.c`

![Untitled](https://github.com/AkshathRaghav/vsdsquadronmini/assets/75845563/e3998877-ae91-4f72-95e2-908ef7f6bf01)
![Untitled-1](https://github.com/AkshathRaghav/vsdsquadronmini/assets/75845563/8523d7e6-f24e-47aa-bcb4-c53c1c28032f)


2. To convert to assembly, we ran `riscv64-unknown-elf-gcc -O1 -mabi=lp64 -march=rv64i -o sum1ton.o sum1ton.c`

![Untitled](https://github.com/AkshathRaghav/vsdsquadronmini/assets/75845563/3aace0f4-a4d3-4c61-85c0-8fe8dc577bf3)

3. To visualize it we ran `riscv64-unknown-elf-objdump -d sum1ton.o | less`. We were also able to use the 'programmer' mode on Windows Calculator to cross check the opcodes. 

![Untitled-1](https://github.com/AkshathRaghav/vsdsquadronmini/assets/75845563/e87645fc-eb27-4333-aa62-64c5a9057822)

4. We also test the compilation flags to view the difference with `#instructions`. If we replace `-o1` with `-ofast`, we reduce the instructions from 15 to 12.

![image](https://github.com/AkshathRaghav/vsdsquadronmini/assets/75845563/61f07e59-8ac8-4376-af5b-e1516d09fcbe)
