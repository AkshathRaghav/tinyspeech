### Task 3 

### RISCV Overview 

#### 1. Register Lengths

RISC-V supports the following register lengths: 

- **RV32I**: 32-bit registers.
- **RV64I**: 64-bit registers.
- **RV128I**: 128-bit registers.

#### 2. Types of Instructions

RISC-V instructions are categorized into six main types based on their functionality:

- **R-Type**: Register-register operations.
- **I-Type**: Immediate operations.
- **S-Type**: Store operations.
- **B-Type**: Branch operations.
- **U-Type**: Upper immediate operations.
- **J-Type**: Jump operations.

#### 3. Register Formats for Each Type

- **R-Type (Register-Register Instructions)**:
  - Format: `opcode rd, rs1, rs2`

- **I-Type (Immediate Instructions)**:
  - Format: `opcode rd, rs1, imm`

- **S-Type (Store Instructions)**:
  - Format: `opcode rs1, imm(rs2)`

- **B-Type (Branch Instructions)**:
  - Format: `opcode rs1, rs2, offset`

- **U-Type (Upper Immediate Instructions)**:
  - Format: `opcode rd, imm`

- **J-Type (Jump Instructions)**:
  - Format: `opcode rd, offset`

## Identifying Instruction Code 

Here are the 32-bit instruction codes:

1. **ADD r1, r2, r3**
   - Instruction Type: R-Type
   - Opcode (ADD): `0110011`
   - rd (r1): `00001`
   - rs1 (r2): `00010`
   - rs2 (r3): `00011`
   - Funct3 (ADD): `000`
   - Funct7 (ADD): `0000000`
   - **Code**: `0000000 00011 00010 000 00001 0110011`

2. **SUB r3, r1, r2**
   - Instruction Type: R-Type
   - Opcode (SUB): `0110011`
   - rd (r3): `00011`
   - rs1 (r1): `00001`
   - rs2 (r2): `00010`
   - Funct3 (SUB): `000`
   - Funct7 (SUB): `0100000`
   - **Code**: `0100000 00010 00001 000 00011 0110011`

3. **AND r2, r1, r3**
   - Instruction Type: R-Type
   - Opcode (AND): `0110011`
   - rd (r2): `00010`
   - rs1 (r1): `00001`
   - rs2 (r3): `00011`
   - Funct3 (AND): `111`
   - Funct7 (AND): `0000000`
   - **Code**: `0000000 00011 00001 111 00010 0110011`

4. **OR r8, r2, r5**
   - Instruction Type: R-Type
   - Opcode (OR): `0110011`
   - rd (r8): `01000`
   - rs1 (r2): `00010`
   - rs2 (r5): `00101`
   - Funct3 (OR): `110`
   - Funct7 (OR): `0000000`
   - **Code**: `0000000 00101 00010 110 01000 0110011`

5. **XOR r8, r1, r4**
   - Instruction Type: R-Type
   - Opcode (XOR): `0110011`
   - rd (r8): `01000`
   - rs1 (r1): `00001`
   - rs2 (r4): `00100`
   - Funct3 (XOR): `100`
   - Funct7 (XOR): `0000000`
   - **Code**: `0000000 00100 00001 100 01000 0110011`

6. **SLT r10, r2, r4**
   - Instruction Type: R-Type
   - Opcode (SLT): `0110011`
   - rd (r10): `01010`
   - rs1 (r2): `00010`
   - rs2 (r4): `00100`
   - Funct3 (SLT): `010`
   - Funct7 (SLT): `0000000`
   - **Code**: `0000000 00100 00010 010 01010 0110011`

7. **ADDI r12, r3, 5**
   - Instruction Type: I-Type
   - Opcode (ADDI): `0010011`
   - rd (r12): `01100`
   - rs1 (r3): `00011`
   - Funct3 (SLT): `000`
   - Imm (11:0): `000000001001`
   - **Code**: `000000001001 00011 000 01100 0010011`

8. **SW r3, r1, 4**
   - Instruction Type: S-Type
   - Opcode (SW): `0100011`
   - rs1 (r1): `00001`
   - rs2 (r3): `00011`
   - Funct3 (SLT): `010`
   - Imm (12|10:5): `0000000`
   - Imm (4:1|11): `00100`
   - **Code**: `0000000 00011 00001 010 00100 0100011`

9. **SRL r16, r11, r2**
   - Instruction Type: R-Type
   - Opcode (SRL): `0110011`
   - rd (r16): `10000`
   - rs1 (r11): `01011`
   - rs2 (r2): `00010`
   - Funct3 (SRL): `101`
   - Funct7 (SRL): `0000000`
   - **Code**: `0000000 00010 01011 101 10000 0110011`

10. **BNE r0, r1, 20**
    - Instruction Type: B-Type
    - Opcode (BNE): `1100011`
    - rs1 (r0): `00000`
    - rs2 (r1): `00001`
    - Funct3 (SRL): `001`
    - Imm (12|10:5): `0000001`
    - Imm (4:1|11): `01000`
    - **Code**: `0000001 00001 00000 001 01000 1100011`

11. **BEQ r0, r0, 15**
    - Instruction Type: B-Type
    - Opcode (BEQ): `1100011`
    - rs1 (r0): `00000`
    - rs2 (r0): `00000`
    - Funct3 (SRL): `000`
    - Imm (12|10:5): `0000000`
    - Imm (4:1|11): `11110`
    - **Code**: `0000000 00000 00000 000 11110 1100011`

12. **LW r13, r11, 2**
    - Instruction Type: I-Type
    - Opcode (LW): `0000011`
    - rd (r13): `01101`
    - rs1 (r11): `01011`
    - Funct3 (SRL): `010`
    - Imm (11:0): `000000000010`
    - **Code**: `000000000010 01011 010 01101 0000011`

13. **SLL r15, r11, r2**
    - Instruction Type: R-Type
    - Opcode (SLL): `0110011`
    - rd (r15): `01111`
    - rs1 (r11): `01011`
    - rs2 (r2): `00010`
    - Funct3 (SLL): `001`
    - Funct7 (SLL): `0000000`
    - **Code**: `0000000 00010 01011 001 01111 0110011`
