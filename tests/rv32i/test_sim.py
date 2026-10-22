"""Comprehensive unit tests for the RV32I instruction-set simulator."""

from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.rv32i.sim import (
    RV32ISimulator,
    encode_b,
    encode_i,
    encode_j,
    encode_r,
    encode_s,
    encode_u,
)

OP_LUI = 0x37
OP_AUIPC = 0x17
OP_JAL = 0x6F
OP_JALR = 0x67
OP_BRANCH = 0x63
OP_LOAD = 0x03
OP_STORE = 0x23
OP_IMM = 0x13
OP_OP = 0x33
OP_FENCE = 0x0F
OP_SYSTEM = 0x73

EBREAK = encode_i(OP_SYSTEM, 0, 0x0, 0, 1)
ECALL = encode_i(OP_SYSTEM, 0, 0x0, 0, 0)


def test_x0_always_zero() -> None:
    """Verify x0 cannot be modified by ALU, LUI, LOAD, JAL, or JALR."""
    sim = RV32ISimulator()
    sim.write_u32(0x100, 0xDEADBEEF)
    prog = [
        encode_i(OP_IMM, 0, 0x0, 0, 123),      # addi x0, x0, 123
        encode_u(OP_LUI, 0, 0xABCDE),          # lui x0, 0xabcde
        encode_i(OP_LOAD, 0, 0x2, 0, 0x100),   # lw x0, 0x100(x0)
        encode_j(OP_JAL, 0, 8),                # jal x0, +8
        encode_i(OP_IMM, 1, 0x0, 0, 999),      # skipped!
        encode_i(OP_IMM, 1, 0x0, 0, 0x20),     # addi x1, x0, 0x20
        encode_i(OP_JALR, 0, 0x0, 1, 0),       # jalr x0, 0(x1) -> PC = 0x20
    ]
    sim.load_program(prog, base_addr=0)
    sim.write_u32(0x20, encode_r(OP_OP, 0, 0x0, 1, 1, 0x00))  # add x0, x1, x1
    sim.write_u32(0x24, EBREAK)
    sim.run()

    assert sim.regs[0] == 0
    assert sim.regs[1] == 0x20


def test_lui_auipc_and_immediate_sign_extension() -> None:
    """Test LUI, AUIPC, and 12-bit signed immediate extension on ADDI, SLTI, SLTIU, XORI, ORI, ANDI."""
    sim = RV32ISimulator()
    prog = [
        encode_u(OP_LUI, 1, 0x12345),          # x1 = 0x12345000
        encode_u(OP_AUIPC, 2, 0x00001),        # x2 = 0x1004 + 0x1000 = 0x2004
        encode_i(OP_IMM, 3, 0x0, 0, -1),       # addi x3, x0, -1 -> 0xFFFFFFFF
        encode_i(OP_IMM, 4, 0x0, 0, -2048),    # addi x4, x0, -2048 -> 0xFFFFF800
        encode_i(OP_IMM, 5, 0x2, 3, 0),        # slti x5, x3, 0 -> (-1 < 0) = 1
        encode_i(OP_IMM, 6, 0x3, 3, -1),       # sltiu x6, x3, -1 -> (0xFFFFFFFF < 0xFFFFFFFF) = 0
        encode_i(OP_IMM, 7, 0x3, 0, -1),       # sltiu x7, x0, -1 -> (0 < 0xFFFFFFFF) = 1
        encode_i(OP_IMM, 8, 0x4, 0, -1),       # xori x8, x0, -1 -> 0xFFFFFFFF
        encode_i(OP_IMM, 9, 0x6, 0, -16),      # ori x9, x0, -16 -> 0xFFFFFFF0
        encode_i(OP_IMM, 10, 0x7, 8, 0x0FF),   # andi x10, x8, 0x0FF -> 0x000000FF
        EBREAK,
    ]
    sim.load_program(prog, base_addr=0x1000)
    sim.run()

    assert sim.regs[1] == 0x12345000
    assert sim.regs[2] == 0x00002004
    assert sim.regs[3] == 0xFFFFFFFF
    assert sim.regs[4] == 0xFFFFF800
    assert sim.regs[5] == 1
    assert sim.regs[6] == 0
    assert sim.regs[7] == 1
    assert sim.regs[8] == 0xFFFFFFFF
    assert sim.regs[9] == 0xFFFFFFF0
    assert sim.regs[10] == 0x000000FF
    assert sim.counts["alu"] == 10


def test_shifts_logical_and_arithmetic() -> None:
    """Test SLLI, SRLI, SRAI, SLL, SRL, SRA including sign-bit preservation and 5-bit shamt mask."""
    sim = RV32ISimulator()
    prog = [
        encode_u(OP_LUI, 1, 0x80000),          # x1 = 0x80000000 (-2147483648)
        encode_i(OP_IMM, 2, 0x5, 1, 4),        # srli x2, x1, 4 -> 0x08000000
        encode_i(OP_IMM, 3, 0x5, 1, (0x20 << 5) | 4),  # srai x3, x1, 4 -> 0xF8000000
        encode_i(OP_IMM, 4, 0x1, 2, 4),        # slli x4, x2, 4 -> 0x80000000
        encode_i(OP_IMM, 5, 0x0, 0, 36),       # x5 = 36 (36 & 31 == 4)
        encode_r(OP_OP, 6, 0x1, 2, 5, 0x00),   # sll x6, x2, x5 -> shift by 4 -> 0x80000000
        encode_r(OP_OP, 7, 0x5, 1, 5, 0x00),   # srl x7, x1, x5 -> 0x08000000
        encode_r(OP_OP, 8, 0x5, 1, 5, 0x20),   # sra x8, x1, x5 -> 0xF8000000
        EBREAK,
    ]
    sim.load_program(prog)
    sim.run()

    assert sim.regs[2] == 0x08000000
    assert sim.regs[3] == 0xF8000000
    assert sim.regs[4] == 0x80000000
    assert sim.regs[6] == 0x80000000
    assert sim.regs[7] == 0x08000000
    assert sim.regs[8] == 0xF8000000


def test_register_alu_ops() -> None:
    """Test ADD, SUB, SLT, SLTU, XOR, OR, AND, FENCE."""
    sim = RV32ISimulator()
    prog = [
        encode_i(OP_IMM, 1, 0x0, 0, -10),      # x1 = -10 (0xFFFFFFF6)
        encode_i(OP_IMM, 2, 0x0, 0, 25),       # x2 = 25
        encode_r(OP_OP, 3, 0x0, 1, 2, 0x00),   # add x3, x1, x2 -> 15
        encode_r(OP_OP, 4, 0x0, 1, 2, 0x20),   # sub x4, x1, x2 -> -35 (0xFFFFFFDD)
        encode_r(OP_OP, 5, 0x2, 1, 2, 0x00),   # slt x5, x1, x2 -> (-10 < 25) = 1
        encode_r(OP_OP, 6, 0x3, 1, 2, 0x00),   # sltu x6, x1, x2 -> (0xFFFFFFF6 < 25) = 0
        encode_r(OP_OP, 7, 0x4, 1, 2, 0x00),   # xor x7, x1, x2
        encode_r(OP_OP, 8, 0x6, 1, 2, 0x00),   # or x8, x1, x2
        encode_r(OP_OP, 9, 0x7, 1, 2, 0x00),   # and x9, x1, x2
        encode_i(OP_FENCE, 0, 0x0, 0, 0),      # fence
        ECALL,
    ]
    sim.load_program(prog)
    sim.run()

    assert sim.regs[3] == 15
    assert sim.regs[4] == (-35 & 0xFFFFFFFF)
    assert sim.regs[5] == 1
    assert sim.regs[6] == 0
    assert sim.regs[7] == ((0xFFFFFFF6 ^ 25) & 0xFFFFFFFF)
    assert sim.regs[8] == ((0xFFFFFFF6 | 25) & 0xFFFFFFFF)
    assert sim.regs[9] == (0xFFFFFFF6 & 25)


def test_loads_and_stores_sign_extension_and_endianness() -> None:
    """Test SB, SH, SW, LB, LBU, LH, LHU, LW with negative offsets and sign extension."""
    sim = RV32ISimulator()
    prog = [
        encode_i(OP_IMM, 1, 0x0, 0, 0x200),    # x1 = base address 0x200
        encode_u(OP_LUI, 2, 0x89ABC),          # x2 = 0x89ABC000
        encode_i(OP_IMM, 2, 0x6, 2, 0x7EF),    # x2 = 0x89ABC7EF
        encode_s(OP_STORE, 0x2, 1, 2, 0),      # sw x2, 0(x1)
        encode_i(OP_LOAD, 3, 0x0, 1, 0),       # lb x3, 0(x1) -> 0xEF sign-ext = 0xFFFFFFEF
        encode_i(OP_LOAD, 4, 0x4, 1, 0),       # lbu x4, 0(x1) -> 0x000000EF
        encode_i(OP_LOAD, 5, 0x1, 1, 2),       # lh x5, 2(x1) -> 0x89AB sign-ext = 0xFFFF89AB
        encode_i(OP_LOAD, 6, 0x5, 1, 2),       # lhu x6, 2(x1) -> 0x000089AB
        # Store byte and halfword with negative offset from x1 + 16
        encode_i(OP_IMM, 7, 0x0, 1, 16),       # x7 = 0x210
        encode_i(OP_IMM, 8, 0x0, 0, 0x81),     # x8 = 0x81
        encode_s(OP_STORE, 0x0, 7, 8, -4),     # sb x8, -4(x7) -> addr 0x20C
        encode_i(OP_LOAD, 9, 0x0, 7, -4),      # lb x9, -4(x7) -> 0xFFFFFF81
        encode_i(OP_IMM, 10, 0x0, 0, -300),    # x10 = 0xFFFFFED4
        encode_s(OP_STORE, 0x1, 7, 10, -2),    # sh x10, -2(x7) -> addr 0x20E
        encode_i(OP_LOAD, 11, 0x1, 7, -2),     # lh x11, -2(x7) -> 0xFFFFFED4
        encode_i(OP_LOAD, 12, 0x2, 1, 0),      # lw x12, 0(x1) -> 0x89ABC7EF
        EBREAK,
    ]
    sim.load_program(prog)
    sim.run()

    assert sim.regs[3] == 0xFFFFFFEF
    assert sim.regs[4] == 0x000000EF
    assert sim.regs[5] == 0xFFFF89AB
    assert sim.regs[6] == 0x000089AB
    assert sim.regs[9] == 0xFFFFFF81
    assert sim.regs[11] == 0xFFFFFED4
    assert sim.regs[12] == 0x89ABC7EF
    assert sim.counts["load"] == 7
    assert sim.counts["store"] == 3


def test_all_branches_and_backward_loop() -> None:
    """Test BEQ, BNE, BLT, BGE, BLTU, BGEU including backward branch loop."""
    sim = RV32ISimulator()
    prog = [
        encode_i(OP_IMM, 1, 0x0, 0, -5),       # 0x00: x1 = -5 (0xFFFFFFFB)
        encode_i(OP_IMM, 2, 0x0, 0, 5),        # 0x04: x2 = 5
        encode_b(OP_BRANCH, 0x0, 1, 2, 8),     # 0x08: beq x1, x2, +8 (not taken)
        encode_b(OP_BRANCH, 0x1, 1, 2, 8),     # 0x0C: bne x1, x2, +8 (taken -> 0x14)
        encode_i(OP_IMM, 30, 0x0, 0, 1),       # 0x10: skipped
        encode_b(OP_BRANCH, 0x4, 1, 2, 8),     # 0x14: blt x1, x2, +8 (-5 < 5, taken -> 0x1C)
        encode_i(OP_IMM, 30, 0x0, 0, 2),       # 0x18: skipped
        encode_b(OP_BRANCH, 0x5, 2, 1, 8),     # 0x1C: bge x2, x1, +8 (5 >= -5, taken -> 0x24)
        encode_i(OP_IMM, 30, 0x0, 0, 3),       # 0x20: skipped
        encode_b(OP_BRANCH, 0x6, 2, 1, 8),     # 0x24: bltu x2, x1, +8 (5 < 0xFFFFFFFB, taken -> 0x2C)
        encode_i(OP_IMM, 30, 0x0, 0, 4),       # 0x28: skipped
        encode_b(OP_BRANCH, 0x7, 1, 2, 8),     # 0x2C: bgeu x1, x2, +8 (0xFFFFFFFB >= 5, taken -> 0x34)
        encode_i(OP_IMM, 30, 0x0, 0, 5),       # 0x30: skipped
        # Backward loop summing 1..5 into x10
        encode_i(OP_IMM, 10, 0x0, 0, 0),       # 0x34: x10 = 0
        encode_i(OP_IMM, 11, 0x0, 0, 5),       # 0x38: x11 = 5
        # loop (0x3C):
        encode_r(OP_OP, 10, 0x0, 10, 11, 0x00),  # 0x3C: add x10, x10, x11
        encode_i(OP_IMM, 11, 0x0, 11, -1),       # 0x40: addi x11, x11, -1
        encode_b(OP_BRANCH, 0x1, 11, 0, -8),     # 0x44: bne x11, x0, -8 (back to 0x3C)
        EBREAK,                                  # 0x48
    ]
    sim.load_program(prog)
    sim.run()

    assert sim.regs[30] == 0
    assert sim.regs[10] == 15
    assert sim.regs[11] == 0
    assert sim.counts["branch"] == 11


def test_jal_and_jalr_lsb_clearing() -> None:
    """Test JAL and JALR link address saving and JALR LSB clearing."""
    sim = RV32ISimulator()
    prog = [
        encode_j(OP_JAL, 1, 12),               # 0x00: jal x1, +12 -> x1 = 0x04, PC = 0x0C
        encode_i(OP_IMM, 2, 0x0, 0, 99),       # 0x04: x2 = 99 (executed after return)
        EBREAK,                                # 0x08
        encode_i(OP_IMM, 3, 0x0, 1, 1),        # 0x0C: x3 = x1 + 1 = 0x05 (odd address!)
        encode_i(OP_JALR, 4, 0x0, 3, 0),       # 0x10: jalr x4, 0(x3) -> PC = 0x05 & ~1 = 0x04, x4 = 0x14
    ]
    sim.load_program(prog)
    sim.run()

    assert sim.regs[1] == 0x04
    assert sim.regs[2] == 99
    assert sim.regs[4] == 0x14
    assert sim.counts["jump"] == 2
