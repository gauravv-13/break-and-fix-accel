"""Full RV32I base instruction set simulator with instruction mix accounting and custom AES opcodes.

Supported ISA:
- Full RV32I base integer instruction set (40 user-level instructions):
  - LUI, AUIPC
  - JAL, JALR
  - BEQ, BNE, BLT, BGE, BLTU, BGEU
  - LB, LH, LW, LBU, LHU
  - SB, SH, SW
  - ADDI, SLTI, SLTIU, XORI, ORI, ANDI, SLLI, SRLI, SRAI
  - ADD, SUB, SLL, SLT, SLTU, XOR, SRL, SRA, OR, AND
  - FENCE, ECALL, EBREAK
- Custom / Crypto extensions for Part B comparison:
  - Standard RISC-V Scalar Crypto RV32 Zkne (`aes32esi`, `aes32esmi`), matching `../AES_processor`
  - Custom-0 (`0001011`) 32-bit AES column-round instructions (`aes_round_col`, `aes_final_col`),
    adapted from `../AES_processor/rtl/aes_round64.v`.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict
from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection

from src.aes.aes import SBOX, xtime

PAGE_SIZE = 0x10000
PAGE_MASK = 0xFFFF


def sign_extend(val: int, bits: int) -> int:
    """Sign-extend an integer of `bits` width to a signed Python int."""
    sign_bit = 1 << (bits - 1)
    return (val & (sign_bit - 1)) - (val & sign_bit)


def to_signed32(val: int) -> int:
    """Interpret a 32-bit unsigned integer as a signed 32-bit integer."""
    val &= 0xFFFFFFFF
    return val - 0x100000000 if (val & 0x80000000) else val


def rol32(val: int, shamt: int) -> int:
    """Rotate a 32-bit unsigned integer left by `shamt` bits."""
    shamt &= 31
    val &= 0xFFFFFFFF
    if shamt == 0:
        return val
    return ((val << shamt) | (val >> (32 - shamt))) & 0xFFFFFFFF


def _aes_mix_column_word(word: int) -> int:
    """Apply AES SubBytes and MixColumns to a 32-bit column word (little-endian bytes 0..3)."""
    s0 = SBOX[word & 0xFF]
    s1 = SBOX[(word >> 8) & 0xFF]
    s2 = SBOX[(word >> 16) & 0xFF]
    s3 = SBOX[(word >> 24) & 0xFF]

    t0 = xtime(s0)
    t1 = xtime(s1)
    t2 = xtime(s2)
    t3 = xtime(s3)

    r0 = t0 ^ (t1 ^ s1) ^ s2 ^ s3
    r1 = s0 ^ t1 ^ (t2 ^ s2) ^ s3
    r2 = s0 ^ s1 ^ t2 ^ (t3 ^ s3)
    r3 = (t0 ^ s0) ^ s1 ^ s2 ^ t3
    return (r0 | (r1 << 8) | (r2 << 16) | (r3 << 24)) & 0xFFFFFFFF


def _aes_sub_word(word: int) -> int:
    """Apply AES SubBytes to all 4 bytes of a 32-bit word."""
    s0 = SBOX[word & 0xFF]
    s1 = SBOX[(word >> 8) & 0xFF]
    s2 = SBOX[(word >> 16) & 0xFF]
    s3 = SBOX[(word >> 24) & 0xFF]
    return (s0 | (s1 << 8) | (s2 << 16) | (s3 << 24)) & 0xFFFFFFFF


# Instruction encoding helpers for unit tests and small hand-written programs
def encode_r(opcode: int, rd: int, funct3: int, rs1: int, rs2: int, funct7: int) -> int:
    return (
        ((funct7 & 0x7F) << 25)
        | ((rs2 & 0x1F) << 20)
        | ((rs1 & 0x1F) << 15)
        | ((funct3 & 0x7) << 12)
        | ((rd & 0x1F) << 7)
        | (opcode & 0x7F)
    )


def encode_i(opcode: int, rd: int, funct3: int, rs1: int, imm: int) -> int:
    return (
        ((imm & 0xFFF) << 20)
        | ((rs1 & 0x1F) << 15)
        | ((funct3 & 0x7) << 12)
        | ((rd & 0x1F) << 7)
        | (opcode & 0x7F)
    )


def encode_s(opcode: int, funct3: int, rs1: int, rs2: int, imm: int) -> int:
    imm &= 0xFFF
    imm_11_5 = (imm >> 5) & 0x7F
    imm_4_0 = imm & 0x1F
    return (
        (imm_11_5 << 25)
        | ((rs2 & 0x1F) << 20)
        | ((rs1 & 0x1F) << 15)
        | ((funct3 & 0x7) << 12)
        | (imm_4_0 << 7)
        | (opcode & 0x7F)
    )


def encode_b(opcode: int, funct3: int, rs1: int, rs2: int, imm: int) -> int:
    if imm & 1:
        raise ValueError(f"Branch offset must be 2-byte aligned, got {imm}")
    imm &= 0x1FFF
    b12 = (imm >> 12) & 1
    b11 = (imm >> 11) & 1
    b10_5 = (imm >> 5) & 0x3F
    b4_1 = (imm >> 1) & 0xF
    return (
        (b12 << 31)
        | (b10_5 << 25)
        | ((rs2 & 0x1F) << 20)
        | ((rs1 & 0x1F) << 15)
        | ((funct3 & 0x7) << 12)
        | (b4_1 << 8)
        | (b11 << 7)
        | (opcode & 0x7F)
    )


def encode_u(opcode: int, rd: int, imm20: int) -> int:
    return ((imm20 & 0xFFFFF) << 12) | ((rd & 0x1F) << 7) | (opcode & 0x7F)


def encode_j(opcode: int, rd: int, imm: int) -> int:
    if imm & 1:
        raise ValueError(f"JAL offset must be 2-byte aligned, got {imm}")
    imm &= 0x1FFFFF
    b20 = (imm >> 20) & 1
    b19_12 = (imm >> 12) & 0xFF
    b11 = (imm >> 11) & 1
    b10_1 = (imm >> 1) & 0x3FF
    return (
        (b20 << 31)
        | (b10_1 << 21)
        | (b11 << 20)
        | (b19_12 << 12)
        | ((rd & 0x1F) << 7)
        | (opcode & 0x7F)
    )


class RV32ISimulator:
    """32-bit RISC-V (RV32I + custom AES) instruction-level simulator."""

    def __init__(self, tohost_addr: int | None = 0x8000F000) -> None:
        self.regs: list[int] = [0] * 32
        self.pc: int = 0
        self.pages: Dict[int, bytearray] = {}
        self.symbols: Dict[str, int] = {}
        self.tohost_addr: int | None = tohost_addr
        self.tohost_value: int = 0
        self.halted: bool = False
        self.counts: Dict[str, int] = {
            "alu": 0,
            "load": 0,
            "store": 0,
            "branch": 0,
            "jump": 0,
            "custom": 0,
            "system": 0,
        }

    def reset_counts(self) -> None:
        """Reset instruction execution counters to zero."""
        for k in self.counts:
            self.counts[k] = 0

    @property
    def total_instructions(self) -> int:
        """Return total executed instructions across all categories."""
        return sum(self.counts.values())

    def estimate_cycles(self, cpi: float = 1.0) -> float:
        """Estimate processor clock cycles for a given average CPI."""
        return self.total_instructions * cpi

    # ------------------------------------------------------------------
    # Byte-addressable little-endian memory
    # ------------------------------------------------------------------
    def _get_page(self, page_id: int, create: bool = False) -> bytearray | None:
        page = self.pages.get(page_id)
        if page is None and create:
            page = bytearray(PAGE_SIZE)
            self.pages[page_id] = page
        return page

    def read_u8(self, addr: int) -> int:
        addr &= 0xFFFFFFFF
        page = self.pages.get(addr >> 16)
        if page is None:
            return 0
        return page[addr & PAGE_MASK]

    def write_u8(self, addr: int, val: int) -> None:
        addr &= 0xFFFFFFFF
        page = self._get_page(addr >> 16, create=True)
        assert page is not None
        page[addr & PAGE_MASK] = val & 0xFF

    def read_u16(self, addr: int) -> int:
        addr &= 0xFFFFFFFF
        off = addr & PAGE_MASK
        if off <= PAGE_SIZE - 2:
            page = self.pages.get(addr >> 16)
            if page is None:
                return 0
            return page[off] | (page[off + 1] << 8)
        return self.read_u8(addr) | (self.read_u8(addr + 1) << 8)

    def write_u16(self, addr: int, val: int) -> None:
        addr &= 0xFFFFFFFF
        val &= 0xFFFF
        off = addr & PAGE_MASK
        if off <= PAGE_SIZE - 2:
            page = self._get_page(addr >> 16, create=True)
            assert page is not None
            page[off] = val & 0xFF
            page[off + 1] = (val >> 8) & 0xFF
        else:
            self.write_u8(addr, val & 0xFF)
            self.write_u8(addr + 1, (val >> 8) & 0xFF)

    def read_u32(self, addr: int) -> int:
        addr &= 0xFFFFFFFF
        off = addr & PAGE_MASK
        if off <= PAGE_SIZE - 4:
            page = self.pages.get(addr >> 16)
            if page is None:
                return 0
            return (
                page[off]
                | (page[off + 1] << 8)
                | (page[off + 2] << 16)
                | (page[off + 3] << 24)
            )
        return (
            self.read_u8(addr)
            | (self.read_u8(addr + 1) << 8)
            | (self.read_u8(addr + 2) << 16)
            | (self.read_u8(addr + 3) << 24)
        )

    def write_u32(self, addr: int, val: int) -> None:
        addr &= 0xFFFFFFFF
        val &= 0xFFFFFFFF
        if self.tohost_addr is not None and addr == (self.tohost_addr & 0xFFFFFFFF) and val != 0:
            self.tohost_value = val
            self.halted = True
        off = addr & PAGE_MASK
        if off <= PAGE_SIZE - 4:
            page = self._get_page(addr >> 16, create=True)
            assert page is not None
            page[off] = val & 0xFF
            page[off + 1] = (val >> 8) & 0xFF
            page[off + 2] = (val >> 16) & 0xFF
            page[off + 3] = (val >> 24) & 0xFF
        else:
            self.write_u8(addr, val & 0xFF)
            self.write_u8(addr + 1, (val >> 8) & 0xFF)
            self.write_u8(addr + 2, (val >> 16) & 0xFF)
            self.write_u8(addr + 3, (val >> 24) & 0xFF)

    def write_bytes(self, addr: int, data: bytes | bytearray) -> None:
        """Write a contiguous byte buffer into memory starting at `addr`."""
        for i, b in enumerate(data):
            self.write_u8(addr + i, b)

    def read_bytes(self, addr: int, length: int) -> bytes:
        """Read `length` bytes from memory starting at `addr`."""
        return bytes(self.read_u8(addr + i) for i in range(length))

    def load_program(self, words: list[int], base_addr: int = 0x00000000) -> None:
        """Load a list of 32-bit instruction words into memory at `base_addr`."""
        for i, w in enumerate(words):
            self.write_u32(base_addr + 4 * i, w)
        self.pc = base_addr & 0xFFFFFFFF
        self.halted = False

    def load_elf(self, elf_path: str | Path) -> int:
        """Load an RV32 ELF file using `pyelftools` and set PC to entry point."""
        with open(elf_path, "rb") as f:
            elf = ELFFile(f)
            if elf.elfclass != 32:
                raise ValueError(f"Expected 32-bit ELF (RV32), got {elf.elfclass}-bit ELF")

            for segment in elf.iter_segments():
                if segment["p_type"] == "PT_LOAD":
                    vaddr = int(segment["p_vaddr"])
                    data = segment.data()
                    memsz = int(segment["p_memsz"])
                    self.write_bytes(vaddr, data)
                    if memsz > len(data):
                        for z in range(len(data), memsz):
                            self.write_u8(vaddr + z, 0)

            self.symbols.clear()
            for section in elf.iter_sections():
                if isinstance(section, SymbolTableSection):
                    for sym in section.iter_symbols():
                        if sym.name:
                            self.symbols[sym.name] = int(sym["st_value"])

            if "tohost" in self.symbols:
                self.tohost_addr = self.symbols["tohost"]

            entry = int(elf.header["e_entry"]) & 0xFFFFFFFF
            self.pc = entry
            self.halted = False
            return entry

    # ------------------------------------------------------------------
    # Single-step execution
    # ------------------------------------------------------------------
    def step(self) -> bool:
        """Fetch, decode, and execute one 32-bit RV32I instruction.

        Returns True if the processor is still running, or False if halted.
        """
        if self.halted:
            return False

        instr = self.read_u32(self.pc)
        opcode = instr & 0x7F
        rd = (instr >> 7) & 0x1F
        funct3 = (instr >> 12) & 0x7
        rs1 = (instr >> 15) & 0x1F
        rs2 = (instr >> 20) & 0x1F
        funct7 = (instr >> 25) & 0x7F

        v_rs1 = self.regs[rs1]
        v_rs2 = self.regs[rs2]
        next_pc = (self.pc + 4) & 0xFFFFFFFF

        if opcode == 0x37:  # LUI
            self.counts["alu"] += 1
            if rd != 0:
                self.regs[rd] = instr & 0xFFFFF000

        elif opcode == 0x17:  # AUIPC
            self.counts["alu"] += 1
            if rd != 0:
                self.regs[rd] = (self.pc + (instr & 0xFFFFF000)) & 0xFFFFFFFF

        elif opcode == 0x6F:  # JAL
            self.counts["jump"] += 1
            b20 = (instr >> 31) & 1
            b10_1 = (instr >> 21) & 0x3FF
            b11 = (instr >> 20) & 1
            b19_12 = (instr >> 12) & 0xFF
            imm_j = sign_extend((b20 << 20) | (b19_12 << 12) | (b11 << 11) | (b10_1 << 1), 21)
            if rd != 0:
                self.regs[rd] = next_pc
            next_pc = (self.pc + imm_j) & 0xFFFFFFFF

        elif opcode == 0x67:  # JALR
            if funct3 != 0:
                raise ValueError(f"Illegal JALR funct3={funct3} at PC=0x{self.pc:08x}")
            self.counts["jump"] += 1
            imm_i = sign_extend(instr >> 20, 12)
            target = ((v_rs1 + imm_i) & ~1) & 0xFFFFFFFF
            if rd != 0:
                self.regs[rd] = next_pc
            next_pc = target

        elif opcode == 0x63:  # BRANCH
            self.counts["branch"] += 1
            b12 = (instr >> 31) & 1
            b10_5 = (instr >> 25) & 0x3F
            b4_1 = (instr >> 8) & 0xF
            b11 = (instr >> 7) & 1
            imm_b = sign_extend((b12 << 12) | (b11 << 11) | (b10_5 << 5) | (b4_1 << 1), 13)

            if funct3 == 0x0:    # BEQ
                taken = v_rs1 == v_rs2
            elif funct3 == 0x1:  # BNE
                taken = v_rs1 != v_rs2
            elif funct3 == 0x4:  # BLT
                taken = to_signed32(v_rs1) < to_signed32(v_rs2)
            elif funct3 == 0x5:  # BGE
                taken = to_signed32(v_rs1) >= to_signed32(v_rs2)
            elif funct3 == 0x6:  # BLTU
                taken = v_rs1 < v_rs2
            elif funct3 == 0x7:  # BGEU
                taken = v_rs1 >= v_rs2
            else:
                raise ValueError(f"Illegal BRANCH funct3={funct3} at PC=0x{self.pc:08x}")

            if taken:
                next_pc = (self.pc + imm_b) & 0xFFFFFFFF

        elif opcode == 0x03:  # LOAD
            self.counts["load"] += 1
            imm_i = sign_extend(instr >> 20, 12)
            addr = (v_rs1 + imm_i) & 0xFFFFFFFF

            if funct3 == 0x0:    # LB
                val = sign_extend(self.read_u8(addr), 8) & 0xFFFFFFFF
            elif funct3 == 0x1:  # LH
                val = sign_extend(self.read_u16(addr), 16) & 0xFFFFFFFF
            elif funct3 == 0x2:  # LW
                val = self.read_u32(addr)
            elif funct3 == 0x4:  # LBU
                val = self.read_u8(addr)
            elif funct3 == 0x5:  # LHU
                val = self.read_u16(addr)
            else:
                raise ValueError(f"Illegal LOAD funct3={funct3} at PC=0x{self.pc:08x}")

            if rd != 0:
                self.regs[rd] = val

        elif opcode == 0x23:  # STORE
            self.counts["store"] += 1
            imm_s = sign_extend((funct7 << 5) | rd, 12)
            addr = (v_rs1 + imm_s) & 0xFFFFFFFF

            if funct3 == 0x0:    # SB
                self.write_u8(addr, v_rs2)
            elif funct3 == 0x1:  # SH
                self.write_u16(addr, v_rs2)
            elif funct3 == 0x2:  # SW
                self.write_u32(addr, v_rs2)
            else:
                raise ValueError(f"Illegal STORE funct3={funct3} at PC=0x{self.pc:08x}")

        elif opcode == 0x13:  # OP-IMM
            self.counts["alu"] += 1
            imm_i = sign_extend(instr >> 20, 12)
            uimm_i = imm_i & 0xFFFFFFFF
            shamt = rs2

            if funct3 == 0x0:    # ADDI
                res = (v_rs1 + imm_i) & 0xFFFFFFFF
            elif funct3 == 0x2:  # SLTI
                res = 1 if to_signed32(v_rs1) < imm_i else 0
            elif funct3 == 0x3:  # SLTIU
                res = 1 if v_rs1 < uimm_i else 0
            elif funct3 == 0x4:  # XORI
                res = (v_rs1 ^ uimm_i) & 0xFFFFFFFF
            elif funct3 == 0x6:  # ORI
                res = (v_rs1 | uimm_i) & 0xFFFFFFFF
            elif funct3 == 0x7:  # ANDI
                res = (v_rs1 & uimm_i) & 0xFFFFFFFF
            elif funct3 == 0x1:  # SLLI
                if funct7 != 0x00:
                    raise ValueError(f"Illegal SLLI funct7=0x{funct7:02x} at PC=0x{self.pc:08x}")
                res = (v_rs1 << shamt) & 0xFFFFFFFF
            elif funct3 == 0x5:  # SRLI / SRAI
                if funct7 == 0x00:    # SRLI
                    res = (v_rs1 >> shamt) & 0xFFFFFFFF
                elif funct7 == 0x20:  # SRAI
                    res = (to_signed32(v_rs1) >> shamt) & 0xFFFFFFFF
                else:
                    raise ValueError(f"Illegal SRLI/SRAI funct7=0x{funct7:02x} at PC=0x{self.pc:08x}")
            else:
                raise ValueError(f"Illegal OP-IMM funct3={funct3} at PC=0x{self.pc:08x}")

            if rd != 0:
                self.regs[rd] = res

        elif opcode == 0x33:  # OP (and RV32 Zkne instructions aes32esi / aes32esmi)
            # Check RV32 Zkne first: funct3 == 0 and funct7[4:0] in (0x11, 0x13)
            zkne_tag = funct7 & 0x1F
            if funct3 == 0x0 and zkne_tag in (0x11, 0x13):
                self.counts["custom"] += 1
                bs = (funct7 >> 5) & 0x3
                shamt = bs * 8
                si = (v_rs2 >> shamt) & 0xFF
                so = SBOX[si]
                if zkne_tag == 0x11:  # aes32esi (final round byte S-box + ShiftRows + XOR)
                    val = so
                else:                 # aes32esmi (middle round byte S-box + MixColumns + XOR)
                    m2 = xtime(so)
                    m3 = m2 ^ so
                    val = (m3 << 24) | (so << 16) | (so << 8) | m2
                res = (v_rs1 ^ rol32(val, shamt)) & 0xFFFFFFFF
            else:
                self.counts["alu"] += 1
                shamt = v_rs2 & 0x1F
                if funct3 == 0x0 and funct7 == 0x00:    # ADD
                    res = (v_rs1 + v_rs2) & 0xFFFFFFFF
                elif funct3 == 0x0 and funct7 == 0x20:  # SUB
                    res = (v_rs1 - v_rs2) & 0xFFFFFFFF
                elif funct3 == 0x1 and funct7 == 0x00:  # SLL
                    res = (v_rs1 << shamt) & 0xFFFFFFFF
                elif funct3 == 0x2 and funct7 == 0x00:  # SLT
                    res = 1 if to_signed32(v_rs1) < to_signed32(v_rs2) else 0
                elif funct3 == 0x3 and funct7 == 0x00:  # SLTU
                    res = 1 if v_rs1 < v_rs2 else 0
                elif funct3 == 0x4 and funct7 == 0x00:  # XOR
                    res = (v_rs1 ^ v_rs2) & 0xFFFFFFFF
                elif funct3 == 0x5 and funct7 == 0x00:  # SRL
                    res = (v_rs1 >> shamt) & 0xFFFFFFFF
                elif funct3 == 0x5 and funct7 == 0x20:  # SRA
                    res = (to_signed32(v_rs1) >> shamt) & 0xFFFFFFFF
                elif funct3 == 0x6 and funct7 == 0x00:  # OR
                    res = (v_rs1 | v_rs2) & 0xFFFFFFFF
                elif funct3 == 0x7 and funct7 == 0x00:  # AND
                    res = (v_rs1 & v_rs2) & 0xFFFFFFFF
                else:
                    raise ValueError(
                        f"Illegal OP funct3={funct3} funct7=0x{funct7:02x} at PC=0x{self.pc:08x}"
                    )

            if rd != 0:
                self.regs[rd] = res

        elif opcode == 0x0B:  # custom-0 (0001011): Custom AES round instructions (cf. aes_round64.v)
            self.counts["custom"] += 1
            if funct7 != 0x00:
                raise ValueError(f"Illegal custom-0 funct7=0x{funct7:02x} at PC=0x{self.pc:08x}")
            if funct3 == 0x0:    # aes_round_col: rd = MixColumns(SubBytes(rs1)) ^ rs2
                res = _aes_mix_column_word(v_rs1) ^ v_rs2
            elif funct3 == 0x1:  # aes_final_col: rd = SubBytes(rs1) ^ rs2
                res = _aes_sub_word(v_rs1) ^ v_rs2
            elif funct3 == 0x2:  # aes_shift_even: pack row 0 from rs1 and row 2 from rs2
                res = (v_rs1 & 0x0000FFFF) | (v_rs2 & 0xFFFF0000)
            elif funct3 == 0x3:  # aes_shift_diag: pack bytes [rs1[7:0], rs2[15:8], rs1[23:16], rs2[31:24]]
                res = (v_rs1 & 0x00FF00FF) | (v_rs2 & 0xFF00FF00)
            else:
                raise ValueError(f"Illegal custom-0 funct3={funct3} at PC=0x{self.pc:08x}")

            if rd != 0:
                self.regs[rd] = res & 0xFFFFFFFF

        elif opcode == 0x0F:  # FENCE
            self.counts["alu"] += 1

        elif opcode == 0x73:  # SYSTEM (ECALL / EBREAK)
            self.counts["system"] += 1
            self.halted = True
            self.pc = next_pc
            return False

        else:
            raise ValueError(
                f"Illegal instruction 0x{instr:08x} (opcode=0x{opcode:02x}) at PC=0x{self.pc:08x}"
            )

        self.regs[0] = 0
        self.pc = next_pc
        return not self.halted

    def run(self, max_steps: int = 2_000_000) -> int:
        """Execute instructions until halted or `max_steps` is reached."""
        steps = 0
        while not self.halted and steps < max_steps:
            self.step()
            steps += 1
        if not self.halted:
            raise RuntimeError(f"Simulation exceeded max_steps={max_steps} at PC=0x{self.pc:08x}")
        return steps
