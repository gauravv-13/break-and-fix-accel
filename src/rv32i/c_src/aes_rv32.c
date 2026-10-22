/*
 * Bare-metal RV32I AES-128 implementations compiled with:
 *   riscv64-unknown-elf-gcc -march=rv32i -mabi=ilp32 -O2
 *
 * Includes:
 *   1. Table-free AES-128 (S-box + explicit ShiftRows + GF(2^8) xtime MixColumns)
 *   2. T-table AES-128 (32-bit Te0..Te3 lookup tables fusing SubBytes+ShiftRows+MixColumns)
 *   3. RV32 Zkne AES-128 (ratified Scalar Crypto `aes32esmi` / `aes32esi` instructions)
 *   4. Custom-0 AES-128 (`0x0B` custom opcode modeled on ../AES_processor/rtl/aes_round64.v)
 */

#include <stdint.h>

static const uint8_t SBOX[256] = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
};

static const uint8_t RCON[10] = {
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36
};

/* Precomputed little-endian T-table Te0: byte 0 = 2*S[x], byte 1 = S[x], byte 2 = S[x], byte 3 = 3*S[x] */
#define GF2(x) ((((x) << 1) ^ ((((x) >> 7) & 1) * 0x1b)) & 0xff)
#define TE0_ENTRY(s) ((uint32_t)(GF2(s)) | ((uint32_t)(s) << 8) | ((uint32_t)(s) << 16) | ((uint32_t)(GF2(s) ^ (s)) << 24))
#define ROTL8(w)  (((w) << 8) | ((w) >> 24))
#define ROTL16(w) (((w) << 16) | ((w) >> 16))
#define ROTL24(w) (((w) << 24) | ((w) >> 8))

static uint32_t Te0[256];
static uint32_t Te1[256];
static uint32_t Te2[256];
static uint32_t Te3[256];

void __attribute__((noinline)) init_ttables(void) {
    for (int i = 0; i < 256; i++) {
        uint32_t s = SBOX[i];
        uint32_t s2 = ((s << 1) ^ ((s & 0x80) ? 0x1b : 0x00)) & 0xff;
        uint32_t s3 = s2 ^ s;
        uint32_t w = s2 | (s << 8) | (s << 16) | (s3 << 24);
        Te0[i] = w;
        Te1[i] = ROTL8(w);
        Te2[i] = ROTL16(w);
        Te3[i] = ROTL24(w);
    }
}

/* Expand 16-byte key into 176-byte round key buffer (44 words) */
void __attribute__((noinline)) aes128_expand_key(const uint8_t *key, uint8_t *rk) {
    for (int i = 0; i < 16; i++) {
        rk[i] = key[i];
    }
    for (int i = 4; i < 44; i++) {
        uint8_t t0 = rk[4 * (i - 1) + 0];
        uint8_t t1 = rk[4 * (i - 1) + 1];
        uint8_t t2 = rk[4 * (i - 1) + 2];
        uint8_t t3 = rk[4 * (i - 1) + 3];
        if ((i & 3) == 0) {
            uint8_t u0 = SBOX[t1] ^ RCON[(i >> 2) - 1];
            uint8_t u1 = SBOX[t2];
            uint8_t u2 = SBOX[t3];
            uint8_t u3 = SBOX[t0];
            t0 = u0; t1 = u1; t2 = u2; t3 = u3;
        }
        rk[4 * i + 0] = rk[4 * (i - 4) + 0] ^ t0;
        rk[4 * i + 1] = rk[4 * (i - 4) + 1] ^ t1;
        rk[4 * i + 2] = rk[4 * (i - 4) + 2] ^ t2;
        rk[4 * i + 3] = rk[4 * (i - 4) + 3] ^ t3;
    }
}

static inline uint8_t xtime_u8(uint8_t x) {
    uint32_t hi = (uint32_t)(x >> 7);
    return (uint8_t)((x << 1) ^ ((0U - hi) & 0x1bU));
}

/* 1. Table-free AES-128 block encryption (16-byte block, pre-expanded round keys) */
void __attribute__((noinline)) aes128_tablefree_block(
    uint8_t *out,
    const uint8_t *in,
    const uint8_t *rk
) {
    uint8_t s[16];
    for (int i = 0; i < 16; i++) {
        s[i] = in[i] ^ rk[i];
    }

    for (int round = 1; round <= 9; round++) {
        const uint8_t *k = rk + (round << 4);
        /* SubBytes */
        uint8_t b0  = SBOX[s[0]],  b1  = SBOX[s[1]],  b2  = SBOX[s[2]],  b3  = SBOX[s[3]];
        uint8_t b4  = SBOX[s[4]],  b5  = SBOX[s[5]],  b6  = SBOX[s[6]],  b7  = SBOX[s[7]];
        uint8_t b8  = SBOX[s[8]],  b9  = SBOX[s[9]],  b10 = SBOX[s[10]], b11 = SBOX[s[11]];
        uint8_t b12 = SBOX[s[12]], b13 = SBOX[s[13]], b14 = SBOX[s[14]], b15 = SBOX[s[15]];

        /* ShiftRows produces columns:
         * col 0: b0,  b5,  b10, b15
         * col 1: b4,  b9,  b14, b3
         * col 2: b8,  b13, b2,  b7
         * col 3: b12, b1,  b6,  b11
         */
        uint8_t c[16] = {
            b0,  b5,  b10, b15,
            b4,  b9,  b14, b3,
            b8,  b13, b2,  b7,
            b12, b1,  b6,  b11
        };

        /* MixColumns + AddRoundKey */
        for (int col = 0; col < 4; col++) {
            int idx = col << 2;
            uint8_t a0 = c[idx + 0];
            uint8_t a1 = c[idx + 1];
            uint8_t a2 = c[idx + 2];
            uint8_t a3 = c[idx + 3];
            uint8_t t = a0 ^ a1 ^ a2 ^ a3;
            s[idx + 0] = a0 ^ t ^ xtime_u8(a0 ^ a1) ^ k[idx + 0];
            s[idx + 1] = a1 ^ t ^ xtime_u8(a1 ^ a2) ^ k[idx + 1];
            s[idx + 2] = a2 ^ t ^ xtime_u8(a2 ^ a3) ^ k[idx + 2];
            s[idx + 3] = a3 ^ t ^ xtime_u8(a3 ^ a0) ^ k[idx + 3];
        }
    }

    /* Final round (Round 10: SubBytes + ShiftRows + AddRoundKey) */
    const uint8_t *k10 = rk + 160;
    out[0]  = SBOX[s[0]]  ^ k10[0];
    out[1]  = SBOX[s[5]]  ^ k10[1];
    out[2]  = SBOX[s[10]] ^ k10[2];
    out[3]  = SBOX[s[15]] ^ k10[3];

    out[4]  = SBOX[s[4]]  ^ k10[4];
    out[5]  = SBOX[s[9]]  ^ k10[5];
    out[6]  = SBOX[s[14]] ^ k10[6];
    out[7]  = SBOX[s[3]]  ^ k10[7];

    out[8]  = SBOX[s[8]]  ^ k10[8];
    out[9]  = SBOX[s[13]] ^ k10[9];
    out[10] = SBOX[s[2]]  ^ k10[10];
    out[11] = SBOX[s[7]]  ^ k10[11];

    out[12] = SBOX[s[12]] ^ k10[12];
    out[13] = SBOX[s[1]]  ^ k10[13];
    out[14] = SBOX[s[6]]  ^ k10[14];
    out[15] = SBOX[s[11]] ^ k10[15];
}

/* 2. T-table AES-128 block encryption (16-byte block, pre-expanded 32-bit round keys) */
void __attribute__((noinline)) aes128_ttable_block(
    uint8_t *out,
    const uint8_t *in,
    const uint32_t *rk32
) {
    const uint32_t *in32 = (const uint32_t *)in;
    uint32_t *out32 = (uint32_t *)out;

    uint32_t s0 = in32[0] ^ rk32[0];
    uint32_t s1 = in32[1] ^ rk32[1];
    uint32_t s2 = in32[2] ^ rk32[2];
    uint32_t s3 = in32[3] ^ rk32[3];

    for (int r = 1; r <= 9; r++) {
        const uint32_t *k = rk32 + (r << 2);
        uint32_t t0 = Te0[s0 & 0xff] ^ Te1[(s1 >> 8) & 0xff] ^ Te2[(s2 >> 16) & 0xff] ^ Te3[s3 >> 24] ^ k[0];
        uint32_t t1 = Te0[s1 & 0xff] ^ Te1[(s2 >> 8) & 0xff] ^ Te2[(s3 >> 16) & 0xff] ^ Te3[s0 >> 24] ^ k[1];
        uint32_t t2 = Te0[s2 & 0xff] ^ Te1[(s3 >> 8) & 0xff] ^ Te2[(s0 >> 16) & 0xff] ^ Te3[s1 >> 24] ^ k[2];
        uint32_t t3 = Te0[s3 & 0xff] ^ Te1[(s0 >> 8) & 0xff] ^ Te2[(s1 >> 16) & 0xff] ^ Te3[s2 >> 24] ^ k[3];
        s0 = t0; s1 = t1; s2 = t2; s3 = t3;
    }

    const uint32_t *k10 = rk32 + 40;
    out32[0] = ((uint32_t)SBOX[s0 & 0xff])
             | ((uint32_t)SBOX[(s1 >> 8) & 0xff] << 8)
             | ((uint32_t)SBOX[(s2 >> 16) & 0xff] << 16)
             | ((uint32_t)SBOX[s3 >> 24] << 24);
    out32[0] ^= k10[0];

    out32[1] = ((uint32_t)SBOX[s1 & 0xff])
             | ((uint32_t)SBOX[(s2 >> 8) & 0xff] << 8)
             | ((uint32_t)SBOX[(s3 >> 16) & 0xff] << 16)
             | ((uint32_t)SBOX[s0 >> 24] << 24);
    out32[1] ^= k10[1];

    out32[2] = ((uint32_t)SBOX[s2 & 0xff])
             | ((uint32_t)SBOX[(s3 >> 8) & 0xff] << 8)
             | ((uint32_t)SBOX[(s0 >> 16) & 0xff] << 16)
             | ((uint32_t)SBOX[s1 >> 24] << 24);
    out32[2] ^= k10[2];

    out32[3] = ((uint32_t)SBOX[s3 & 0xff])
             | ((uint32_t)SBOX[(s0 >> 8) & 0xff] << 8)
             | ((uint32_t)SBOX[(s1 >> 16) & 0xff] << 16)
             | ((uint32_t)SBOX[s2 >> 24] << 24);
    out32[3] ^= k10[3];
}

/* 3. RV32 Zkne instructions (`aes32esmi` and `aes32esi`) */
#define AES32ESMI(rd, rs1, rs2, bs) \
    __asm__ volatile (".insn r 0x33, 0, %3, %0, %1, %2" : "=r"(rd) : "r"(rs1), "r"(rs2), "i"(((bs) << 5) | 0x13))

#define AES32ESI(rd, rs1, rs2, bs) \
    __asm__ volatile (".insn r 0x33, 0, %3, %0, %1, %2" : "=r"(rd) : "r"(rs1), "r"(rs2), "i"(((bs) << 5) | 0x11))

void __attribute__((noinline)) aes128_zkne_block(
    uint8_t *out,
    const uint8_t *in,
    const uint32_t *rk32
) {
    const uint32_t *in32 = (const uint32_t *)in;
    uint32_t *out32 = (uint32_t *)out;

    uint32_t s0 = in32[0] ^ rk32[0];
    uint32_t s1 = in32[1] ^ rk32[1];
    uint32_t s2 = in32[2] ^ rk32[2];
    uint32_t s3 = in32[3] ^ rk32[3];

    for (int r = 1; r <= 9; r++) {
        const uint32_t *k = rk32 + (r << 2);
        uint32_t t0 = k[0], t1 = k[1], t2 = k[2], t3 = k[3];

        AES32ESMI(t0, t0, s0, 0); AES32ESMI(t0, t0, s1, 1); AES32ESMI(t0, t0, s2, 2); AES32ESMI(t0, t0, s3, 3);
        AES32ESMI(t1, t1, s1, 0); AES32ESMI(t1, t1, s2, 1); AES32ESMI(t1, t1, s3, 2); AES32ESMI(t1, t1, s0, 3);
        AES32ESMI(t2, t2, s2, 0); AES32ESMI(t2, t2, s3, 1); AES32ESMI(t2, t2, s0, 2); AES32ESMI(t2, t2, s1, 3);
        AES32ESMI(t3, t3, s3, 0); AES32ESMI(t3, t3, s0, 1); AES32ESMI(t3, t3, s1, 2); AES32ESMI(t3, t3, s2, 3);

        s0 = t0; s1 = t1; s2 = t2; s3 = t3;
    }

    const uint32_t *k10 = rk32 + 40;
    uint32_t t0 = k10[0], t1 = k10[1], t2 = k10[2], t3 = k10[3];
    AES32ESI(t0, t0, s0, 0); AES32ESI(t0, t0, s1, 1); AES32ESI(t0, t0, s2, 2); AES32ESI(t0, t0, s3, 3);
    AES32ESI(t1, t1, s1, 0); AES32ESI(t1, t1, s2, 1); AES32ESI(t1, t1, s3, 2); AES32ESI(t1, t1, s0, 3);
    AES32ESI(t2, t2, s2, 0); AES32ESI(t2, t2, s3, 1); AES32ESI(t2, t2, s0, 2); AES32ESI(t2, t2, s1, 3);
    AES32ESI(t3, t3, s3, 0); AES32ESI(t3, t3, s0, 1); AES32ESI(t3, t3, s1, 2); AES32ESI(t3, t3, s2, 3);

    out32[0] = t0;
    out32[1] = t1;
    out32[2] = t2;
    out32[3] = t3;
}

/* 4. Custom-0 (`0x0B`) AES column-round instruction modeled on ../AES_processor/rtl/aes_round64.v */
#define AES_ROUND_COL(rd, rs1, rs2) \
    __asm__ volatile (".insn r 0x0b, 0, 0, %0, %1, %2" : "=r"(rd) : "r"(rs1), "r"(rs2))

#define AES_FINAL_COL(rd, rs1, rs2) \
    __asm__ volatile (".insn r 0x0b, 1, 0, %0, %1, %2" : "=r"(rd) : "r"(rs1), "r"(rs2))

#define AES_SHIFT_EVEN(rd, rs1, rs2) \
    __asm__ volatile (".insn r 0x0b, 2, 0, %0, %1, %2" : "=r"(rd) : "r"(rs1), "r"(rs2))

#define AES_SHIFT_DIAG(rd, rs1, rs2) \
    __asm__ volatile (".insn r 0x0b, 3, 0, %0, %1, %2" : "=r"(rd) : "r"(rs1), "r"(rs2))

void __attribute__((noinline)) aes128_custom0_block(
    uint8_t *out,
    const uint8_t *in,
    const uint32_t *rk32
) {
    const uint32_t *in32 = (const uint32_t *)in;
    uint32_t *out32 = (uint32_t *)out;

    uint32_t s0 = in32[0] ^ rk32[0];
    uint32_t s1 = in32[1] ^ rk32[1];
    uint32_t s2 = in32[2] ^ rk32[2];
    uint32_t s3 = in32[3] ^ rk32[3];

    for (int r = 1; r <= 9; r++) {
        const uint32_t *k = rk32 + (r << 2);
        uint32_t e02, e13, o02, o13, c0, c1, c2, c3;
        /* ShiftRows across the four 32-bit columns */
        AES_SHIFT_DIAG(e02, s0, s1);
        AES_SHIFT_DIAG(o02, s2, s3);
        AES_SHIFT_DIAG(e13, s1, s2);
        AES_SHIFT_DIAG(o13, s3, s0);

        AES_SHIFT_EVEN(c0, e02, o02);
        AES_SHIFT_EVEN(c2, o02, e02);
        AES_SHIFT_EVEN(c1, e13, o13);
        AES_SHIFT_EVEN(c3, o13, e13);

        /* SubBytes + MixColumns + AddRoundKey per column (1 custom instruction each) */
        AES_ROUND_COL(s0, c0, k[0]);
        AES_ROUND_COL(s1, c1, k[1]);
        AES_ROUND_COL(s2, c2, k[2]);
        AES_ROUND_COL(s3, c3, k[3]);
    }

    const uint32_t *k10 = rk32 + 40;
    uint32_t e02, e13, o02, o13, c0, c1, c2, c3;
    AES_SHIFT_DIAG(e02, s0, s1);
    AES_SHIFT_DIAG(o02, s2, s3);
    AES_SHIFT_DIAG(e13, s1, s2);
    AES_SHIFT_DIAG(o13, s3, s0);

    AES_SHIFT_EVEN(c0, e02, o02);
    AES_SHIFT_EVEN(c2, o02, e02);
    AES_SHIFT_EVEN(c1, e13, o13);
    AES_SHIFT_EVEN(c3, o13, e13);

    AES_FINAL_COL(out32[0], c0, k10[0]);
    AES_FINAL_COL(out32[1], c1, k10[1]);
    AES_FINAL_COL(out32[2], c2, k10[2]);
    AES_FINAL_COL(out32[3], c3, k10[3]);
}

/* Shared buffers in .data/.bss for standalone or simulator-driven runs */
uint8_t g_key[16] __attribute__((aligned(16))) = {
    0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
    0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c
};
uint8_t g_pt[16] __attribute__((aligned(16))) = {
    0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d,
    0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07, 0x34
};
uint8_t g_rk[176] __attribute__((aligned(16)));
uint8_t g_ct_tablefree[16] __attribute__((aligned(16)));
uint8_t g_ct_ttable[16] __attribute__((aligned(16)));
uint8_t g_ct_zkne[16] __attribute__((aligned(16)));
uint8_t g_ct_custom0[16] __attribute__((aligned(16)));

void __attribute__((section(".text.init"), naked)) _start(void) {
    __asm__ volatile (
        "lui sp, 0x80020\n"
        "jal ra, init_ttables\n"
        "la a0, g_key\n"
        "la a1, g_rk\n"
        "jal ra, aes128_expand_key\n"
        "la a0, g_ct_tablefree\n"
        "la a1, g_pt\n"
        "la a2, g_rk\n"
        "jal ra, aes128_tablefree_block\n"
        "la a0, g_ct_ttable\n"
        "la a1, g_pt\n"
        "la a2, g_rk\n"
        "jal ra, aes128_ttable_block\n"
        "la a0, g_ct_zkne\n"
        "la a1, g_pt\n"
        "la a2, g_rk\n"
        "jal ra, aes128_zkne_block\n"
        "la a0, g_ct_custom0\n"
        "la a1, g_pt\n"
        "la a2, g_rk\n"
        "jal ra, aes128_custom0_block\n"
        "ebreak\n"
    );
}
