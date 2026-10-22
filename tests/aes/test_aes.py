"""Tests for AES-128 against FIPS-197 Appendix A.1, B, C.1 and cryptography."""

from __future__ import annotations
import os
import sys
import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.aes.aes import (
    aes128_encrypt_block,
    aes128_encrypt_rounds,
    key_expansion,
    key_expansion_words,
)


def test_fips197_appendix_a1_key_expansion() -> None:
    """Verify AES-128 key expansion against FIPS-197 Appendix A.1."""
    cipher_key = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")
    expected_round_keys_hex = [
        "2b7e151628aed2a6abf7158809cf4f3c",  # w[0..3]
        "a0fafe1788542cb123a339392a6c7605",  # w[4..7]
        "f2c295f27a96b9435935807a7359f67f",  # w[8..11]
        "3d80477d4716fe3e1e237e446d7a883b",  # w[12..15]
        "ef44a541a8525b7fb671253bdb0bad00",  # w[16..19]
        "d4d1c6f87c839d87caf2b8bc11f915bc",  # w[20..23]
        "6d88a37a110b3efddbf98641ca0093fd",  # w[24..27]
        "4e54f70e5f5fc9f384a64fb24ea6dc4f",  # w[28..31]
        "ead27321b58dbad2312bf5607f8d292f",  # w[32..35]
        "ac7766f319fadc2128d12941575c006e",  # w[36..39]
        "d014f9a8c9ee2589e13f0cc8b6630ca6",  # w[40..43]
    ]
    round_keys = key_expansion(cipher_key)
    assert len(round_keys) == 11
    for r, (got, exp_hex) in enumerate(zip(round_keys, expected_round_keys_hex)):
        assert got == bytes.fromhex(exp_hex), f"Round {r} key mismatch: {got.hex()} != {exp_hex}"

    words = key_expansion_words(cipher_key)
    assert len(words) == 44
    assert words[4].hex() == "a0fafe17"
    assert words[43].hex() == "b6630ca6"


def test_fips197_appendix_b_cipher_example() -> None:
    """Verify full encryption and round states against FIPS-197 Appendix B."""
    plaintext = bytes.fromhex("3243f6a8885a308d313198a2e0370734")
    key = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")
    expected_ciphertext = bytes.fromhex("3925841d02dc09fbdc118597196a0b32")

    info = aes128_encrypt_rounds(plaintext, key)
    assert info["ciphertext"] == expected_ciphertext
    assert aes128_encrypt_block(plaintext, key) == expected_ciphertext

    # Check Round 1 intermediate values from FIPS-197 Appendix B
    assert info["after_add_round_key"][0] == bytes.fromhex("193de3bea0f4e22b9ac68d2ae9f84808")
    assert info["after_sub_bytes"][0] == bytes.fromhex("d42711aee0bf98f1b8b45de51e415230")
    assert info["after_shift_rows"][0] == bytes.fromhex("d4bf5d30e0b452aeb84111f11e2798e5")
    assert info["after_mix_columns"][0] == bytes.fromhex("046681e5e0cb199a48f8d37a2806264c")
    assert info["after_add_round_key"][1] == bytes.fromhex("a49c7ff2689f352b6b5bea43026a5049")


def test_fips197_appendix_c1_example_vector() -> None:
    """Verify full encryption and key schedule against FIPS-197 Appendix C.1."""
    plaintext = bytes.fromhex("00112233445566778899aabbccddeeff")
    key = bytes.fromhex("000102030405060708090a0b0c0d0e0f")
    expected_ciphertext = bytes.fromhex("69c4e0d86a7b0430d8cdb78070b4c55a")

    round_keys = key_expansion(key)
    assert round_keys[0].hex() == "000102030405060708090a0b0c0d0e0f"
    assert round_keys[1].hex() == "d6aa74fdd2af72fadaa678f1d6ab76fe"
    assert round_keys[10].hex() == "13111d7fe3944a17f307a78b4d2b30c5"

    assert aes128_encrypt_block(plaintext, key) == expected_ciphertext


def test_random_1000_against_cryptography() -> None:
    """Check 1000 random (key, plaintext) pairs against Python cryptography."""
    rng = np.random.default_rng(seed=20261022)
    for _ in range(1000):
        key = rng.bytes(16)
        plaintext = rng.bytes(16)

        got = aes128_encrypt_block(plaintext, key)

        encryptor = Cipher(algorithms.AES(key), modes.ECB()).encryptor()
        expected = encryptor.update(plaintext) + encryptor.finalize()

        assert got == expected
