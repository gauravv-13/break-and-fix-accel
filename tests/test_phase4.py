import numpy as np
import pytest
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from src import accel, faults, memory_encryption as me, quantize

KEY = bytes(range(16))
NONCE = bytes([0xA5] * 8)


@pytest.fixture(scope="module")
def qm():
    return quantize.load_npz()


def test_keystream_matches_cryptography_ctr():
    n = 16 * 10 + 5
    enc = Cipher(algorithms.AES(KEY), modes.CTR(NONCE + bytes(8))).encryptor()
    ref = np.frombuffer(enc.update(bytes(n)) + enc.finalize(), dtype=np.uint8)
    np.testing.assert_array_equal(me.keystream(KEY, NONCE, n), ref)


def test_encrypt_then_decrypt_gives_identical_inference(qm):
    q, Xq, _ = qm
    ct = me.encrypt_weights(q, KEY, NONCE)
    assert np.mean(ct != faults.weight_memory(q)) > 0.95       # DRAM really holds ciphertext
    plain = me.decrypt_weights(q, ct, KEY, NONCE)
    for k in faults.WEIGHT_KEYS:
        np.testing.assert_array_equal(plain[k], q[k])
    np.testing.assert_array_equal(accel.run_inference(Xq, plain)[0], accel.run_inference(Xq, q)[0])


def test_ctr_gives_no_integrity_a_ciphertext_flip_is_the_same_plaintext_flip(qm):
    # Encryption hides the weights but does NOT stop Phase-2 style bit flips.
    q, _, _ = qm
    ct = me.encrypt_weights(q, KEY, NONCE)
    for p in [0, 7, 1234, faults.n_bits(q) - 1]:
        bad = ct.copy()
        bad[p // 8] ^= np.uint8(1 << (p % 8))
        dec = me.decrypt_weights(q, bad, KEY, NONCE)
        expect = faults.flip_bits(q, [p])
        for k in faults.WEIGHT_KEYS:
            np.testing.assert_array_equal(dec[k], expect[k])


def test_block_count(qm):
    q, _, _ = qm
    assert me.n_blocks_for(q) == -(-(64 * 32 + 32 * 10) // 16) == 148


def test_rv32i_counts_are_read_from_the_aes_track_results():
    counts, keyexp, cpis = me.load_rv32i_counts()
    assert {k: v["instr_per_block"] for k, v in counts.items()} == \
        {"tablefree": 2792, "ttable": 921, "zkne": 264, "custom0": 197}
    assert keyexp == 1471 and 1.3 in cpis


def test_extra_cycle_formulas():
    n = 148
    assert me.extra_cycles("engine_pipelined", n) == n + 10
    assert me.extra_cycles("engine_iterative", n) == 11 * n
    assert me.extra_cycles("zkne", n, cpi=1.0) == (264 + me.CTR_OVERHEAD_INSTR) * n
    assert me.extra_cycles("aes_processor_rv64", n) == 54 * n
    assert me.extra_cycles("engine_pipelined", n, sessions=360) == 360 * (n + 10)
