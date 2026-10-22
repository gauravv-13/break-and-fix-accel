"""AES-128 implementation and simulated side-channel analysis package."""
from .aes import (
    SBOX,
    RCON,
    sub_bytes,
    shift_rows,
    mix_columns,
    add_round_key,
    key_expansion,
    aes128_encrypt_block,
    aes128_encrypt_rounds,
)
from .traces import (
    SBOX_LUT,
    HW_LUT,
    TraceDataset,
    hamming_weight,
    simulate_traces,
)
from .cpa import (
    CPAResult,
    compute_hypothetical_hw,
    correlate_byte_time_series,
    run_cpa,
    run_cpa_on_dataset,
)

__all__ = [
    "SBOX",
    "RCON",
    "sub_bytes",
    "shift_rows",
    "mix_columns",
    "add_round_key",
    "key_expansion",
    "aes128_encrypt_block",
    "aes128_encrypt_rounds",
    "SBOX_LUT",
    "HW_LUT",
    "TraceDataset",
    "hamming_weight",
    "simulate_traces",
    "CPAResult",
    "compute_hypothetical_hw",
    "correlate_byte_time_series",
    "run_cpa",
    "run_cpa_on_dataset",
]
