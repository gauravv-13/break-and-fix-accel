import shutil
import subprocess

import pytest

from src import experiments_phase5 as p5, hw_vectors

needs_iverilog = pytest.mark.skipif(not (shutil.which("iverilog") and shutil.which("vvp")),
                                    reason="Icarus Verilog not installed")
needs_yosys = pytest.mark.skipif(not shutil.which("yosys"), reason="Yosys not installed")


def test_vectors_are_reproducible():
    a = hw_vectors.write_vectors()
    first = {f.name: f.read_text() for f in hw_vectors.VEC_DIR.glob("*.hex")}
    b = hw_vectors.write_vectors()
    second = {f.name: f.read_text() for f in hw_vectors.VEC_DIR.glob("*.hex")}
    assert a == b and first == second and len(first) == 6


@needs_iverilog
def test_verilog_checkers_match_python_models():
    res = p5.run_testbenches()
    assert res["8B"]["passed"] and res["2B"]["passed"]
    assert res["8B"]["vectors"] > 2000 and res["2B"]["vectors"] > 500


@needs_iverilog
def test_testbench_detects_a_wrong_expected_value(tmp_path):
    hw_vectors.write_vectors()
    for f in hw_vectors.VEC_DIR.glob("*_2B.hex"):
        lines = f.read_text().split()
        if f.name.startswith("crc8"):
            lines[3] = f"{int(lines[3], 16) ^ 1:0{len(lines[3])}x}"
        (tmp_path / f.name).write_text("\n".join(lines) + "\n")
    exe = tmp_path / "tb"
    srcs = [str(p5.TB)] + [str(p5.RTL / f) for f in ("row_parity.v", "row_crc8.v", "row_secded.v")]
    subprocess.run(["iverilog", "-g2005", "-P", "tb_row_checkers.N_BYTES=2", "-P", "tb_row_checkers.R=5",
                    "-o", str(exe)] + srcs, check=True)
    out = subprocess.run(["vvp", "-n", str(exe), f"+DIR={tmp_path}"], capture_output=True, text=True).stdout
    assert "FAIL crc8 vec 3" in out and "PASS" not in out


@needs_yosys
def test_parity_synthesis_is_an_xor_tree():
    r = p5.synth("row_parity", {"N_BYTES": 8})
    assert r["cells"] == 64           # 63 XORs for the tree + 1 compare
    assert r["depth"] == 7            # log2(64) = 6 levels + 1 compare


@needs_yosys
def test_crc8_small_synthesis_is_shallow():
    r = p5.synth("row_crc8", {"N_BYTES": 2})
    assert r["depth"] <= 10
