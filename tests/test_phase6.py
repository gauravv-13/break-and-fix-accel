import json
import re
from pathlib import Path

from src import build_dashboard

ROOT = Path(__file__).resolve().parent.parent


def embedded_data(html):
    return json.loads(re.search(r"const D = (\{.*?\});\n", html, re.S).group(1))


def test_dashboard_is_self_contained_and_built_from_results():
    path, _ = build_dashboard.build()
    html = path.read_text()
    assert not re.search(r"""<(script|link)[^>]+(src|href)=["']https?://""", html)   # nothing external
    d = embedded_data(html)
    p1 = json.loads((ROOT / "results/phase1/summary.json").read_text())
    assert d["phase1"]["cycles_1"] == p1["total_cycles_batch=1"] == 920
    assert d["aes"]["n90"] == [15, 30, 80, 275]
    assert len(d["aes"]["success"]) == len(d["aes"]["sigma"]) == 4
    assert set(d["adaptive"]) == {"none", "parity", "crc8", "secded"}
    assert d["memory_bits"] == {"parity": 320, "crc8": 2560, "secded": 2496}


def test_dashboard_labels_analytical_entries():
    html = build_dashboard.build()[0].read_text()
    assert html.count('class="tag ana"') >= 3
    assert "MAC module was built but not tested" in html


def test_report_images_exist():
    report = (ROOT / "docs/report.md").read_text()
    links = re.findall(r"\]\((\.\./results/[^)]+\.png)\)", report)
    assert len(links) >= 5
    for link in links:
        assert (ROOT / "docs" / link).resolve().exists(), link
    for section in ["## 1. Problem", "## 2. Setup", "## 3. Results", "## 4. Related work",
                    "## 5. Limits", "## 6. Gaps and future work"]:
        assert section in report
