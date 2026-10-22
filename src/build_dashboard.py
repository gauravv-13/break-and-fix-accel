"""Phase 6: build results/dashboard.html, one self-contained page from precomputed results.

Run:  .venv/bin/python -m src.build_dashboard

It only READS the result files of earlier phases (no experiment is re-run) and embeds the
numbers as JSON inside the page. No external scripts or fonts: it opens offline from disk.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "dashboard.html"
N_LIST = [1, 2, 5, 10, 20, 50, 100]


def load(p):
    return json.loads((RES / p).read_text())


def median_curve(runs, xkey, ykey, xmax):
    """Median accuracy over runs on the grid 0..xmax; a run that stopped keeps its last value."""
    grid = list(range(xmax + 1))
    cols = []
    for r in runs:
        xs, ys = r[xkey], r[ykey]
        col, j = [], 0
        for g in grid:
            while j + 1 < len(xs) and xs[j + 1] <= g:
                j += 1
            col.append(ys[j])
        cols.append(col)
    a = np.array(cols)
    return {"x": grid, "median": np.median(a, 0).round(4).tolist(),
            "lo": a.min(0).round(4).tolist(), "hi": a.max(0).round(4).tolist()}


def _sigma_keys(aes):
    keys = list(aes["success_curves"].keys())
    assert [float(k) for k in keys] == [float(v) for v in aes["sigma_values"]]
    return keys


def collect():
    p1 = load("phase1/summary.json")
    p2 = load("phase2/results.json")
    p2b = load("phase2/phase2b_results.json")
    p3 = load("phase3/phase3_results.json")
    p4 = load("phase4/phase4_results.json")
    p5 = load("phase5/phase5_results.json")
    aes = load("aes/cpa_results.json")

    rnd = {str(N): {"mean": p2["random"][str(N)]["mean"], "std": p2["random"][str(N)]["std"]} for N in N_LIST}
    tg_runs = [{"x": list(range(len(r["test_acc"]))), "y": r["test_acc"]} for r in p2["targeted"]]
    targeted = median_curve(tg_runs, "x", "y", 100)
    targeted["n_flips"] = [r["n_flips"] for r in p2["targeted"]]

    adaptive = {}
    for a, runs in p2b["adaptive"]["runs"].items():
        c = median_curve(runs, "bits", "test_acc", 48)
        s = p2b["adaptive"]["summary"][a]
        c.update({"bits_median": s["bits_median"], "bits_min": s["bits_min"], "bits_max": s["bits_max"],
                  "runs": s["runs"], "reached": s["reached_chance"]})
        adaptive[a] = c

    rw = p2b["secded"]["random_weights_only"]
    random_def = {str(N): {"alarm": rw[str(N)]["alarm_rate"],
                           "secded_repaired": rw[str(N)]["secded_fully_repaired"],
                           "secded_silent_no_alarm": rw[str(N)]["secded_silent_corruption_no_alarm"],
                           "acc_no_defense": rw[str(N)]["acc_no_defense_mean"],
                           "acc_after_secded": rw[str(N)]["acc_after_secded_mean"]} for N in N_LIST}
    false_alarm = {d: {str(N): p2b["checksum_memory"][d][str(N)]["false_alarm_rate"] for N in N_LIST}
                   for d in ("parity", "crc8", "secded")}
    same_row = p2b["secded"]["same_row"]
    hw = {r["kind"] + "_" + str(r["params"].get("N_BYTES", "")): {"cells": r["cells"], "depth": r["depth"]}
          for r in p5["synthesis"]}
    cost2b = p2b["secded"]["cost"]
    p3opts = {o["option"]: {k: o[k] for k in ("cycles_batch1", "cycles_batch360", "random_bits_batch1",
                                              "random_bits_batch360")} for o in p3["options"]}
    short = {"tablefree": "Software, table-free (RV32I)", "ttable": "Software, T-table (RV32I)",
             "zkne": "RV32I + Zkne", "custom0": "RV32I + Custom-0", "aes_processor_rv64": "AES_processor RV64 + Zkne",
             "engine_iterative": "AES engine, iterative", "engine_pipelined": "AES engine, pipelined"}
    p4rows = {r["option"]: {"label": r["label"], "short": short[r["option"]],
                            "extra_1": r["1 image|cpi1.3|extra"],
                            "extra_batch": r["360 images, one batch|cpi1.3|extra"],
                            "extra_each": r["360 images, one at a time|cpi1.3|extra"]} for r in p4["rows"]}
    return {
        "phase1": {"float_acc": p1["float_acc"], "int8_acc": p1["int8_acc"],
                   "cycles_1": p1["total_cycles_batch=1"], "cycles_360": p1["total_cycles_batch=360"],
                   "util_l1_360": p1["layers"]["layer1_batch=360"]["utilization"],
                   "util_l2_360": p1["layers"]["layer2_batch=360"]["utilization"]},
        "n_list": N_LIST, "random": rnd, "targeted": targeted, "adaptive": adaptive,
        "random_def": random_def, "false_alarm": false_alarm, "same_row": same_row,
        "memory_bits": {d: cost2b[d]["memory_bits"] for d in ("parity", "crc8", "secded")},
        "weight_bits": p2b["checksum_memory"]["parity"]["weight_bits"],
        "hw": hw, "pe": {"cells": p5["pe_mac"]["cells"], "depth": p5["pe_mac"]["depth"]},
        "phase3": p3opts, "shuffle": p3["shuffle_costs"], "phase4": p4rows,
        # lists aligned with sigma_values (the JSON keys are strings like "1.0")
        "aes": {"sigma": aes["sigma_values"], "grid": aes["trace_grid"],
                "success": [aes["success_curves"][k] for k in _sigma_keys(aes)],
                "n90": [aes["traces_for_90pct"][k] for k in _sigma_keys(aes)]},
    }


def build():
    data = collect()
    html = TEMPLATE.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    OUT.write_text(html)
    return OUT, len(html)


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Accelerator Security Lab</title>
<style>
:root {
  color-scheme: light;
  --bg: #f6f5f2; --surface: #fcfcfb; --border: #e4e3df; --grid: #ebeae6;
  --text: #0b0b0b; --text2: #52514e; --muted: #7a7974;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4;
  --band1: rgba(42,120,214,.16); --band2: rgba(235,104,52,.13);
  --warn-bg: #fff4dc; --warn-text: #6b4a00;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --bg: #121211; --surface: #1a1a19; --border: #2e2e2c; --grid: #262624;
    --text: #f0efec; --text2: #c3c2b7; --muted: #8f8e88;
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181;
    --band1: rgba(57,135,229,.22); --band2: rgba(217,89,38,.15);
    --warn-bg: #3a2f12; --warn-text: #f3d48a;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #121211; --surface: #1a1a19; --border: #2e2e2c; --grid: #262624;
  --text: #f0efec; --text2: #c3c2b7; --muted: #8f8e88;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181;
  --band1: rgba(57,135,229,.22); --band2: rgba(217,89,38,.15);
  --warn-bg: #3a2f12; --warn-text: #f3d48a;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
header, main { max-width: 1180px; margin: 0 auto; padding: 0 16px; }
header { padding-top: 24px; padding-bottom: 8px; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 0 0 4px; }
.sub { color: var(--text2); margin: 0; }
.controls { position: sticky; top: 0; z-index: 5; background: var(--bg); border-bottom: 1px solid var(--border);
  padding: 10px 0; margin-bottom: 14px; display: flex; flex-wrap: wrap; gap: 10px 22px; align-items: center; }
.group { display: flex; flex-wrap: wrap; gap: 6px 12px; align-items: center; max-width: 100%; min-width: 0; }
.group > b { font-size: 12px; color: var(--text2); text-transform: uppercase; letter-spacing: .04em; margin-right: 2px; }
label.tog { display: inline-flex; align-items: center; gap: 5px; cursor: pointer; white-space: nowrap; }
.slider { display: inline-flex; align-items: center; gap: 8px; max-width: 100%; }
.slider input[type=range] { min-width: 0; width: 140px; }
.slider output { font-variant-numeric: tabular-nums; min-width: 3.2em; font-weight: 600; }
select { max-width: 100%; font: inherit; color: var(--text); background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 2px 4px; }
.grid2 { display: grid; grid-template-columns: minmax(0, 3fr) minmax(0, 2fr); gap: 14px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; margin-bottom: 14px; min-width: 0; }
.card p.note { color: var(--text2); font-size: 12.5px; margin: 6px 0 0; }
.tag { display: inline-block; font-size: 11px; padding: 1px 6px; border-radius: 4px; margin-left: 6px; vertical-align: 1px; }
.tag.meas { background: var(--band1); color: var(--text); }
.tag.ana { background: var(--warn-bg); color: var(--warn-text); }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 5px 6px; border-bottom: 1px solid var(--border); vertical-align: top; }
th { color: var(--text2); font-weight: 600; font-size: 12px; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
tr.off td { color: var(--muted); }
.kpis { display: flex; flex-wrap: wrap; gap: 10px; margin: 6px 0 2px; }
.kpi { border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; min-width: 140px; flex: 1; }
.kpi .v { font-size: 20px; font-weight: 650; font-variant-numeric: tabular-nums; }
.kpi .l { font-size: 12px; color: var(--text2); }
svg { display: block; width: 100%; height: auto; overflow: visible; }
svg text { fill: var(--text2); font-size: 11px; }
.axis line, .axis path { stroke: var(--muted); }
.gridline { stroke: var(--grid); }
.legend { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 12px; color: var(--text2); margin-top: 4px; }
.legend i { display: inline-block; width: 14px; height: 3px; border-radius: 2px; vertical-align: middle; margin-right: 5px; }
.tip { position: fixed; pointer-events: none; background: var(--surface); color: var(--text); border: 1px solid var(--border);
  border-radius: 6px; padding: 6px 8px; font-size: 12px; box-shadow: 0 2px 8px rgba(0,0,0,.15); display: none; z-index: 10; max-width: 280px; }
.scroll { overflow-x: auto; }
#tb_defs { min-width: 760px; }
footer { color: var(--text2); font-size: 12px; max-width: 1180px; margin: 0 auto; padding: 4px 16px 28px; }
@media (max-width: 860px) { .grid2 { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <h1>Break-and-fix lab: a tiny ML accelerator, attacked and defended</h1>
  <p class="sub">8x8 systolic array running an INT8 MLP (64-32-10) on scikit-learn digits. Every number comes from
  the saved results of Phases 1-5. <span class="tag meas">measured</span> = simulated in this project;
  <span class="tag ana">analytical</span> = formula or published result, not measured here.</p>
</header>
<main>
  <div class="controls" id="controls">
    <div class="group"><b>Row checks</b>
      <label class="tog"><input type="checkbox" id="d_parity"> Parity</label>
      <label class="tog"><input type="checkbox" id="d_crc8" checked> CRC-8</label>
      <label class="tog"><input type="checkbox" id="d_secded"> SECDED</label>
      <label class="tog"><input type="checkbox" id="d_mac8"> MAC-8</label>
      <label class="tog"><input type="checkbox" id="d_mac16"> MAC-16</label>
    </div>
    <div class="group"><b>Side channel</b>
      <label class="tog"><input type="checkbox" id="d_shuffle"> Shuffling</label>
      <label class="tog"><input type="checkbox" id="d_mask"> Masking</label>
      <select id="sca_mode" aria-label="randomness refresh">
        <option value="batch">new randomness per batch</option>
        <option value="image">new randomness per image</option>
      </select>
      <select id="mask_dp" aria-label="masking datapath">
        <option value="parallel">masking: 2 multipliers/PE</option>
        <option value="serial">masking: 1 multiplier, 2 passes</option>
      </select>
    </div>
    <div class="group"><b>Memory</b>
      <label class="tog"><input type="checkbox" id="d_enc"> AES-CTR encryption</label>
      <select id="enc_impl" aria-label="AES implementation"></select>
    </div>
    <div class="group"><b>Sliders</b>
      <span class="slider"><label for="s_flips">flipped bits</label>
        <input type="range" id="s_flips" min="0" max="6" step="1" value="3"><output id="o_flips"></output></span>
      <span class="slider"><label for="s_sigma">noise &sigma;</label>
        <input type="range" id="s_sigma" min="0" max="3" step="1" value="1"><output id="o_sigma"></output></span>
    </div>
  </div>

  <div class="card">
    <h2>Total cost of the selected defenses <span class="tag meas">measured parts, added up</span></h2>
    <div class="kpis" id="kpis"></div>
    <p class="note" id="cost_note"></p>
  </div>

  <div class="grid2">
    <div class="card">
      <h2>Bit flips in weight memory <span class="tag meas">measured</span></h2>
      <div id="ch_flips"></div>
      <div class="legend" id="lg_flips"></div>
      <p class="note">Random: 20 trials per point (mean &plusmn; 1 std). Targeted: progressive bit search (BFA idea),
      20 runs, median and min-max; a run that reached chance (&le;11%) keeps its last value. Dashed: an attacker who only
      uses flips the selected linear check cannot see (Phase 2b, x = bits flipped).</p>
    </div>
    <div class="card">
      <h2>At <span id="t_flips"></span> flipped bits</h2>
      <div class="scroll"><table id="tb_flips"></table></div>
      <p class="note" id="flip_note"></p>
    </div>
  </div>

  <div class="card">
    <h2>Row-check comparison</h2>
    <div class="scroll"><table id="tb_defs"></table></div>
    <p class="note">Same-row columns: share of trials where several flips in ONE 8-weight row end up silently wrong
    (missed or mis-corrected), 2,000 trials each. False alarms: random flips may also hit the stored check bits.
    MAC rows are analytical only: the MAC module was built but not tested, and no MAC experiments were run (Phase 2c, partial).</p>
  </div>

  <div class="grid2">
    <div class="card">
      <h2>Power side channel: AES-128 CPA <span class="tag meas">measured (AES track)</span></h2>
      <div id="ch_sca"></div>
      <div class="legend" id="lg_sca"></div>
      <p class="note">Simulated Hamming-weight traces plus Gaussian noise; success = all 16 key bytes right,
      20 runs per point. No side-channel attack was run on the accelerator itself.</p>
    </div>
    <div class="card">
      <h2>Side-channel defenses</h2>
      <div id="sca_text"></div>
    </div>
  </div>

  <div class="grid2">
    <div class="card">
      <h2>Weight encryption (AES-128-CTR) <span class="tag meas">counts measured, cycles modelled</span></h2>
      <div id="ch_enc"></div>
      <p class="note">Total time with decryption divided by time without, CPI 1.3 for the RV32I options. Instruction
      counts per block are measured on the AES track's RV32I simulator; engines and the CTR overhead are assumptions.
      Encryption hides the weights but does not stop bit flips: a flipped ciphertext bit flips the same weight bit.</p>
    </div>
    <div class="card">
      <h2>Row checkers in hardware <span class="tag meas">Yosys</span></h2>
      <div class="scroll"><table id="tb_hw"></table></div>
      <p class="note">2-input gate cells and logic depth (gate levels) for an 8-weight row. One PE multiply-accumulate
      sets the clock. All three checkers fit in one cycle, so each adds 1 cycle per tile (40 per inference).</p>
    </div>
  </div>
</main>
<footer id="foot"></footer>
<div class="tip" id="tip"></div>

<script>
const D = /*__DATA__*/null;
const $ = (id) => document.getElementById(id);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const fmt = (v, d = 0) => Number(v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (v, d = 1) => (100 * v).toFixed(d) + "%";
const SIGMAS = D.aes.sigma;
const NAMES = { parity: "Parity", crc8: "CRC-8", secded: "SECDED", mac8: "MAC-8", mac16: "MAC-16" };

// ---------- state ----------
function state() {
  const on = (id) => $(id).checked;
  return {
    parity: on("d_parity"), crc8: on("d_crc8"), secded: on("d_secded"), mac8: on("d_mac8"), mac16: on("d_mac16"),
    shuffle: on("d_shuffle"), mask: on("d_mask"), enc: on("d_enc"),
    mode: $("sca_mode").value, dp: $("mask_dp").value, impl: $("enc_impl").value,
    N: D.n_list[+$("s_flips").value], si: +$("s_sigma").value, sigma: SIGMAS[+$("s_sigma").value],
  };
}

// ---------- tiny SVG chart ----------
const tip = $("tip");
function showTip(ev, html) { tip.innerHTML = html; tip.style.display = "block";
  const x = Math.min(ev.clientX + 14, window.innerWidth - 290); tip.style.left = x + "px"; tip.style.top = (ev.clientY + 12) + "px"; }
function hideTip() { tip.style.display = "none"; }

function chart(el, cfg) {
  const W = 640, H = 300, m = { l: 46, r: 14, t: 10, b: 40 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const lx = cfg.xlog ? (v) => Math.log10(v) : (v) => v;
  const [x0, x1] = cfg.xdom.map(lx), [y0, y1] = cfg.ydom;
  const X = (v) => m.l + (lx(v) - x0) / (x1 - x0) * iw;
  const Y = (v) => m.t + ih - (v - y0) / (y1 - y0) * ih;
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${cfg.label}">`;
  for (const t of cfg.yticks) s += `<line class="gridline" x1="${m.l}" x2="${W - m.r}" y1="${Y(t)}" y2="${Y(t)}"/>` +
    `<text x="${m.l - 6}" y="${Y(t) + 4}" text-anchor="end">${cfg.yfmt ? cfg.yfmt(t) : t}</text>`;
  for (const t of cfg.xticks) s += `<text x="${X(t)}" y="${H - m.b + 16}" text-anchor="middle">${t}</text>`;
  s += `<line class="axis" x1="${m.l}" x2="${W - m.r}" y1="${m.t + ih}" y2="${m.t + ih}" stroke="${css("--muted")}"/>`;
  s += `<text x="${m.l + iw / 2}" y="${H - 4}" text-anchor="middle">${cfg.xlabel}</text>`;
  s += `<text transform="translate(12 ${m.t + ih / 2}) rotate(-90)" text-anchor="middle">${cfg.ylabel}</text>`;
  for (const b of cfg.bands || []) {
    const pts = b.x.map((x, i) => `${X(x)},${Y(b.hi[i])}`).concat(b.x.map((x, i) => `${X(x)},${Y(b.lo[i])}`).reverse());
    s += `<polygon points="${pts.join(" ")}" fill="${b.fill}" stroke="none"/>`;
  }
  for (const r of cfg.rules || []) s += r.axis === "x"
    ? `<line x1="${X(r.v)}" x2="${X(r.v)}" y1="${m.t}" y2="${m.t + ih}" stroke="${r.color}" stroke-width="1.5" stroke-dasharray="${r.dash || ""}"/>` +
      (r.text ? `<text x="${X(r.v) + 4}" y="${m.t + ih - 6}">${r.text}</text>` : "")
    : `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(r.v)}" y2="${Y(r.v)}" stroke="${r.color}" stroke-width="1" stroke-dasharray="${r.dash || ""}"/>` +
      (r.text ? `<text x="${W - m.r - 4}" y="${Y(r.v) - 4}" text-anchor="end">${r.text}</text>` : "");
  for (const L of cfg.lines) {
    const pts = L.x.map((x, i) => `${X(x)},${Y(L.y[i])}`).join(" ");
    s += `<polyline points="${pts}" fill="none" stroke="${L.color}" stroke-width="2" stroke-dasharray="${L.dash || ""}" stroke-linejoin="round"/>`;
    if (L.markers) for (let i = 0; i < L.x.length; i++) s += `<circle cx="${X(L.x[i])}" cy="${Y(L.y[i])}" r="4" fill="${L.color}" stroke="${css("--surface")}" stroke-width="2"/>`;
  }
  s += `<rect x="${m.l}" y="${m.t}" width="${iw}" height="${ih}" fill="transparent" class="hit"/>`;
  s += `<line class="cross" x1="0" x2="0" y1="${m.t}" y2="${m.t + ih}" stroke="${css("--muted")}" stroke-width="1" visibility="hidden"/></svg>`;
  el.innerHTML = s;
  const svg = el.querySelector("svg"), cross = svg.querySelector(".cross");
  svg.querySelector(".hit").addEventListener("mousemove", (ev) => {
    const r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) / r.width * W;
    const xv = cfg.xlog ? Math.pow(10, x0 + (px - m.l) / iw * (x1 - x0)) : x0 + (px - m.l) / iw * (x1 - x0);
    const rows = [];
    let snapX = null;
    for (const L of cfg.lines) {
      let best = 0;
      for (let i = 1; i < L.x.length; i++) if (Math.abs(lx(L.x[i]) - lx(xv)) < Math.abs(lx(L.x[best]) - lx(xv))) best = i;
      if (snapX === null) snapX = L.x[best];
      rows.push(`<i style="display:inline-block;width:10px;height:3px;background:${L.color};margin-right:5px;vertical-align:middle"></i>${L.name}: <b>${cfg.yfmt ? cfg.yfmt(L.y[best]) : L.y[best]}</b> at ${L.x[best]}`);
    }
    cross.setAttribute("x1", X(snapX)); cross.setAttribute("x2", X(snapX)); cross.setAttribute("visibility", "visible");
    showTip(ev, rows.join("<br>"));
  });
  svg.querySelector(".hit").addEventListener("mouseleave", () => { hideTip(); cross.setAttribute("visibility", "hidden"); });
}

function legend(el, items) {
  el.innerHTML = items.map((it) => `<span><i style="background:${it.color};${it.dash ? "height:0;border-top:2px dashed " + it.color + ";background:none" : ""}"></i>${it.name}</span>`).join("");
}

// ---------- cost model (measured per-defense costs, added up) ----------
function rowCheckLatency(S) {
  let lat = 0;
  if (S.parity || S.crc8 || S.secded) lat = 1;          // Phase 5: 1 cycle each, run in parallel
  if (S.mac8 || S.mac16) lat = Math.max(lat, 11);       // assumption (pipelined AES-CMAC), module untested
  return lat;
}
function costs(S) {
  const tiles = 40, base1 = D.phase1.cycles_1, base360 = D.phase1.cycles_360;
  const perImage = (S.shuffle || S.mask) && S.mode === "image";
  const lat = rowCheckLatency(S);
  const maskConv = S.mask ? 2 * tiles : 0;
  const enc = S.enc ? D.phase4[S.impl] : null;
  const one = (base) => (S.mask && S.dp === "serial" ? 2 * base : base) + lat * tiles + maskConv;
  let c1 = one(base1) + (enc ? enc.extra_1 : 0);
  let c360 = perImage ? 360 * c1 : one(base360) + (enc ? enc.extra_batch : 0);
  let mem = 0;
  for (const d of ["parity", "crc8", "secded"]) if (S[d]) mem += D.memory_bits[d];
  if (S.mac8) mem += 320 * 8;
  if (S.mac16) mem += 320 * 16;
  if (S.mask) mem += D.weight_bits;                    // second Boolean share of every weight
  const loads = perImage ? 360 : 1;
  const rbits1 = (S.shuffle ? D.shuffle.random_bits : 0) + (S.mask ? 151552 : 0);
  return { c1, c360, base1, base360, mem, rbits1, rbits360: rbits1 * loads, perImage, lat };
}

// ---------- renderers ----------
function renderKpis(S) {
  const c = costs(S);
  const k = (v, l) => `<div class="kpi"><div class="v">${v}</div><div class="l">${l}</div></div>`;
  $("kpis").innerHTML =
    k(fmt(c.c1), `cycles, 1 image (${(c.c1 / c.base1).toFixed(2)}x of ${fmt(c.base1)})`) +
    k(fmt(c.c360), `cycles, 360 images (${(c.c360 / c.base360).toFixed(2)}x of ${fmt(c.base360)})`) +
    k(fmt(c.mem), `extra memory bits (${pct(c.mem / D.weight_bits)} of the weights)`) +
    k(fmt(c.rbits360), "random bits for 360 images") +
    k(pct(D.phase1.int8_acc, 2), "clean accuracy (unchanged by every defense)");
  const notes = [];
  if (c.perImage) notes.push("Fresh randomness per image forces one image at a time (weights reload per image).");
  if (S.mac8 || S.mac16) notes.push("MAC cycles (+11 per tile) are an assumption; the MAC module is untested.");
  if (S.enc) notes.push(`Encryption: ${D.phase4[S.impl].label}.`);
  notes.push("Per-defense costs were measured separately and are simply added up here; their interaction was not simulated. Row checks run in parallel, so only the slowest one counts.");
  $("cost_note").textContent = notes.join(" ");
}

function renderFlips(S) {
  const N = S.N, R = D.random, T = D.targeted;
  $("o_flips").textContent = N; $("t_flips").textContent = N;
  const rx = D.n_list, rm = rx.map((n) => R[n].mean), rs = rx.map((n) => R[n].std);
  const tx = T.x.slice(1), tm = T.median.slice(1), tlo = T.lo.slice(1), thi = T.hi.slice(1);
  const lines = [
    { name: "random flips", x: rx, y: rm, color: css("--s1"), markers: true },
    { name: "targeted (median)", x: tx, y: tm, color: css("--s2") },
  ];
  const adColors = { parity: css("--s5"), crc8: css("--s3"), secded: css("--s4") };
  const leg = [{ name: "random flips (mean ± 1 std)", color: css("--s1") }, { name: "targeted search (median, min-max)", color: css("--s2") }];
  for (const d of ["parity", "crc8", "secded"]) if (S[d]) {
    const A = D.adaptive[d];
    lines.push({ name: `adaptive vs ${NAMES[d]}`, x: A.x.slice(1), y: A.median.slice(1), color: adColors[d], dash: "5 4" });
    leg.push({ name: `attacker invisible to ${NAMES[d]} (median)`, color: adColors[d], dash: true });
  }
  chart($("ch_flips"), {
    label: "accuracy versus number of flipped bits", xlog: true, xdom: [1, 100], ydom: [0, 1],
    xticks: [1, 2, 5, 10, 20, 50, 100], yticks: [0, 0.2, 0.4, 0.6, 0.8, 1.0], yfmt: (v) => Math.round(v * 100) + "%",
    xlabel: "flipped bits (log scale)", ylabel: "test accuracy",
    bands: [{ x: rx, lo: rm.map((v, i) => v - rs[i]), hi: rm.map((v, i) => Math.min(1, v + rs[i])), fill: css("--band1") },
            { x: tx, lo: tlo, hi: thi, fill: css("--band2") }],
    rules: [{ axis: "x", v: N, color: css("--text2"), dash: "3 3" }, { axis: "y", v: 0.1, color: css("--muted"), dash: "2 3", text: "chance 10%" }],
    lines,
  });
  legend($("lg_flips"), leg);

  const RD = D.random_def[N];
  const tAt = T.median[Math.min(N, T.median.length - 1)];
  let rows = `<tr><th>quantity</th><th class="num">value</th></tr>
    <tr><td>accuracy, ${N} random flips (no defense)</td><td class="num">${pct(R[N].mean)}</td></tr>
    <tr><td>accuracy, ${N} targeted flips (median of 20)</td><td class="num">${pct(tAt)}</td></tr>`;
  const any = ["parity", "crc8", "secded", "mac8", "mac16"].filter((d) => S[d]);
  for (const d of any) {
    if (d === "mac8" || d === "mac16") {
      const t = d === "mac8" ? 8 : 16;
      rows += `<tr><td>${NAMES[d]}: chance one changed row passes <span class="tag ana">analytical</span></td><td class="num">2<sup>-${t}</sup> = 1/${fmt(2 ** t)}</td></tr>`;
      continue;
    }
    const undetected = d === "secded" ? RD.secded_silent_no_alarm : 1 - RD.alarm[d];
    rows += `<tr><td>${NAMES[d]}: random flips left wrong with no alarm</td><td class="num">${pct(undetected)}</td></tr>`;
    rows += `<tr><td>${NAMES[d]}: false alarm (check bits attackable)</td><td class="num">${pct(D.false_alarm[d][N])}</td></tr>`;
    if (d === "secded") rows += `<tr><td>SECDED: fully repaired / mean acc. after correction</td><td class="num">${pct(RD.secded_repaired)} / ${pct(RD.acc_after_secded)}</td></tr>`;
    const A = D.adaptive[d];
    rows += `<tr><td>${NAMES[d]}: an attacker who knows it reaches chance after</td><td class="num">${A.bits_median} bits (${A.bits_min}-${A.bits_max}), unseen</td></tr>`;
  }
  if (!any.length) rows += `<tr><td colspan="2">Turn on a row check to see what it catches.</td></tr>`;
  $("tb_flips").innerHTML = rows;
  $("flip_note").textContent = "Random-flip detection uses 1,000 trials per point; targeted runs: 20 attacker batches. The non-adaptive targeted attack was flagged from its first flip by parity and CRC-8 in all 20 runs.";
}

function renderDefs(S) {
  const sr = D.same_row, fa = D.false_alarm, mb = D.memory_bits, W = D.weight_bits;
  const silent = (d, m) => sr[d][m].missed + sr[d][m].miscorrected;
  const head = `<tr><th>check</th><th class="num">extra memory</th><th class="num">cycles / tile</th><th class="num">2 flips in a row: silent</th><th class="num">3 flips: silent</th><th class="num">4 flips: silent</th><th class="num">false alarm, 1 flip</th><th class="num">adaptive attacker</th></tr>`;
  let body = "";
  for (const d of ["parity", "crc8", "secded"]) {
    body += `<tr class="${S[d] ? "" : "off"}"><td>${NAMES[d]}</td><td class="num">${fmt(mb[d])} (${pct(mb[d] / W)})</td><td class="num">1</td>` +
      `<td class="num">${pct(silent(d, 2))}</td><td class="num">${pct(silent(d, 3))}</td><td class="num">${pct(silent(d, 4))}</td>` +
      `<td class="num">${pct(fa[d][1])}</td><td class="num">unseen in 20/20 runs, ${D.adaptive[d].bits_median} bits</td></tr>`;
  }
  for (const [d, t] of [["mac8", 8], ["mac16", 16]]) {
    body += `<tr class="${S[d] ? "" : "off"}"><td>${NAMES[d]} <span class="tag ana">analytical</span></td><td class="num">${fmt(320 * t)} (${pct(320 * t / W)})</td><td class="num">11 (assumed)</td>` +
      `<td class="num" colspan="3">each changed row passes with 2<sup>-${t}</sup>, not measured</td><td class="num">not measured</td><td class="num">key unknown: 2<sup>-${t}</sup> per changed row</td></tr>`;
  }
  $("tb_defs").innerHTML = head + body;
}

function renderSca(S) {
  $("o_sigma").textContent = S.sigma;
  const lines = [], leg = [];
  const cols = [css("--s1"), css("--s2"), css("--s3"), css("--s4")];
  SIGMAS.forEach((sg, i) => {
    const sel = i === S.si;
    lines.push({ name: `σ = ${sg}`, x: D.aes.grid, y: D.aes.success[i], color: cols[i], dash: sel ? "" : "4 4" });
    leg.push({ name: `σ = ${sg}${sel ? " (selected)" : ""}: 90% at ${D.aes.n90[i]} traces`, color: cols[i], dash: !sel });
  });
  chart($("ch_sca"), {
    label: "key recovery success versus number of traces", xlog: false, xdom: [0, D.aes.grid[D.aes.grid.length - 1]], ydom: [0, 1],
    xticks: [0, 100, 200, 300, 400], yticks: [0, 0.5, 0.9, 1], yfmt: (v) => Math.round(v * 100) + "%",
    xlabel: "traces", ylabel: "full-key success",
    rules: [{ axis: "x", v: D.aes.n90[S.si], color: css("--text2"), dash: "3 3", text: `${D.aes.n90[S.si]} traces` },
            { axis: "y", v: 0.9, color: css("--muted"), dash: "2 3", text: "90%" }],
    lines,
  });
  legend($("lg_sca"), leg);

  const n = D.aes.n90[S.si];
  let h = `<p>Measured on the AES model: <b>${n} traces</b> for 90% full-key recovery at σ = ${S.sigma}.</p>`;
  if (!S.shuffle && !S.mask) h += `<p class="note">Turn on shuffling or masking to see their cost (measured in the accelerator model) and the expected effect from published work (analytical).</p>`;
  const P3 = D.phase3;
  if (S.shuffle) {
    const key = S.mode === "image" ? "shuffle, new order per image" : "shuffle, one order per batch";
    const o = P3[key];
    h += `<h3 style="font-size:14px;margin:12px 0 2px">Shuffling</h3>
      <p>Cost <span class="tag meas">measured</span>: ${fmt(o.cycles_batch1)} cycles for 1 image, ${fmt(o.cycles_batch360)} for 360 images;
      ${fmt(o.random_bits_batch360)} random bits for 360 images; 544-bit permutation table.</p>
      <p>Effect <span class="tag ana">analytical, not measured</span>: spreading an operation over m time positions divides its
      correlation by m (about m² more traces for a single-point attack) or by √m if the attacker adds the positions up
      (about m more). Here m is roughly 32-2,048. If the accelerator leaked like this AES model, that would mean very roughly
      ${fmt(n * 32)} to ${fmt(n * 32 * 32)} traces for m = 32 (illustration only).
      ${S.mode === "batch" ? "<b>With one order per batch, images in the same batch are not protected from each other.</b>" : ""}</p>`;
  }
  if (S.mask) {
    const key = `mask (${S.dp === "parallel" ? "2 multipliers per PE" : "1 widened multiplier, 2 passes"}), ${S.mode === "image" ? "new masks per image" : "masks per batch"}`;
    const o = P3[key];
    h += `<h3 style="font-size:14px;margin:12px 0 2px">First-order Boolean masking</h3>
      <p>Cost <span class="tag meas">measured</span>: ${fmt(o.cycles_batch1)} cycles for 1 image, ${fmt(o.cycles_batch360)} for 360 images;
      ${fmt(o.random_bits_batch360)} random bits for 360 images; weight memory doubled; 32-bit shares in every PE.</p>
      <p>Effect <span class="tag ana">analytical, not measured</span>: first-order CPA (like the attack plotted here) should fail; a
      second-order attack is needed, whose trace count grows roughly like σ⁴ instead of σ² (Chari et al. 1999). No absolute
      number is claimed. Hardware glitches can still leak (Mangard et al. 2005), and bias/ReLU are not masked here.</p>`;
  }
  $("sca_text").innerHTML = h;
}

function renderEnc(S) {
  const rows = Object.entries(D.phase4);
  const H = 26 * rows.length + 50, W = 640, m = { l: 190, r: 60, t: 8, b: 34 };
  const iw = W - m.l - m.r;
  const scen = [["extra_1", D.phase1.cycles_1, "1 image"], ["extra_batch", D.phase1.cycles_360, "360, one batch"]];
  const maxv = Math.log10(700), X = (v) => m.l + Math.log10(v) / maxv * iw;
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="slowdown from weight decryption">`;
  for (const t of [1, 2, 5, 10, 50, 100, 500]) s += `<line class="gridline" x1="${X(t)}" x2="${X(t)}" y1="${m.t}" y2="${H - m.b}"/><text x="${X(t)}" y="${H - m.b + 14}" text-anchor="middle">${t}x</text>`;
  s += `<text x="${m.l + iw / 2}" y="${H - 4}" text-anchor="middle">total time / time without encryption (log scale)</text>`;
  rows.forEach(([key, r], i) => {
    const y = m.t + i * 26, sel = S.enc && key === S.impl;
    s += `<text x="${m.l - 8}" y="${y + 15}" text-anchor="end" style="${sel ? "fill:var(--text);font-weight:650" : ""}">${r.short}</text>`;
    scen.forEach(([k, base, nm], j) => {
      const v = (base + r[k]) / base, yy = y + 4 + j * 9;
      s += `<rect x="${m.l}" y="${yy}" width="${Math.max(1, X(v) - m.l)}" height="8" rx="2" fill="${j ? css("--s2") : css("--s1")}" opacity="${S.enc && !sel ? 0.35 : 1}"><title>${r.label}, ${nm}: ${v.toFixed(2)}x (+${fmt(r[k])} cycles)</title></rect>`;
      if (j === 0) s += `<text x="${X(v) + 4}" y="${yy + 8}">${v.toFixed(1)}x</text>`;
    });
  });
  s += `</svg>`;
  $("ch_enc").innerHTML = s + `<div class="legend"><span><i style="background:${css("--s1")}"></i>1 image</span><span><i style="background:${css("--s2")}"></i>360 images in one batch</span></div>`;
}

function renderHw() {
  const hw = D.hw, pe = D.pe;
  let h = `<tr><th>design</th><th class="num">cells</th><th class="num">depth</th><th class="num">vs PE depth</th></tr>`;
  for (const [k, n] of [["parity_8", "Parity"], ["crc8_8", "CRC-8"], ["secded_8", "SECDED"]])
    h += `<tr><td>${n}</td><td class="num">${hw[k].cells}</td><td class="num">${hw[k].depth}</td><td class="num">${(hw[k].depth / pe.depth).toFixed(2)}</td></tr>`;
  h += `<tr><td>one PE (multiply-accumulate)</td><td class="num">${pe.cells}</td><td class="num">${pe.depth}</td><td class="num">1.00</td></tr>`;
  $("tb_hw").innerHTML = h;
}

function render() {
  const S = state();
  $("enc_impl").disabled = !S.enc;
  $("mask_dp").disabled = !S.mask;
  $("sca_mode").disabled = !(S.shuffle || S.mask);
  renderKpis(S); renderFlips(S); renderDefs(S); renderSca(S); renderEnc(S); renderHw();
}

// ---------- init ----------
for (const [k, r] of Object.entries(D.phase4)) {
  const o = document.createElement("option"); o.value = k; o.textContent = r.label.replace(" on RV32I", " (RV32I)");
  $("enc_impl").appendChild(o);
}
$("enc_impl").value = "engine_pipelined";
try { const saved = JSON.parse(localStorage.getItem("dash_state") || "null");
  if (saved) for (const [id, v] of Object.entries(saved)) { const el = $(id); if (!el) continue; if (el.type === "checkbox") el.checked = v; else el.value = v; }
} catch (e) { /* storage unavailable: use defaults */ }
document.querySelectorAll("#controls input, #controls select").forEach((el) => el.addEventListener("input", () => {
  render();
  try { const st = {}; document.querySelectorAll("#controls input, #controls select").forEach((x) => st[x.id] = x.type === "checkbox" ? x.checked : x.value);
    localStorage.setItem("dash_state", JSON.stringify(st)); } catch (e) { /* ignore */ }
}));
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
$("foot").innerHTML = `Float accuracy ${pct(D.phase1.float_acc, 2)}, INT8 ${pct(D.phase1.int8_acc, 2)}; array utilization (360 images) layer 1 ${pct(D.phase1.util_l1_360)}, layer 2 ${pct(D.phase1.util_l2_360)}.
  Built by <code>src/build_dashboard.py</code> from results/phase1-5 and results/aes. Seeds fixed; see PROGRESS.md for every assumption.`;
render();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    path, size = build()
    print(f"wrote {path} ({size / 1024:.0f} KB)")
