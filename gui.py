"""
AI Circuit Simulator — GUI
===========================
Run: streamlit run gui.py
Requires: pip install streamlit ollama matplotlib
"""

import streamlit as st
import sys
import os
import json
import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Make ai_generator importable from same folder ──────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
try:
    import ai_generator as sim
    SIM_AVAILABLE = True
except ImportError as e:
    SIM_AVAILABLE = False
    SIM_ERROR = str(e)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AI Circuit Simulator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — Dark professional theme
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Root ── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}
.stApp {
    background: #0a0e17;
    color: #e2e8f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f1623;
    border-right: 1px solid #1e2d45;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stSlider label {
    color: #7dd3fc !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* ── Header ── */
.main-header {
    background: linear-gradient(135deg, #0f1623 0%, #0a1628 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00d4ff, #0066ff, #7c3aed);
}
.main-title {
    font-size: 2rem;
    font-weight: 700;
    color: #f0f9ff;
    margin: 0;
    letter-spacing: -0.02em;
}
.main-subtitle {
    color: #64748b;
    font-size: 0.9rem;
    margin-top: 4px;
    font-family: 'JetBrains Mono', monospace;
}
.badge-row {
    display: flex;
    gap: 8px;
    margin-top: 14px;
    flex-wrap: wrap;
}
.badge {
    background: #0f2744;
    border: 1px solid #1e4d8c;
    color: #7dd3fc;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}
.badge.green  { background:#052e16; border-color:#166534; color:#4ade80; }
.badge.purple { background:#1e1b4b; border-color:#4338ca; color:#a5b4fc; }
.badge.amber  { background:#2d1b00; border-color:#b45309; color:#fbbf24; }

/* ── Cards ── */
.card {
    background: #0f1623;
    border: 1px solid #1e2d45;
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.card-title {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 12px;
}

/* ── Metric boxes ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 20px;
}
.metric-box {
    background: #0a1628;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 14px 16px;
    text-align: center;
}
.metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #00d4ff;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
}
.metric-label {
    font-size: 0.7rem;
    color: #475569;
    margin-top: 4px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.metric-box.green .metric-value  { color: #4ade80; }
.metric-box.amber .metric-value  { color: #fbbf24; }
.metric-box.purple .metric-value { color: #a5b4fc; }
.metric-box.red .metric-value    { color: #f87171; }

/* ── AI output boxes ── */
.ai-box {
    background: #080e1a;
    border-left: 3px solid #0066ff;
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    margin-bottom: 12px;
    font-size: 0.88rem;
    line-height: 1.7;
    color: #cbd5e1;
}
.ai-box.green  { border-left-color: #16a34a; }
.ai-box.purple { border-left-color: #7c3aed; }
.ai-box.amber  { border-left-color: #d97706; }

/* ── Netlist display ── */
.netlist-block {
    background: #050a12;
    border: 1px solid #1e2d45;
    border-radius: 8px;
    padding: 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #4ade80;
    white-space: pre;
    overflow-x: auto;
    line-height: 1.6;
}

/* ── Status pill ── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}
.status-pill.success { background:#052e16; color:#4ade80; border:1px solid #166534; }
.status-pill.error   { background:#2d0a0a; color:#f87171; border:1px solid #991b1b; }
.status-pill.info    { background:#0c1a2e; color:#7dd3fc; border:1px solid #1e4d8c; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #0066ff, #0044cc) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: 0.02em !important;
    padding: 10px 24px !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0077ff, #0055dd) !important;
    transform: translateY(-1px) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0a0e17;
    border-bottom: 1px solid #1e2d45;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #475569 !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.05em !important;
}
.stTabs [aria-selected="true"] {
    color: #00d4ff !important;
    border-bottom: 2px solid #00d4ff !important;
}

/* ── Inputs ── */
.stTextInput > div > input,
.stSelectbox > div > div {
    background: #0a1628 !important;
    border: 1px solid #1e3a5f !important;
    color: #e2e8f0 !important;
    border-radius: 6px !important;
}
.stTextInput > div > input:focus {
    border-color: #0066ff !important;
    box-shadow: 0 0 0 2px rgba(0,102,255,0.2) !important;
}

/* ── Progress/spinner ── */
.stSpinner > div { border-top-color: #0066ff !important; }

/* ── Divider ── */
hr { border-color: #1e2d45 !important; }

/* ── Session history table ── */
.history-row {
    display: flex;
    align-items: center;
    padding: 10px 16px;
    border-bottom: 1px solid #1e2d45;
    font-size: 0.82rem;
    gap: 12px;
}
.history-row:hover { background: #0f1e30; }
.history-circuit { color: #7dd3fc; font-weight: 600; flex: 2; }
.history-peak    { color: #4ade80; font-family: 'JetBrains Mono', monospace; flex:1; }
.history-ripple  { color: #fbbf24; font-family: 'JetBrains Mono', monospace; flex:1; }
.history-time    { color: #475569; font-size: 0.72rem; flex:1; text-align:right; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "quick_select" not in st.session_state:
    st.session_state.quick_select = ""

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def run_sim(circuit, voltage, resistor, capacitor, inductor):
    """Call ai_generator.simulate and return (x, y, netlist, summary)."""
    if not SIM_AVAILABLE:
        return None, None, None, None
    x, y, netlist = sim.simulate(circuit, voltage, resistor, capacitor, inductor)
    summary = sim.analyze_waveform(x, y) if x and y else {}
    return x, y, netlist, summary

def make_waveform_fig(data_dict, voltage=None, resistor=None,
                      capacitor=None, inductor=None, summary=None):
    """
    data_dict: {"Circuit Name": {"x":[], "y":[], "analysis":{}}}
    If single circuit, show params panel. If multi, comparison layout.
    """
    n      = len(data_dict)
    colors = ["#00d4ff","#f87171","#4ade80","#fbbf24","#a5b4fc"]
    is_single = (n == 1)

    if is_single:
        fig = plt.figure(figsize=(11, 3.8), facecolor="#080e1a")
        gs  = gridspec.GridSpec(1, 2, width_ratios=[3, 1],
                                left=0.07, right=0.97, bottom=0.14, top=0.88)
        ax  = fig.add_subplot(gs[0])
        axp = fig.add_subplot(gs[1])
        axp.axis("off")
        name, d = next(iter(data_dict.items()))
        ax.plot(d["x"], d["y"], color=colors[0], linewidth=1.5)
        ax.set_facecolor("#080e1a")
        ax.spines[["top","right"]].set_visible(False)
        for spine in ["left","bottom"]:
            ax.spines[spine].set_color("#1e3a5f")
        ax.tick_params(colors="#475569", labelsize=7)
        ax.set_xlabel("Time (s)", color="#475569", fontsize=8)
        ax.set_ylabel("Voltage (V)", color="#475569", fontsize=8)
        ax.set_title(name, color="#f0f9ff", fontsize=9, fontweight="bold", pad=8)
        ax.grid(True, alpha=0.15, color="#1e3a5f")

        # Peak annotation
        if d["y"]:
            pv   = max(d["y"])
            pidx = d["y"].index(pv)
            ax.annotate(f"Peak {pv:.3f}V",
                        xy=(d["x"][pidx], pv),
                        xytext=(0.62, 0.92), xycoords=("axes fraction","axes fraction"),
                        fontsize=7.5, color=colors[0],
                        arrowprops=dict(arrowstyle="->", color=colors[0], lw=0.8))

        # Params panel
        a  = d.get("analysis", summary or {})
        lines = [
            "  PARAMETERS",
            "  " + "─"*18,
        ]
        if voltage:   lines.append(f"  Voltage   {voltage} V")
        if resistor:  lines.append(f"  Resistor  {resistor} Ω")
        if capacitor: lines.append(f"  Capacitor {capacitor}")
        if inductor:  lines.append(f"  Inductor  {inductor}")
        if a:
            lines += ["", "  ANALYSIS", "  " + "─"*18,
                      f"  Peak     {a.get('peak_V','—')} V",
                      f"  Trough   {a.get('trough_V','—')} V",
                      f"  Ripple   {a.get('ripple_V','—')} V",
                      f"  Mean     {a.get('mean_V','—')} V",
                      f"  Freq     {a.get('frequency_Hz','—')} Hz",
                      f"  Points   {a.get('data_points','—')}"]
        axp.text(0, 0.95, "\n".join(lines), transform=axp.transAxes,
                 fontsize=7.2, verticalalignment="top",
                 fontfamily="monospace", color="#7dd3fc",
                 bbox=dict(boxstyle="round,pad=0.6", facecolor="#050a12",
                           edgecolor="#1e3a5f", alpha=0.95))
    else:
        fig, axes = plt.subplots(n, 1, figsize=(11, 3.2*n), facecolor="#080e1a")
        fig.subplots_adjust(hspace=0.45, left=0.07, right=0.97,
                            top=0.95, bottom=0.06)
        if n == 1: axes = [axes]
        for idx, (name, d) in enumerate(data_dict.items()):
            ax = axes[idx]
            c  = colors[idx % len(colors)]
            ax.plot(d["x"], d["y"], color=c, linewidth=1.4)
            ax.set_facecolor("#080e1a")
            ax.spines[["top","right"]].set_visible(False)
            for spine in ["left","bottom"]:
                ax.spines[spine].set_color("#1e3a5f")
            ax.tick_params(colors="#475569", labelsize=7)
            ax.set_xlabel("Time (s)", color="#475569", fontsize=7.5)
            ax.set_ylabel("Voltage (V)", color="#475569", fontsize=7.5)
            ax.set_title(name, color="#f0f9ff", fontsize=9, fontweight="bold")
            ax.grid(True, alpha=0.15, color="#1e3a5f")
            a = d.get("analysis", {})
            if d["y"]:
                pv   = max(d["y"])
                pidx = d["y"].index(pv)
                ax.annotate(f"Peak: {pv:.3f}V",
                            xy=(d["x"][pidx], pv),
                            xytext=(10, -16), textcoords="offset points",
                            fontsize=7.5, color=c,
                            arrowprops=dict(arrowstyle="->", color=c, lw=0.8))

    return fig

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 20px 0;'>
      <div style='font-size:1.1rem;font-weight:700;color:#f0f9ff;letter-spacing:-0.01em;'>
        ⚡ AI Circuit Sim
      </div>
      <div style='font-size:0.72rem;color:#475569;margin-top:2px;font-family:monospace;'>
        v2.0 · Ollama + NGSpice
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='card-title'>Component Values</div>", unsafe_allow_html=True)
    voltage   = st.text_input("Voltage (V)",   value="10",  key="v")
    resistor  = st.text_input("Resistor (Ω)",  value="1k",  key="r")
    capacitor = st.text_input("Capacitor (F)", value="10u", key="c")
    inductor  = st.text_input("Inductor (H)",  value="10m", key="l")

    st.markdown("---")
    st.markdown("<div class='card-title'>Supported Circuits</div>", unsafe_allow_html=True)
    circuits_list = [
        "RC low pass filter", "RC high pass filter", "RC integrator",
        "RC differentiator", "RL series circuit", "RL parallel circuit",
        "Diode rectifier", "Voltage divider", "Full wave rectifier",
        "Bandpass filter", "RLC series resonance", "RLC parallel",
        "Diode clamp", "Diode clipper", "Zener regulator",
        "Common emitter amplifier", "Wheatstone bridge",
        "RC oscillator", "Schmitt trigger"
    ]
    for c in circuits_list:
        st.markdown(f"<div style='font-size:0.72rem;color:#475569;padding:2px 0;'>· {c}</div>",
                    unsafe_allow_html=True)

    st.markdown("---")
    if st.session_state.history:
        st.markdown(f"<div style='font-size:0.72rem;color:#4ade80;'>✓ {len(st.session_state.history)} simulations this session</div>",
                    unsafe_allow_html=True)
        if st.button("Clear Session", key="clear"):
            st.session_state.history = []
            st.session_state.last_result = None
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class='main-header'>
  <div class='main-title'>⚡ AI Circuit Simulator</div>
  <div class='main-subtitle'>Natural language → NGSpice netlist → Waveform → AI Analysis</div>
  <div class='badge-row'>
    <span class='badge green'>19 Circuit Types</span>
    <span class='badge'>Ollama LLM</span>
    <span class='badge purple'>AI Fault Diagnosis</span>
    <span class='badge amber'>Multi-Circuit Compare</span>
    <span class='badge'>Session Memory</span>
    <span class='badge'>AI Optimizer</span>
  </div>
</div>
""", unsafe_allow_html=True)

if not SIM_AVAILABLE:
    st.error(f"ai_generator.py not found in same folder. Error: {SIM_ERROR}")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "⚡  Simulate",
    "📊  Compare",
    "🔧  AI Optimizer",
    "🕒  Session History",
    "ℹ️  About"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SIMULATE
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown("<div class='card-title'>Describe Your Circuit</div>",
                    unsafe_allow_html=True)

        # Quick select buttons — set BEFORE text_input so value is ready
        st.markdown("<div style='font-size:0.7rem;color:#475569;margin:0 0 6px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;'>Quick select</div>",
                    unsafe_allow_html=True)
        qcols = st.columns(4)
        quick = ["RC low pass filter", "Diode rectifier",
                 "Bandpass filter", "RL series circuit"]
        for i, q in enumerate(quick):
            if qcols[i].button(q, key=f"q{i}"):
                st.session_state.quick_select = q

        # Text input — uses quick_select as default if set
        default_val = st.session_state.quick_select or ""
        circuit_input = st.text_input(
            "", value=default_val,
            placeholder="e.g.  RC low pass filter  /  bandpass filter  /  diode rectifier",
            key="circuit_input_box", label_visibility="collapsed"
        )
        # Clear quick_select after use
        if st.session_state.quick_select:
            st.session_state.quick_select = ""

        simulate_btn = st.button("▶  Run Simulation", key="sim_btn")

    with col2:
        st.markdown("<div class='card-title'>Component Values</div>",
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class='card' style='padding:12px 16px;'>
          <div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:0.82rem;'>
            <div><span style='color:#475569;'>Voltage</span><br><span style='color:#00d4ff;font-family:monospace;font-weight:600;'>{voltage} V</span></div>
            <div><span style='color:#475569;'>Resistor</span><br><span style='color:#4ade80;font-family:monospace;font-weight:600;'>{resistor} Ω</span></div>
            <div><span style='color:#475569;'>Capacitor</span><br><span style='color:#fbbf24;font-family:monospace;font-weight:600;'>{capacitor} F</span></div>
            <div><span style='color:#475569;'>Inductor</span><br><span style='color:#a5b4fc;font-family:monospace;font-weight:600;'>{inductor} H</span></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Run simulation ──────────────────────────────────────────────────────
    if simulate_btn and circuit_input.strip():

        # Progress container
        progress_container = st.empty()
        status_container   = st.empty()
        time_container     = st.empty()

        def update_progress(pct, step, detail="", color="#00d4ff"):
            progress_container.markdown(f"""
            <div style='background:#0a1628;border:1px solid #1e3a5f;border-radius:10px;padding:16px 20px;margin:8px 0;'>
              <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;'>
                <span style='font-size:0.88rem;font-weight:700;color:#f0f9ff;'>{step}</span>
                <span style='font-size:1rem;font-weight:700;color:{color};font-family:monospace;'>{pct}%</span>
              </div>
              <div style='background:#050a12;border-radius:6px;height:8px;overflow:hidden;'>
                <div style='height:100%;width:{pct}%;background:linear-gradient(90deg,{color},{color}aa);border-radius:6px;transition:width 0.3s;'></div>
              </div>
              <div style='font-size:0.75rem;color:#475569;margin-top:8px;font-family:monospace;'>{detail}</div>
            </div>
            """, unsafe_allow_html=True)

        import time
        start_time = time.time()

        def update_time():
            elapsed = int(time.time() - start_time)
            time_container.markdown(
                f"<div style='font-size:0.72rem;color:#334155;font-family:monospace;text-align:right;'>⏱ {elapsed}s elapsed</div>",
                unsafe_allow_html=True
            )

        # ── Step 1: AI Intent ──
        update_progress(10, "Step 1/5 — AI analyzing circuit type...",
                        f"Sending '{circuit_input}' to Ollama for intent classification", "#7dd3fc")
        update_time()
        intent = sim.ai_classify_intent(circuit_input) if SIM_AVAILABLE else {
            "analysis_type": "tran", "tran_step": "1u", "tran_stop": "20m",
            "complexity": "simple", "reason": "Default"
        }
        analysis_type = intent.get("analysis_type", "tran").upper()
        complexity    = intent.get("complexity", "simple")

        # ── Step 2: Netlist ──
        update_progress(30, "Step 2/5 — Generating NGSpice netlist...",
                        f"Analysis type: {analysis_type} | Complexity: {complexity} | Trying template match first", "#00d4ff")
        update_time()

        # Try template first (fast)
        netlist = sim.template_fallback(circuit_input, voltage, resistor, capacitor, inductor)
        if netlist:
            update_progress(40, "Step 2/5 — Template matched!",
                            f"Using pre-verified template — skipping AI netlist generation (faster)", "#4ade80")
        else:
            update_progress(35, "Step 2/5 — No template found, using AI netlist generator...",
                            "Asking Ollama to write NGSpice netlist from scratch", "#fbbf24")
            update_time()
            netlist = sim.ai_generate_netlist(
                circuit_input, voltage, resistor, capacitor, inductor, intent
            ) if SIM_AVAILABLE else None
            if netlist:
                netlist = sim.ai_tune_sim_params(netlist, intent)
                valid, reason = sim.validate_netlist(netlist)
                if not valid:
                    netlist = None

        if not netlist:
            progress_container.empty()
            status_container.empty()
            time_container.empty()
            st.error("Could not generate netlist. Try a more specific circuit name like 'RC low pass filter'.")
            st.stop()

        # ── Step 3: NGSpice ──
        update_progress(55, "Step 3/5 — Running NGSpice simulation...",
                        "Solving circuit equations — this is the fast part!", "#4ade80")
        update_time()
        sim.save_netlist(netlist)
        result_proc = sim.run_simulation()

        x, y = sim.parse_output() if result_proc and result_proc.returncode == 0 else ([], [])
        summary = sim.analyze_waveform(x, y) if x and y else {}

        if not x or not y:
            progress_container.empty()
            status_container.empty()
            time_container.empty()
            st.warning("Simulation produced no output. Check NGSpice installation.")
            st.code(netlist)
            st.stop()

        update_progress(65, "Step 3/5 — Simulation complete!",
                        f"{len(x):,} data points collected | Peak: {summary.get('peak_V','?')}V | Ripple: {summary.get('ripple_V','?')}V", "#4ade80")
        update_time()

        # ── Step 4: AI Diagnosis ──
        update_progress(72, "Step 4/5 — AI fault diagnosis...",
                        "Analyzing waveform data and checking circuit behavior", "#a5b4fc")
        update_time()
        diagnosis = sim.ai_fault_diagnosis(
            summary, circuit_input, voltage, resistor, capacitor, inductor
        ) if SIM_AVAILABLE else "AI unavailable."

        # ── Step 5: AI Explanation + Recommendations ──
        update_progress(85, "Step 5/5 — AI generating explanation...",
                        "Writing circuit explanation and next circuit recommendations", "#a5b4fc")
        update_time()
        explanation     = sim.ai_explain_circuit(netlist, circuit_input) if SIM_AVAILABLE else "AI unavailable."

        update_progress(95, "Step 5/5 — AI generating recommendations...",
                        "Suggesting 3 related circuits to explore next", "#a5b4fc")
        update_time()
        recommendations = sim.ai_recommend_circuits(circuit_input) if SIM_AVAILABLE else "AI unavailable."

        # ── Done ──
        total_time = int(time.time() - start_time)
        update_progress(100, f"✓ Complete in {total_time}s!",
                        f"Circuit: {circuit_input} | {len(x):,} data points | All AI analysis ready", "#4ade80")
        update_time()
        import time as _t; _t.sleep(0.8)

        # Clear progress UI
        progress_container.empty()
        status_container.empty()
        time_container.empty()

        # Save result
        result = {
            "circuit":         circuit_input,
            "netlist":         netlist,
            "x":               x,
            "y":               y,
            "summary":         summary,
            "diagnosis":       diagnosis,
            "explanation":     explanation,
            "recommendations": recommendations,
            "timestamp":       datetime.datetime.now().strftime("%H:%M:%S"),
            "voltage":         voltage,
            "resistor":        resistor,
            "capacitor":       capacitor,
            "inductor":        inductor,
            "total_time":      total_time,
        }
        st.session_state.last_result = result
        st.session_state.history.append(result)

    elif simulate_btn:
        st.warning("Please enter a circuit name first.")

    # ── Always show last result ─────────────────────────────────────────────
    if st.session_state.last_result:
        r = st.session_state.last_result
        summary  = r["summary"]
        x        = r["x"]
        y        = r["y"]
        netlist  = r["netlist"]

        st.success(f"✓ {r['circuit']}  —  {summary.get('data_points',0):,} data points  —  completed in {r.get('total_time','?')}s  —  {r['timestamp']}")

        # ── Metrics ──────────────────────────────────────────────────────
        st.markdown(f"""
        <div class='metric-grid'>
          <div class='metric-box green'>
            <div class='metric-value'>{summary.get('peak_V','—')}</div>
            <div class='metric-label'>Peak (V)</div>
          </div>
          <div class='metric-box red'>
            <div class='metric-value'>{summary.get('trough_V','—')}</div>
            <div class='metric-label'>Trough (V)</div>
          </div>
          <div class='metric-box amber'>
            <div class='metric-value'>{summary.get('ripple_V','—')}</div>
            <div class='metric-label'>Ripple (V)</div>
          </div>
          <div class='metric-box purple'>
            <div class='metric-value'>{summary.get('mean_V','—')}</div>
            <div class='metric-label'>Mean (V)</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Waveform ──────────────────────────────────────────────────────
        st.markdown("<div class='card-title'>Waveform</div>", unsafe_allow_html=True)
        fig = make_waveform_fig(
            {r["circuit"]: {"x": x, "y": y, "analysis": summary}},
            voltage=r["voltage"], resistor=r["resistor"],
            capacitor=r["capacitor"], inductor=r["inductor"],
            summary=summary
        )
        st.pyplot(fig, use_container_width=True)

        # ── AI Analysis ───────────────────────────────────────────────────
        st.markdown("<div class='card-title' style='margin-top:20px;'>AI Analysis</div>",
                    unsafe_allow_html=True)
        ai_tabs = st.tabs(["🔍 Fault Diagnosis", "📖 Explanation", "💡 What to try next", "📄 Netlist"])

        with ai_tabs[0]:
            st.markdown(f"<div class='ai-box'>{r['diagnosis']}</div>", unsafe_allow_html=True)
        with ai_tabs[1]:
            st.markdown(f"<div class='ai-box green'>{r['explanation']}</div>", unsafe_allow_html=True)
        with ai_tabs[2]:
            st.markdown(f"<div class='ai-box purple'>{r['recommendations']}</div>", unsafe_allow_html=True)
        with ai_tabs[3]:
            st.markdown(f"<div class='netlist-block'>{r['netlist']}</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — COMPARE
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("<div class='card-title'>Multi-Circuit Comparison</div>",
                unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.82rem;color:#475569;margin-bottom:16px;'>Simulate 2–4 circuits and compare their waveforms and parameters side by side.</div>",
                unsafe_allow_html=True)

    ccols = st.columns(2)
    c1 = ccols[0].text_input("Circuit 1", value="RC low pass filter", key="cc1")
    c2 = ccols[1].text_input("Circuit 2", value="RC high pass filter", key="cc2")
    c3 = ccols[0].text_input("Circuit 3 (optional)", value="", key="cc3")
    c4 = ccols[1].text_input("Circuit 4 (optional)", value="", key="cc4")

    compare_btn = st.button("▶  Compare Circuits", key="cmp_btn")

    if compare_btn:
        circuits_to_compare = [c for c in [c1, c2, c3, c4] if c.strip()]
        if len(circuits_to_compare) < 2:
            st.warning("Enter at least 2 circuits.")
        else:
            results_dict = {}
            progress = st.progress(0, text="Starting comparison...")
            for i, circ in enumerate(circuits_to_compare):
                progress.progress((i) / len(circuits_to_compare),
                                  text=f"Simulating: {circ}...")
                x, y, netlist, summary = run_sim(circ, voltage, resistor, capacitor, inductor)
                if x and y:
                    results_dict[circ] = {"x": x, "y": y, "analysis": summary}
                    st.session_state.history.append({
                        "circuit": circ, "netlist": netlist,
                        "x": x, "y": y, "summary": summary,
                        "diagnosis": "", "explanation": "", "recommendations": "",
                        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                        "voltage": voltage, "resistor": resistor,
                        "capacitor": capacitor, "inductor": inductor,
                    })
            progress.progress(1.0, text="Done!")

            if results_dict:
                st.markdown("<div class='card-title' style='margin-top:20px;'>Comparison Waveforms</div>",
                            unsafe_allow_html=True)
                fig = make_waveform_fig(results_dict)
                st.pyplot(fig, use_container_width=True)

                # Comparison table
                st.markdown("<div class='card-title' style='margin-top:20px;'>Side-by-Side Metrics</div>",
                            unsafe_allow_html=True)
                cols = st.columns(len(results_dict))
                colors_hex = ["#00d4ff","#f87171","#4ade80","#fbbf24"]
                for idx, (name, data) in enumerate(results_dict.items()):
                    a = data["analysis"]
                    cols[idx].markdown(f"""
                    <div class='card' style='border-top:2px solid {colors_hex[idx]};'>
                      <div style='font-size:0.78rem;font-weight:700;color:{colors_hex[idx]};margin-bottom:12px;'>{name}</div>
                      <div style='font-size:0.8rem;line-height:2;font-family:monospace;color:#94a3b8;'>
                        Peak: <span style='color:#4ade80;'>{a.get('peak_V','—')}V</span><br>
                        Ripple: <span style='color:#fbbf24;'>{a.get('ripple_V','—')}V</span><br>
                        Mean: <span style='color:#7dd3fc;'>{a.get('mean_V','—')}V</span><br>
                        Freq: <span style='color:#a5b4fc;'>{a.get('frequency_Hz','—')}Hz</span><br>
                        Points: <span style='color:#94a3b8;'>{a.get('data_points','—')}</span>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                # AI comparison
                st.markdown("<div class='card-title' style='margin-top:20px;'>AI Comparison Analysis</div>",
                            unsafe_allow_html=True)
                summaries = {n: d["analysis"] for n, d in results_dict.items()}
                ai_cmp = sim.ai_call(
                    f"Compare these circuits:\n{json.dumps(summaries, indent=2)}\n\n"
                    "In 4-5 sentences: key differences, which is better for filtering, why.",
                    system_text="Electronics expert. Be specific with numbers."
                ) if SIM_AVAILABLE else "AI unavailable."
                st.markdown(f"<div class='ai-box'>{ai_cmp}</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — AI OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("<div class='card-title'>AI Component Optimizer</div>",
                unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.82rem;color:#475569;margin-bottom:16px;'>Describe a design goal — AI will suggest component values and iteratively improve the circuit.</div>",
                unsafe_allow_html=True)

    opt_cols = st.columns(2)
    opt_goal    = opt_cols[0].text_input("Design Goal",
                   placeholder="e.g.  reduce ripple below 0.5V  /  maximize peak voltage",
                   key="opt_goal")
    opt_circuit = opt_cols[1].text_input("Circuit Type",
                   placeholder="e.g.  RC low pass filter",
                   key="opt_circuit")
    opt_iters   = st.slider("Optimization iterations", 1, 5, 3, key="opt_iters")
    opt_btn     = st.button("🔧  Start AI Optimization", key="opt_btn")

    if opt_btn and opt_goal and opt_circuit:
        convo = {
            "input_voltage":   voltage,
            "load_resistance": resistor,
            "capacitor_value": capacitor,
            "inductor_value":  inductor,
            "circuit_type":    "rlc"
        }

        iteration_results = []
        progress_bar = st.progress(0, text="Starting optimization...")

        for i in range(opt_iters):
            progress_bar.progress((i) / opt_iters, text=f"Iteration {i+1}/{opt_iters}...")

            intent  = sim.ai_classify_intent(opt_circuit)
            netlist = sim.ai_generate_netlist(
                opt_circuit,
                convo["input_voltage"], convo["load_resistance"],
                convo["capacitor_value"], convo["inductor_value"],
                intent
            )
            if not netlist:
                netlist = sim.build_netlist_from_conversation(convo)

            netlist = sim.ai_tune_sim_params(netlist, intent)
            valid, reason = sim.validate_netlist(netlist)
            if not valid:
                st.warning(f"Iteration {i+1}: Invalid netlist — {reason}")
                break

            sim.save_netlist(netlist)
            result = sim.run_simulation()
            if not result or result.returncode != 0:
                st.warning(f"Iteration {i+1}: Simulation failed.")
                break

            x, y = sim.parse_output()
            if not x or not y:
                st.warning(f"Iteration {i+1}: No output.")
                break

            summary = sim.analyze_waveform(x, y)
            iteration_results.append({
                "iteration": i + 1,
                "values":    dict(convo),
                "summary":   summary,
                "x": x, "y": y
            })

            if i < opt_iters - 1:
                suggestions = sim.ai_call(
                    f"Circuit: {opt_circuit}\nGoal: {opt_goal}\n"
                    f"Waveform: {json.dumps(summary)}\nValues: {json.dumps(convo)}\n\n"
                    "Return JSON with improved values: input_voltage, load_resistance, "
                    "capacitor_value, inductor_value",
                    system_text="Circuit optimization expert. Return ONLY JSON.",
                    expect_json=True
                )
                if isinstance(suggestions, dict) and "load_resistance" in suggestions:
                    for key in ["input_voltage","load_resistance","capacitor_value","inductor_value"]:
                        if key in suggestions and suggestions[key]:
                            convo[key] = str(suggestions[key])

        progress_bar.progress(1.0, text="Optimization complete!")

        if iteration_results:
            # Show iteration metrics
            st.markdown("<div class='card-title' style='margin-top:20px;'>Iteration Results</div>",
                        unsafe_allow_html=True)
            iter_cols = st.columns(len(iteration_results))
            for i, ir in enumerate(iteration_results):
                a = ir["summary"]
                iter_cols[i].markdown(f"""
                <div class='card' style='border-top:2px solid #0066ff;text-align:center;'>
                  <div style='font-size:0.72rem;color:#475569;font-weight:700;margin-bottom:8px;'>ITERATION {ir['iteration']}</div>
                  <div style='font-size:1.1rem;color:#4ade80;font-family:monospace;font-weight:700;'>{a.get('peak_V','—')}V</div>
                  <div style='font-size:0.7rem;color:#475569;'>peak</div>
                  <div style='font-size:0.9rem;color:#fbbf24;font-family:monospace;margin-top:6px;'>{a.get('ripple_V','—')}V</div>
                  <div style='font-size:0.7rem;color:#475569;'>ripple</div>
                  <div style='margin-top:10px;font-size:0.7rem;color:#334155;line-height:1.8;font-family:monospace;'>
                    R={ir['values'].get('load_resistance','—')}<br>
                    C={ir['values'].get('capacitor_value','—')}<br>
                    L={ir['values'].get('inductor_value','—')}
                  </div>
                </div>
                """, unsafe_allow_html=True)

            # Before vs After comparison
            if len(iteration_results) >= 2:
                st.markdown("<div class='card-title' style='margin-top:20px;'>Before vs After</div>",
                            unsafe_allow_html=True)
                first = iteration_results[0]["summary"]
                last  = iteration_results[-1]["summary"]
                ba_cols = st.columns(3)
                for i, (metric, label) in enumerate([
                    ("peak_V","Peak Voltage"),
                    ("ripple_V","Ripple"),
                    ("mean_V","Mean Voltage")
                ]):
                    b = first.get(metric, 0)
                    a = last.get(metric, 0)
                    arrow = "▲" if a > b else "▼"
                    color = "#4ade80" if (metric == "peak_V" and a > b) or (metric == "ripple_V" and a < b) else "#f87171"
                    ba_cols[i].markdown(f"""
                    <div class='card' style='text-align:center;'>
                      <div class='card-title'>{label}</div>
                      <div style='font-size:0.9rem;color:#475569;font-family:monospace;'>{b}V</div>
                      <div style='font-size:1.2rem;color:{color};margin:4px 0;'>{arrow}</div>
                      <div style='font-size:1rem;color:{color};font-family:monospace;font-weight:700;'>{a}V</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Overlay waveforms
            st.markdown("<div class='card-title' style='margin-top:20px;'>Waveform Evolution</div>",
                        unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(11, 3.5), facecolor="#080e1a")
            ax.set_facecolor("#080e1a")
            ax.spines[["top","right"]].set_visible(False)
            for s in ["left","bottom"]: ax.spines[s].set_color("#1e3a5f")
            ax.tick_params(colors="#475569", labelsize=7)
            ax.set_xlabel("Time (s)", color="#475569", fontsize=8)
            ax.set_ylabel("Voltage (V)", color="#475569", fontsize=8)
            ax.set_title("Optimization — Waveform Evolution", color="#f0f9ff",
                         fontsize=9, fontweight="bold")
            ax.grid(True, alpha=0.12, color="#1e3a5f")
            clrs = ["#475569","#7dd3fc","#00d4ff","#4ade80","#fbbf24"]
            for ir in iteration_results:
                ax.plot(ir["x"], ir["y"],
                        color=clrs[min(ir["iteration"]-1, len(clrs)-1)],
                        linewidth=1.3, alpha=0.85,
                        label=f"Iteration {ir['iteration']}")
            ax.legend(fontsize=7.5, facecolor="#0a0e17", edgecolor="#1e3a5f",
                     labelcolor="#94a3b8")
            st.pyplot(fig, use_container_width=True)

    elif opt_btn:
        st.warning("Please enter both a design goal and circuit type.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SESSION HISTORY
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("<div class='card-title'>Session History</div>", unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown("<div style='color:#475569;font-size:0.85rem;padding:20px;'>No simulations yet. Run a simulation in the Simulate tab.</div>",
                    unsafe_allow_html=True)
    else:
        # Header row
        st.markdown("""
        <div class='history-row' style='border-bottom:1px solid #1e3a5f;'>
          <div class='history-circuit' style='color:#475569;font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;'>Circuit</div>
          <div class='history-peak' style='color:#475569;font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;'>Peak V</div>
          <div class='history-ripple' style='color:#475569;font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;'>Ripple</div>
          <div class='history-time' style='color:#475569;font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;'>Time</div>
        </div>
        """, unsafe_allow_html=True)

        for r in reversed(st.session_state.history):
            s = r.get("summary", {})
            st.markdown(f"""
            <div class='history-row'>
              <div class='history-circuit'>⚡ {r['circuit']}</div>
              <div class='history-peak'>{s.get('peak_V','—')} V</div>
              <div class='history-ripple'>{s.get('ripple_V','—')} V</div>
              <div class='history-time'>{r.get('timestamp','—')}</div>
            </div>
            """, unsafe_allow_html=True)

        # AI session summary
        st.markdown("<div class='card-title' style='margin-top:24px;'>AI Learning Summary</div>",
                    unsafe_allow_html=True)
        if st.button("Generate Session Summary", key="sess_sum"):
            history_data = [
                {"circuit": r["circuit"], "summary": r.get("summary",{})}
                for r in st.session_state.history
            ]
            sess_summary = sim.ai_call(
                f"Student simulated these circuits:\n{json.dumps(history_data, indent=2)}\n\n"
                "Write a 5-6 sentence learning summary: concepts explored, patterns, what to study next.",
                system_text="Electronics teacher. Be encouraging and specific."
            ) if SIM_AVAILABLE else "AI unavailable."
            st.markdown(f"<div class='ai-box purple'>{sess_summary}</div>",
                        unsafe_allow_html=True)

        # Export
        if st.button("Export Session JSON", key="export"):
            export_data = [
                {k: v for k, v in r.items() if k not in ["x","y"]}
                for r in st.session_state.history
            ]
            st.download_button(
                "Download session.json",
                data=json.dumps(export_data, indent=2),
                file_name=f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown("""
    <div class='card'>
      <div class='card-title'>About This Project</div>
      <div style='font-size:0.88rem;color:#94a3b8;line-height:1.9;'>
        This AI Circuit Simulator combines <strong style='color:#00d4ff;'>NGSpice</strong>
        (industry-standard circuit simulation) with a local
        <strong style='color:#a5b4fc;'>Ollama LLM (llama3/mistral)</strong>
        to create a natural language circuit simulation tool.
      </div>
    </div>
    """, unsafe_allow_html=True)

    feat_cols = st.columns(2)
    features = [
        ("⚡", "19 Circuit Templates", "RC, RL, RLC, diode, amplifier, filters and more — Level 1 & 2 circuits"),
        ("🤖", "AI Intent Classifier", "Automatically decides TRAN / AC / DC analysis type from your description"),
        ("🔍", "AI Fault Diagnosis", "Analyzes waveform and gives specific fix suggestions with actual component values"),
        ("🔧", "AI Optimizer", "Iteratively improves component values to meet your design goal"),
        ("📊", "Multi-Circuit Compare", "Simulate 2-4 circuits simultaneously and compare waveforms side by side"),
        ("🧠", "Session Memory", "AI reflects on all simulations in a session and gives a learning summary"),
        ("💾", "Auto-Save Results", "Every simulation saved as JSON in results/ folder with full waveform data"),
        ("📝", "Logging System", "Full logging to logs/ folder — every AI call and simulation step tracked"),
    ]
    for i, (icon, title, desc) in enumerate(features):
        col = feat_cols[i % 2]
        col.markdown(f"""
        <div class='card' style='margin-bottom:12px;'>
          <div style='font-size:1.1rem;margin-bottom:6px;'>{icon}
            <span style='font-size:0.88rem;font-weight:700;color:#f0f9ff;margin-left:6px;'>{title}</span>
          </div>
          <div style='font-size:0.78rem;color:#64748b;line-height:1.6;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='card' style='margin-top:8px;'>
      <div class='card-title'>How to Run</div>
      <div class='netlist-block'>pip install streamlit ollama matplotlib
streamlit run gui.py</div>
    </div>
    """, unsafe_allow_html=True)
