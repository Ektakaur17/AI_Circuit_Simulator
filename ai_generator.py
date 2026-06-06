"""
AI Circuit Simulator  —  Final Merged Version
=============================================
Best of both versions:
  Tumhara: logging, argparse, save_results, sanitize_input fix, optimizer fix
  Mera:    19 circuit templates, expanded keywords, improved Mode 2

AI Backend: Groq API (primary, fast, free) → Ollama/Mistral (fallback, local)

Requirements:
    pip install ollama matplotlib groq python-dotenv
    NGSpice installed
    Groq API key in .env file  OR  Ollama running locally
"""

import os
import re
import json
import subprocess
import matplotlib.pyplot as plt
import datetime
import argparse
import logging

# ── Load .env file for API keys ────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional — key can also be set as env variable

# ── Groq import ────────────────────────────────────────────────────────────────
try:
    from groq import Groq
    GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
    GROQ_AVAILABLE  = bool(GROQ_API_KEY)
    if GROQ_AVAILABLE:
        groq_client = Groq(api_key=GROQ_API_KEY)
except ImportError:
    GROQ_AVAILABLE = False
    groq_client    = None

# ── Ollama import (fallback) ───────────────────────────────────────────────────
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING  (file + terminal)
# ══════════════════════════════════════════════════════════════════════════════

def setup_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir,
        f"simulator_{datetime.datetime.now().strftime('%Y%m%d')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ── AI backend status ──────────────────────────────────────────────────────────
if GROQ_AVAILABLE:
    logger.info("AI Backend: Groq API (primary) + Ollama (fallback)")
elif OLLAMA_AVAILABLE:
    logger.info("AI Backend: Ollama only (Groq API key not set)")
else:
    logger.warning("No AI backend available. Add GROQ_API_KEY to .env file.")

# ── Config ─────────────────────────────────────────────────────────────────────
CONFIG = {
    "ngspice_path":  "ngspice",
    "groq_model":    "llama-3.3-70b-versatile",  # Current Groq model (replaces deprecated llama3-70b-8192)
    "ollama_model":  "mistral",                   # Ollama fallback — local
}

NETLIST_FILE = "circuit.cir"
OUTPUT_FILE  = "output.txt"

DEFAULT_VALUES = {
    "voltage":   "10",
    "resistor":  "1k",
    "capacitor": "10u",
    "inductor":  "10m"
}

# ── Templates (Level 1 + Level 2 circuits) ────────────────────────────────────
TEMPLATES = {

    # ── Level 1 ────────────────────────────────────────────────────────────────

    "rc": """\
* RC Low Pass Filter
V1 1 0 PULSE(0 {voltage} 0 1u 1u 5m 10m)
R1 1 2 {resistor}
C1 2 0 {capacitor}
.tran 10u 20m
.print tran v(2)
.end
""",
    "rc_highpass": """\
* RC High Pass Filter
V1 1 0 PULSE(0 {voltage} 0 1u 1u 5m 10m)
C1 1 2 {capacitor}
R1 2 0 {resistor}
.tran 10u 20m
.print tran v(2)
.end
""",
    "integrator": """\
* RC Integrator
V1 1 0 PULSE(0 {voltage} 0 1u 1u 5m 10m)
R1 1 2 {resistor}
C1 2 0 {capacitor}
.tran 10u 20m
.print tran v(2)
.end
""",
    "differentiator": """\
* RC Differentiator
V1 1 0 PULSE(0 {voltage} 0 1u 1u 5m 10m)
C1 1 2 {capacitor}
R1 2 0 {resistor}
.tran 1u 5m
.print tran v(2)
.end
""",
    "rl": """\
* RL Series Circuit
V1 1 0 SIN(0 {voltage} 50)
R1 1 2 {resistor}
L1 2 0 {inductor}
.tran 1u 5m
.print tran v(2)
.end
""",
    "rl_parallel": """\
* RL Parallel Circuit
V1 1 0 SIN(0 {voltage} 50)
R1 1 2 {resistor}
L1 1 2 {inductor}
.tran 1u 5m
.print tran v(2)
.end
""",
    "diode": """\
* Half Wave Diode Rectifier
V1 1 0 SIN(0 {voltage} 50)
D1 1 2 D
R1 2 0 {resistor}
.model D D
.tran 1u 20m
.print tran v(2)
.end
""",
    "voltage_divider": """\
* Voltage Divider
V1 1 0 DC {voltage}
R1 1 2 {resistor}
R2 2 0 {resistor}
.op
.print op v(2)
.end
""",

    # ── Level 2 ────────────────────────────────────────────────────────────────

    "full_wave_rectifier": """\
* Full Wave Bridge Rectifier with Capacitor Filter
V1 1 0 SIN(0 {voltage} 50)
D1 1 3 D
D2 0 3 D
D3 4 1 D
D4 4 0 D
C1 3 4 {capacitor}
R1 3 4 {resistor}
.model D D(Is=1e-14 N=1 Rs=0.1)
.tran 100u 60m
.print tran v(3,4)
.end
""",
    "bandpass": """\
* RLC Bandpass Filter
V1 1 0 SIN(0 {voltage} 1k)
R1 1 2 {resistor}
L1 2 3 {inductor}
C1 3 0 {capacitor}
.tran 1u 10m
.print tran v(3)
.end
""",
    "rlc_series": """\
* RLC Series Resonance
V1 1 0 SIN(0 {voltage} 1k)
R1 1 2 {resistor}
L1 2 3 {inductor}
C1 3 0 {capacitor}
.tran 1u 10m
.print tran v(3)
.end
""",
    "rlc_parallel": """\
* RLC Parallel Resonance
V1 1 0 SIN(0 {voltage} 1k)
R1 1 0 {resistor}
L1 1 0 {inductor}
C1 1 0 {capacitor}
R2 2 1 100
.tran 1u 10m
.print tran v(1)
.end
""",
    "clamp": """\
* Diode Clamp Circuit
V1 1 0 SIN(0 {voltage} 50)
C1 1 2 {capacitor}
D1 0 2 D
R1 2 0 {resistor}
.model D D
.tran 1u 40m
.print tran v(2)
.end
""",
    "clipper": """\
* Diode Clipper Circuit
V1 1 0 SIN(0 {voltage} 50)
R1 1 2 {resistor}
D1 2 0 D
.model D D
.tran 1u 20m
.print tran v(2)
.end
""",
    "zener_regulator": """\
* Zener Voltage Regulator
V1 1 0 DC {voltage}
R1 1 2 {resistor}
D1 0 2 ZENER
R2 2 0 {resistor}
.model ZENER D(BV=5.1 IBV=0.001 RS=10)
.dc V1 0 15 0.1
.print dc v(2)
.end
""",
    "ce_amplifier": """\
* Common Emitter Amplifier (BJT NPN)
* VCC=12V, RC=4.7k, RE=1k, R1=47k, R2=10k
* Input: 10mV 1kHz sine, Output: amplified at node 6
VCC 1 0 DC 12
Vin 7 0 SIN(0 0.01 1000)
R1  1 2 47k
R2  2 0 10k
RC  1 3 4700
RE  5 0 1000
CE  5 0 100u
Cin 7 2 10u
Cout 3 6 10u
RL  6 0 10k
Q1  3 2 5 NPN
.model NPN NPN(BF=200 IS=1e-14 VAF=100)
.tran 1u 5m
.print tran v(7) v(6)
.end
""",
    "schmitt_trigger": """\
* Op-Amp Schmitt Trigger
V1 3 0 DC {voltage}
V2 0 4 DC {voltage}
Vin 1 0 SIN(0 {voltage} 50)
R1 1 2 {resistor}
R2 2 5 {resistor}
R3 5 0 {resistor}
XU1 2 5 3 4 5 OPAMP
.subckt OPAMP inp inn vp vn out
Rin inp inn 1Meg
Eout out 0 inp inn 100k
Rout out 0 1k
.ends OPAMP
.tran 1u 20m
.print tran v(5)
.end
""",
    "wheatstone": """\
* Wheatstone Bridge
V1 1 0 DC {voltage}
R1 1 2 {resistor}
R2 1 3 {resistor}
R3 2 4 {resistor}
R4 3 4 {resistor}
.dc V1 0 {voltage} 0.1
.print dc v(2) v(3) v(4)
.end
""",
    "rc_oscillator": """\
* RC Phase Shift Oscillator
V1 1 0 SIN(0 {voltage} 1k)
R1 1 2 {resistor}
C1 2 3 {capacitor}
R2 3 4 {resistor}
C2 4 5 {capacitor}
R3 5 0 {resistor}
C3 5 0 {capacitor}
.tran 1u 10m
.print tran v(5)
.end
""",
}

TEMPLATE_KEYWORDS = {
    # Level 1
    "integrator":        ["integrator", "rc integrator"],
    "differentiator":    ["differentiator", "rc differentiator"],
    "rc":                ["rc low pass", "rc lowpass", "low pass filter",
                         "lowpass filter", "rc filter", "rc circuit",
                         "rc network", "rc low"],
    "rc_highpass":       ["rc high pass", "high pass filter", "highpass filter",
                         "rc highpass", "high pass rc"],
    "rl":                ["rl series", "series rl", "rl circuit",
                         "rl series circuit", "resistor inductor"],
    "rl_parallel":       ["rl parallel", "parallel rl"],
    "diode":             ["half wave", "half-wave", "diode rectifier",
                         "simple rectifier", "diode circuit"],
    "voltage_divider":   ["voltage divider", "resistor divider",
                         "potential divider"],
    # Level 2
    "full_wave_rectifier": ["full wave", "full-wave", "bridge rectifier",
                            "full wave rectifier", "bridge circuit",
                            "full wave bridge", "bridge rectifier filter",
                            "full wave bridge rectifier", "capacitor filter",
                            "rectifier filter", "full wave with filter",
                            "bridge with capacitor"],
    "bandpass":          ["bandpass", "band pass", "band-pass filter",
                         "bandpass filter", "bp filter"],
    "rlc_series":        ["rlc series", "series rlc", "rlc circuit",
                         "rlc resonance", "series resonance"],
    "rlc_parallel":      ["rlc parallel", "parallel rlc",
                         "parallel resonance"],
    "clamp":             ["clamp", "clamper", "diode clamp",
                         "clamping circuit"],
    "clipper":           ["clipper", "clipping circuit", "diode clipper"],
    "zener_regulator":   ["zener", "zener regulator", "voltage regulator",
                         "zener diode", "regulator circuit",
                         "zener diode voltage regulator",
                         "zener voltage regulator",
                         "diode voltage regulator",
                         "voltage regulation", "zener clamp"],
    "ce_amplifier":      ["common emitter", "ce amplifier", "bjt amplifier",
                         "transistor amplifier", "npn amplifier"],
    "schmitt_trigger":   ["schmitt", "schmitt trigger", "hysteresis",
                         "comparator"],
    "wheatstone":        ["wheatstone", "wheatstone bridge", "bridge circuit",
                         "balanced bridge"],
    "rc_oscillator":     ["phase shift oscillator", "rc oscillator",
                         "rc phase shift", "phase shift"],
}

# ══════════════════════════════════════════════════════════════════════════════
# CLI ARGUMENTS
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="AI Circuit Simulator — Final",
        epilog="""
Examples:
  python ai_generator.py
  python ai_generator.py --circuit "RC low pass filter"
  python ai_generator.py --circuit "bandpass filter" --voltage 5 --resistor 470
  python ai_generator.py --circuit "diode rectifier" --no-plot
        """
    )
    parser.add_argument("--circuit",   type=str, help="Circuit name (skips interactive prompt)")
    parser.add_argument("--voltage",   type=str, default=None)
    parser.add_argument("--resistor",  type=str, default=None)
    parser.add_argument("--capacitor", type=str, default=None)
    parser.add_argument("--inductor",  type=str, default=None)
    parser.add_argument("--no-plot",   action="store_true", help="Skip graph generation")
    parser.add_argument("--mode",      type=str, default=None, choices=["1","2"])
    return parser.parse_args()

# ══════════════════════════════════════════════════════════════════════════════
# INPUT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def sanitize_input(text):
    """Fixed — uses regex patterns, won't ban words like 'system' in '555 timer system'."""
    BANNED_PATTERNS = [
        r"exec\s*\(", r"eval\s*\(", r"__import__\s*\(",
        r"os\.system\s*\(", r"subprocess\.(run|call|Popen)\s*\(",
        r"open\s*\(", r";\s*(rm|del|format|shutdown)",
    ]
    for pattern in BANNED_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            raise ValueError(f"Unsafe pattern detected: '{pattern}'")
    return text.strip()

def normalize_value(v):
    v = v.strip().lower().replace(" ", "")
    for unit in ["ohm", "ohms", "farad", "henry", "volt", "volts"]:
        v = v.replace(unit, "")
    return v or None

def get_user_parameters():
    print("\nPress ENTER to use defaults shown in brackets.\n")
    def ask(label, key):
        default = DEFAULT_VALUES[key]
        raw = input(f"  {label} [{default}]: ").strip()
        val = normalize_value(raw) if raw else None
        return val if val else default
    voltage   = ask("Voltage   (V)", "voltage")
    resistor  = ask("Resistor  (Ω)", "resistor")
    capacitor = ask("Capacitor (F)", "capacitor")
    inductor  = ask("Inductor  (H)", "inductor")
    logger.info("Parameters: V=%s R=%s C=%s L=%s", voltage, resistor, capacitor, inductor)
    print()
    return voltage, resistor, capacitor, inductor

# ══════════════════════════════════════════════════════════════════════════════
# AI CORE  —  3 AI-powered steps before simulation
# ══════════════════════════════════════════════════════════════════════════════

def ai_call(prompt_text, system_text=None, expect_json=False):
    """
    Central AI call — Groq first (fast, powerful), Ollama fallback (local).
    Returns string or dict depending on expect_json.
    """
    if system_text is None:
        system_text = "You are a helpful electronics engineering assistant. Be concise."

    def _parse(content):
        if expect_json:
            content = re.sub(r"```[a-z]*", "", content).replace("```", "").strip()
            return json.loads(content)
        return content

    # ── Try Groq first (llama3-70b — much better quality) ──
    if GROQ_AVAILABLE:
        try:
            response = groq_client.chat.completions.create(
                model=CONFIG["groq_model"],
                messages=[
                    {"role": "system", "content": system_text},
                    {"role": "user",   "content": prompt_text}
                ],
                temperature=0.2,
                max_tokens=2048
            )
            content = response.choices[0].message.content.strip()
            logger.info("AI: Groq (%s)", CONFIG["groq_model"])
            return _parse(content)
        except json.JSONDecodeError as e:
            logger.error("Groq JSON parse error: %s", e)
            return {} if expect_json else "AI parse error."
        except Exception as e:
            logger.warning("Groq failed (%s) — trying Ollama fallback", e)

    # ── Fallback: Ollama/Mistral (local) ──
    if OLLAMA_AVAILABLE:
        try:
            response = ollama.chat(
                model=CONFIG["ollama_model"],
                messages=[
                    {"role": "system", "content": system_text},
                    {"role": "user",   "content": prompt_text}
                ]
            )
            content = response["message"]["content"].strip()
            logger.info("AI: Ollama (%s)", CONFIG["ollama_model"])
            return _parse(content)
        except json.JSONDecodeError as e:
            logger.error("Ollama JSON parse error: %s", e)
            return {} if expect_json else "AI parse error."
        except Exception as e:
            logger.error("Ollama failed: %s", e)

    logger.error("No AI backend available")
    return {} if expect_json else "AI unavailable."


# ── STEP 1: AI Intent Classifier ───────────────────────────────────────────────
def ai_classify_intent(prompt):
    """
    AI decides:
      - analysis_type : tran | ac | dc | op
      - sim_time      : suggested .tran or .ac parameters
      - complexity    : simple | moderate | complex
    Returns dict.
    """
    system = (
        "You are an NGSpice simulation expert. "
        "Respond ONLY with a JSON object — no explanation, no markdown."
    )
    user = (
        f"Circuit description: '{prompt}'\n\n"
        "Return a JSON with these keys:\n"
        "  analysis_type : one of 'tran', 'ac', 'dc', 'op'\n"
        "  tran_step     : suggested time step string e.g. '1u'\n"
        "  tran_stop     : suggested stop time string e.g. '20m'\n"
        "  ac_start      : AC start freq e.g. '1' (only if analysis_type is ac)\n"
        "  ac_stop       : AC stop freq e.g. '1Meg' (only if analysis_type is ac)\n"
        "  complexity    : one of 'simple', 'moderate', 'complex'\n"
        "  reason        : one sentence why you chose this analysis type"
    )
    result = ai_call(user, system_text=system, expect_json=True)
    if not result:
        # safe defaults
        result = {
            "analysis_type": "tran",
            "tran_step": "1u",
            "tran_stop": "20m",
            "complexity": "simple",
            "reason": "Defaulting to transient analysis."
        }
    print(f"\n[AI] Intent: {result.get('analysis_type','tran').upper()} analysis "
          f"| Complexity: {result.get('complexity','?')} "
          f"| Reason: {result.get('reason','')}")
    return result


# ── STEP 2: AI Netlist Generator (PRIMARY) ─────────────────────────────────────
def ai_generate_netlist(prompt, voltage, resistor, capacitor, inductor, intent):
    """
    AI generates the full NGSpice netlist.
    This is the PRIMARY path — called for ALL circuits.
    """
    analysis = intent.get("analysis_type", "tran")
    tran_step = intent.get("tran_step", "1u")
    tran_stop = intent.get("tran_stop", "20m")
    ac_start  = intent.get("ac_start", "1")
    ac_stop   = intent.get("ac_stop", "1Meg")

    if analysis == "ac":
        analysis_line = f".ac dec 20 {ac_start} {ac_stop}"
        print_line    = ".print ac vdb(2) vp(2)"
    elif analysis == "dc":
        analysis_line = ".dc V1 0 15 0.1"
        print_line    = ".print dc v(2)"
    elif analysis == "op":
        analysis_line = ".op"
        print_line    = ".print op v(2)"
    else:
        analysis_line = f".tran {tran_step} {tran_stop}"
        print_line    = ".print tran v(2)"

    system = (
        "You are an NGSpice netlist expert. "
        "Output ONLY a valid NGSpice netlist. "
        "No explanation. No markdown. No code fences. "
        "Start with a comment line (*). End with .end"
    )
    user = (
        f"Generate an NGSpice netlist for: {prompt}\n\n"
        f"Component values to use:\n"
        f"  Voltage source: {voltage} V\n"
        f"  Resistor: {resistor}\n"
        f"  Capacitor: {capacitor}\n"
        f"  Inductor: {inductor}\n\n"
        f"Use this analysis line exactly:\n  {analysis_line}\n"
        f"Use this print line exactly:\n  {print_line}\n\n"
        "Rules:\n"
        "  1. Node 0 is always ground.\n"
        "  2. Output node must be node 2.\n"
        "  3. Include .model lines for any diodes or transistors.\n"
        "  4. The last line must be .end\n"
        "  5. No markdown, no explanation — ONLY the netlist."
    )
    print("[AI] Generating netlist for any circuit...")
    netlist = ai_call(user, system_text=system)
    if isinstance(netlist, dict):   # JSON parse went wrong
        return None
    # Strip markdown fences
    netlist = re.sub(r"```[a-z]*", "", netlist).replace("```", "").strip()
    # Remove lines that are commented-out .tran/.print (AI sometimes duplicates them)
    cleaned_lines = []
    for line in netlist.splitlines():
        stripped = line.strip()
        # Skip lines like "*.tran ..." or "*.print ..." — these are accidental comments
        if re.match(r'^\*\s*\.(tran|print|ac|dc|op)\b', stripped, re.IGNORECASE):
            continue
        # Fix title lines like "*RC Low Pass Filter*" — remove trailing *
        if stripped.startswith("*") and stripped.endswith("*") and len(stripped) > 2:
            line = line.rstrip("* ").rstrip()
        cleaned_lines.append(line)
    netlist = "\n".join(cleaned_lines).strip()
    return netlist


# ── STEP 3: Physics-based Sim-Param Tuner (no extra AI call) ──────────────────
def ai_tune_sim_params(netlist, intent):
    """
    Calculate simulation time from actual component values in the netlist.
    Uses physics (RC = time constant) instead of an extra AI call.
    This is faster and more reliable than asking AI again.
    """
    if intent.get("analysis_type") != "tran":
        return netlist

    # Parse R and C values from netlist
    r_val = _parse_component_value(netlist, r"^R\w+\s+\S+\s+\S+\s+(\S+)", default=1e3)
    c_val = _parse_component_value(netlist, r"^C\w+\s+\S+\s+\S+\s+(\S+)", default=10e-6)
    l_val = _parse_component_value(netlist, r"^L\w+\s+\S+\s+\S+\s+(\S+)", default=10e-3)

    # RC time constant
    tau_rc = r_val * c_val
    # RL time constant
    tau_rl = l_val / r_val if r_val > 0 else 0
    tau    = max(tau_rc, tau_rl)

    if tau <= 0:
        return netlist

    # Run for 5x time constant, step = tau/1000
    stop = tau * 5
    step = tau / 1000

    # Format nicely
    def fmt(val):
        if val >= 1:       return f"{val:.3g}"
        if val >= 1e-3:    return f"{val*1e3:.3g}m"
        if val >= 1e-6:    return f"{val*1e6:.3g}u"
        if val >= 1e-9:    return f"{val*1e9:.3g}n"
        return f"{val:.3e}"

    step_s = fmt(step)
    stop_s = fmt(stop)
    print(f"[AUTO] Sim params from physics: τ={fmt(tau)}s → step={step_s}, stop={stop_s}")
    netlist = re.sub(r"\.tran\s+\S+\s+\S+", f".tran {step_s} {stop_s}", netlist)
    return netlist


def _parse_component_value(netlist, pattern, default):
    """Extract and convert first matching component value from netlist."""
    SUFFIXES = {
        "t": 1e12, "g": 1e9, "meg": 1e6, "k": 1e3,
        "m": 1e-3, "u": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15
    }
    for line in netlist.splitlines():
        m = re.match(pattern, line.strip(), re.IGNORECASE)
        if m:
            raw = m.group(1).lower().strip()
            # Try direct float
            try:
                return float(raw)
            except ValueError:
                pass
            # Try with suffix
            for suffix, mult in sorted(SUFFIXES.items(), key=lambda x: -len(x[0])):
                if raw.endswith(suffix):
                    try:
                        return float(raw[:-len(suffix)]) * mult
                    except ValueError:
                        pass
    return default

# ══════════════════════════════════════════════════════════════════════════════
# NETLIST VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_netlist(netlist):
    """
    Validates NGSpice netlist for common AI-generated errors.
    Checks: .end present, valid source, passive component,
    voltage source has correct 3-part syntax (name n1 n2 value).
    """
    if not netlist or not netlist.strip():
        return False, "Netlist is empty."

    lines  = [l.strip() for l in netlist.strip().splitlines() if l.strip()]
    code   = [l for l in lines if not l.startswith("*") and not l.startswith("$")]

    if not any(l.lower() == ".end" for l in code):
        return False, "Missing .end"

    # Check voltage/current source exists AND has valid syntax (min 4 parts: V name n1 n2 value)
    has_source = False
    for l in code:
        if l and l[0].isalpha() and l[0].upper() in ("V", "I"):
            parts = l.split()
            if len(parts) >= 4:
                # n1 and n2 must be valid node names (not just one number)
                has_source = True
                break
    if not has_source:
        return False, "No valid voltage/current source found (needs: Vname n1 n2 value)."

    if not any(l[0].upper() in ("R","C","L","D","Q","M","E","F","G","H","J","K")
               for l in code if l and l[0].isalpha()):
        return False, "No passive/active component found."

    # Check node 0 (ground) is referenced somewhere
    all_text = " ".join(code)
    if " 0 " not in all_text and not any(l.endswith(" 0") for l in code):
        return False, "Node 0 (ground) not found — circuit has no ground reference."

    return True, ""

# ── Fallback: keyword template ─────────────────────────────────────────────────
def template_fallback(prompt, voltage, resistor, capacitor, inductor):
    """
    Try to match prompt to a known template.
    First tries multi-word keywords, then single-word fallbacks.
    """
    p = prompt.lower()

    # Multi-word keyword match
    for key, keywords in TEMPLATE_KEYWORDS.items():
        if any(kw in p for kw in keywords):
            print(f"[FALLBACK] Using template: {key}")
            return TEMPLATES[key].format(
                voltage=voltage, resistor=resistor,
                capacitor=capacitor, inductor=inductor
            )

    # Single-word fallback for common short prompts
    words = p.split()
    if "rc" in words:
        print("[FALLBACK] Using template: rc")
        return TEMPLATES["rc"].format(voltage=voltage, resistor=resistor,
                                      capacitor=capacitor, inductor=inductor)
    if "rl" in words:
        print("[FALLBACK] Using template: rl")
        return TEMPLATES["rl"].format(voltage=voltage, resistor=resistor,
                                      capacitor=capacitor, inductor=inductor)
    if "diode" in words or "rectifier" in words:
        print("[FALLBACK] Using template: diode")
        return TEMPLATES["diode"].format(voltage=voltage, resistor=resistor,
                                         capacitor=capacitor, inductor=inductor)
    return None

# ══════════════════════════════════════════════════════════════════════════════
# FILE / SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def save_netlist(netlist):
    with open(NETLIST_FILE, "w") as f:
        f.write(netlist)
    print(f"[SIM] Netlist saved → {NETLIST_FILE}")

def run_simulation():
    if not os.path.exists(NETLIST_FILE):
        print("[SIM] Netlist file not found.")
        return None
    try:
        result = subprocess.run(
            [CONFIG["ngspice_path"], "-b", NETLIST_FILE, "-o", OUTPUT_FILE],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            print(f"[SIM] NGSpice error (code {result.returncode}):")
            print(result.stderr[:600])
        return result
    except FileNotFoundError:
        print("[SIM] NGSpice not found. Install: https://ngspice.sourceforge.net/")
        return None
    except subprocess.TimeoutExpired:
        print("[SIM] Simulation timed out (30s).")
        return None

def parse_output():
    """
    Parse NGSpice output for tran, dc, and op analysis.
    Handles single and multi-column output (e.g. v(7) v(6)).
    Always uses the LAST voltage column as y (output signal).
    Returns (x, y) lists.
    """
    x, y = [], []
    if not os.path.exists(OUTPUT_FILE):
        return x, y
    with open(OUTPUT_FILE, "r", errors="ignore") as f:
        start = False
        for line in f:
            # Transient / DC analysis header
            if "Index" in line and ("time" in line.lower() or "v-sweep" in line.lower()):
                start = True
                continue
            # Operating point — single value
            if "v(" in line.lower() and "=" in line:
                parts = line.split("=")
                if len(parts) == 2:
                    try:
                        y.append(float(parts[1].strip()))
                        x.append(0.0)
                    except ValueError:
                        pass
                continue
            if start:
                parts = line.split()
                # Need at least index + time + one voltage column
                if len(parts) >= 3:
                    try:
                        x.append(float(parts[1]))
                        # Use last column — for CE amp this is v(6) output
                        y.append(float(parts[-1]))
                    except ValueError:
                        pass
    logger.info("Parsed %d data points from output", len(x))
    return x, y

# ══════════════════════════════════════════════════════════════════════════════
# WAVEFORM ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_waveform(x, y):
    if not x or not y:
        return {}
    peak   = max(y)
    trough = min(y)
    mean   = sum(y) / len(y)
    ripple = peak - trough
    lo, hi = 0.1 * peak, 0.9 * peak
    t_lo = next((x[i] for i in range(len(y)) if y[i] >= lo), None)
    t_hi = next((x[i] for i in range(len(y)) if y[i] >= hi), None)
    rise_time = (t_hi - t_lo) if (t_lo and t_hi) else None
    crossings = [x[i] for i in range(1, len(y)) if y[i-1] <= mean < y[i]]
    if len(crossings) >= 2:
        period = 2 * (crossings[-1] - crossings[0]) / (len(crossings) - 1)
        freq   = 1 / period if period > 0 else 0
    else:
        freq = 0
    return {
        "peak_V":       round(peak, 4),
        "trough_V":     round(trough, 4),
        "ripple_V":     round(ripple, 4),
        "mean_V":       round(mean, 4),
        "rise_time_s":  rise_time,
        "frequency_Hz": round(freq, 2),
        "data_points":  len(x)
    }

# ══════════════════════════════════════════════════════════════════════════════
# AI ANALYSIS  —  fault diagnosis + explanation
# ══════════════════════════════════════════════════════════════════════════════

def ai_fault_diagnosis(summary, prompt, voltage, resistor, capacitor, inductor):
    system = (
        "You are an electronics engineering expert giving advice to a student. "
        "Be specific and actionable. Mention actual component values."
    )
    user = (
        f"Circuit: {prompt}\n"
        f"Component values: V={voltage}, R={resistor}, C={capacitor}, L={inductor}\n"
        f"Simulation results: {json.dumps(summary)}\n\n"
        "Give a 3-4 sentence diagnosis:\n"
        "1. What the waveform shows (use actual numbers)\n"
        "2. Whether the circuit is behaving correctly\n"
        "3. If anything is wrong, suggest a specific fix with new component values\n"
        "4. What the RC/RL time constant is and how it relates to the results"
    )
    return ai_call(user, system_text=system)

def ai_explain_circuit(netlist, prompt):
    system = "You are a clear electronics teacher. Explain simply for a student."
    user   = (
        f"Circuit: {prompt}\n\n"
        f"NGSpice netlist:\n{netlist}\n\n"
        "Explain: what each component does, how the circuit works, "
        "and what the output waveform shape we expect."
    )
    return ai_call(user, system_text=system)

def ai_recommend_circuits(prompt):
    system = "You are an electronics mentor. Be specific and practical."
    user   = (
        f"A student just simulated: '{prompt}'\n"
        "Suggest 3 related circuits they could build next to deepen understanding. "
        "For each: name, one-line description, and what new concept it teaches."
    )
    return ai_call(user, system_text=system)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — SESSION MEMORY  (AI yaad rakhta hai pichle simulations)
# ══════════════════════════════════════════════════════════════════════════════

class SimulationSession:
    """Tracks all simulations in one run — AI reflects on the full session."""

    def __init__(self):
        self.history    = []
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def add(self, circuit, netlist, summary, ai_diagnosis):
        self.history.append({
            "circuit":   circuit,
            "summary":   summary,
            "diagnosis": ai_diagnosis,
            "timestamp": datetime.datetime.now().isoformat()
        })
        logger.info("Session: added '%s' (total=%d)", circuit, len(self.history))

    def ai_session_summary(self):
        """AI reflects on the entire session."""
        if not self.history:
            return "No simulations in this session."
        return ai_call(
            f"A student simulated these circuits in one session:\n"
            f"{json.dumps(self.history, indent=2)}\n\n"
            "Write a 5-6 sentence learning summary:\n"
            "1. What concepts were explored\n"
            "2. What patterns emerged across circuits\n"
            "3. What should they study next",
            system_text="Electronics teacher. Be encouraging and specific."
        )

    def save(self):
        os.makedirs("results", exist_ok=True)
        path = os.path.join("results", f"session_{self.session_id}.json")
        with open(path, "w") as f:
            json.dump({
                "session_id": self.session_id,
                "total":      len(self.history),
                "history":    self.history
            }, f, indent=2)
        logger.info("Session saved → %s", path)
        return path

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — MULTI-CIRCUIT COMPARISON  (Mode 3)
# ══════════════════════════════════════════════════════════════════════════════

def compare_circuits(circuit_list, voltage, resistor, capacitor, inductor, session=None):
    """
    Simulate multiple circuits and compare waveforms side by side.
    Usage: compare_circuits(["RC low pass", "RC high pass", "RL series"])
    """
    logger.info("Comparing %d circuits: %s", len(circuit_list), circuit_list)
    results = {}

    for circuit in circuit_list:
        logger.info("Simulating: %s", circuit)
        x, y, netlist = simulate(circuit, voltage, resistor, capacitor, inductor)
        if x and y:
            analysis = analyze_waveform(x, y)
            results[circuit] = {
                "x":        x,
                "y":        y,
                "analysis": analysis,
                "netlist":  netlist
            }
            logger.info("  %s → peak=%.4fV ripple=%.4fV",
                        circuit, analysis.get("peak_V",0), analysis.get("ripple_V",0))
        else:
            logger.warning("Failed to simulate: %s", circuit)

    if not results:
        logger.error("No circuits simulated successfully")
        return None

    # ── Plot all on same figure with params panel ──
    colors = ["#1D9E75", "#E74C3C", "#3498DB", "#F39C12", "#9B59B6"]
    n      = len(results)
    fig    = plt.figure(figsize=(14, 4 * n))

    for idx, (name, data) in enumerate(results.items()):
        # Waveform axes (left 70%)
        ax = fig.add_axes([0.06, (n - idx - 1) / n + 0.04/n,
                           0.60, 0.86/n])
        color = colors[idx % len(colors)]
        ax.plot(data["x"], data["y"], color=color, linewidth=1.4)
        ax.set_title(name, fontsize=10, fontweight="bold", pad=6)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Voltage (V)", fontsize=8)
        ax.grid(True, alpha=0.3)

        # Peak annotation
        peak_val = data["analysis"]["peak_V"]
        if data["y"]:
            peak_idx = data["y"].index(max(data["y"]))
            ax.annotate(
                f'Peak: {peak_val}V',
                xy=(data["x"][peak_idx], data["y"][peak_idx]),
                xytext=(10, -18), textcoords="offset points",
                fontsize=8, color=color,
                arrowprops=dict(arrowstyle="->", color=color)
            )

        # Params panel (right 28%)
        ax_p = fig.add_axes([0.70, (n - idx - 1) / n + 0.04/n,
                             0.28, 0.86/n])
        ax_p.axis("off")
        a = data["analysis"]
        info = (
            f"{name}\n"
            f"{'─'*22}\n"
            f"Peak V   : {a.get('peak_V','N/A')} V\n"
            f"Trough V : {a.get('trough_V','N/A')} V\n"
            f"Ripple   : {a.get('ripple_V','N/A')} V\n"
            f"Mean V   : {a.get('mean_V','N/A')} V\n"
            f"Freq     : {a.get('frequency_Hz','N/A')} Hz\n"
            f"Points   : {a.get('data_points','N/A')}"
        )
        ax_p.text(
            0.05, 0.95, info,
            transform=ax_p.transAxes,
            fontsize=7.5, verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5",
                      edgecolor="#cccccc", alpha=0.9)
        )

    # Filename from circuit names
    names_slug = "_vs_".join(_safe_filename(c) for c in circuit_list[:3])
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("graphs", f"compare_{names_slug}_{ts}.png")
    os.makedirs("graphs", exist_ok=True)
    fig.suptitle("Circuit Comparison", fontsize=13, fontweight="bold", y=1.01)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Comparison graph saved → %s", path)
    print(f"\n[PLOT] Comparison saved → {path}")

    # ── AI comparison analysis ──
    summaries = {name: data["analysis"] for name, data in results.items()}
    print("\n" + "="*52)
    print("[AI] COMPARISON ANALYSIS")
    print("="*52)
    ai_summary = ai_call(
        f"Compare these circuit simulation results:\n{json.dumps(summaries, indent=2)}\n\n"
        "In 4-5 sentences explain:\n"
        "1. Key differences in their waveform behavior (use actual numbers)\n"
        "2. Which performs better for filtering and why\n"
        "3. Which has lower ripple and what that means",
        system_text="Electronics expert. Be specific with numbers."
    )
    print(ai_summary)

    # ── Add to session if provided ──
    if session:
        for name, data in results.items():
            session.add(name, data["netlist"], data["analysis"], "Part of comparison run")

    # ── Save comparison JSON ──
    saved = save_results(
        f"comparison_{len(circuit_list)}_circuits",
        "multiple — see individual results",
        list(range(len(circuit_list))),
        [r["analysis"].get("peak_V", 0) for r in results.values()],
        summaries
    )
    print(f"[SAVED] → {saved}")
    return results

# ══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS  (auto-saves every run to results/ folder)
# ══════════════════════════════════════════════════════════════════════════════

def save_results(prompt, netlist, x, y, summary):
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^\w]", "_", prompt.lower())[:40]
    filepath  = os.path.join(results_dir, f"{safe_name}_{timestamp}.json")
    data = {
        "timestamp": timestamp,
        "circuit":   prompt,
        "netlist":   netlist,
        "waveform":  {"time": x[:50], "voltage": y[:50]},
        "analysis":  summary
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Results saved → %s", filepath)
    return filepath

# ══════════════════════════════════════════════════════════════════════════════
# AI OPTIMIZATION LOOP  (prompt-aware + tracks best result)
# ══════════════════════════════════════════════════════════════════════════════

def ai_optimize_circuit(goal, convo, prompt, iterations=3):
    """
    Prompt-aware optimizer — uses actual circuit name to generate netlists.
    Tracks best result across all iterations.
    """
    logger.info("Optimization start — goal='%s' circuit='%s'", goal, prompt)
    best_summary, best_convo = None, convo.copy()
    history = []

    for i in range(iterations):
        print(f"\n[OPT] ── Iteration {i+1}/{iterations} ──")
        print(f"[OPT] V={convo['input_voltage']} R={convo['load_resistance']} "
              f"C={convo['capacitor_value']} L={convo['inductor_value']}")

        intent  = ai_classify_intent(prompt)
        netlist = ai_generate_netlist(
            prompt,
            convo["input_voltage"], convo["load_resistance"],
            convo["capacitor_value"], convo["inductor_value"],
            intent
        )
        if not netlist:
            netlist = build_netlist_from_conversation(convo)

        netlist = ai_tune_sim_params(netlist, intent)
        valid, reason = validate_netlist(netlist)
        if not valid:
            logger.warning("Invalid netlist at iteration %d: %s", i+1, reason)
            break

        save_netlist(netlist)
        result = run_simulation()
        if not result or result.returncode != 0:
            logger.error("Simulation failed at iteration %d", i+1)
            break

        x, y = parse_output()
        if not x or not y:
            logger.warning("No output at iteration %d", i+1)
            break

        summary = analyze_waveform(x, y)
        history.append({"iteration": i+1, "values": dict(convo), "summary": summary})
        print(f"[OPT] peak={summary.get('peak_V')}V  ripple={summary.get('ripple_V')}V")
        plot_graph(x, y, f"Iteration {i+1} — {goal}")

        # Track best result (lowest ripple = best for most filter goals)
        if (best_summary is None or
                summary.get("ripple_V", 999) < best_summary.get("ripple_V", 999)):
            best_summary = summary
            best_convo   = convo.copy()
            logger.info("New best result at iteration %d", i+1)

        if i < iterations - 1:
            system = (
                "You are a circuit optimization expert. "
                "Respond ONLY with a JSON object — no explanation, no markdown."
            )
            user = (
                f"Circuit: {prompt}\nGoal: {goal}\n"
                f"Waveform: {json.dumps(summary)}\n"
                f"Values: {json.dumps(convo)}\n\n"
                "Return JSON with improved values:\n"
                "  input_voltage, load_resistance, capacitor_value, inductor_value"
            )
            suggestions = ai_call(user, system_text=system, expect_json=True)
            if isinstance(suggestions, dict) and "load_resistance" in suggestions:
                for key in ["input_voltage","load_resistance","capacitor_value","inductor_value"]:
                    if key in suggestions and suggestions[key]:
                        convo[key] = str(suggestions[key])
                logger.info("AI updated values: %s", convo)
            else:
                logger.warning("No valid suggestions from AI at iteration %d", i+1)

    # Before / After comparison
    if len(history) >= 2:
        print(f"\n{'='*52}")
        print("  OPTIMIZATION — Before vs After")
        print(f"{'='*52}")
        for m in ["peak_V","ripple_V","mean_V"]:
            b = history[0]["summary"].get(m, "N/A")
            a = history[-1]["summary"].get(m, "N/A")
            if isinstance(b, float) and isinstance(a, float):
                arrow = "▲" if a > b else "▼" if a < b else "─"
                print(f"  {m:12}: {b:.4f}V  →  {a:.4f}V  {arrow}")
        print(f"  Start : {history[0]['values']}")
        print(f"  Final : {history[-1]['values']}")
        print(f"{'='*52}")

    return best_summary, best_convo

# ══════════════════════════════════════════════════════════════════════════════
# GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def _safe_filename(name):
    """Convert circuit name to safe filename — spaces to underscores, remove special chars."""
    name = name.lower().strip()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"\s+", "_", name)
    return name[:50]   # max 50 chars

def plot_graph(x, y, title="AI Circuit Simulation",
               voltage=None, resistor=None, capacitor=None,
               inductor=None, summary=None):
    """
    Save graph with:
    - Circuit name as filename (not random timestamp)
    - Parameters panel on the right side
    - Waveform stats annotated on graph
    """
    folder = "graphs"
    os.makedirs(folder, exist_ok=True)

    # ── Filename from circuit name ──
    safe   = _safe_filename(title)
    ts     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path   = os.path.join(folder, f"{safe}_{ts}.png")

    # ── Layout: waveform left (75%) + params right (25%) ──
    fig = plt.figure(figsize=(13, 4))
    ax_wave  = fig.add_axes([0.06, 0.15, 0.62, 0.72])   # waveform area
    ax_panel = fig.add_axes([0.72, 0.05, 0.26, 0.90])   # params panel
    ax_panel.axis("off")

    # ── Waveform ──
    ax_wave.plot(x, y, linewidth=1.3, color="#1D9E75")
    ax_wave.grid(True, alpha=0.35)
    ax_wave.set_xlabel("Time (s)", fontsize=9)
    ax_wave.set_ylabel("Voltage (V)", fontsize=9)
    ax_wave.set_title(title, fontsize=10, fontweight="bold", pad=6)

    # Annotate peak on waveform
    if y:
        peak_val = max(y)
        peak_idx = y.index(peak_val)
        ax_wave.annotate(
            f"Peak: {peak_val:.3f}V",
            xy=(x[peak_idx], peak_val),
            xytext=(0.65, 0.92), xycoords=("axes fraction", "axes fraction"),
            fontsize=8, color="#1D9E75",
            arrowprops=dict(arrowstyle="->", color="#1D9E75",
                           connectionstyle="arc3,rad=0.2")
        )

    # ── Parameters panel ──
    panel_lines = ["PARAMETERS", "─" * 16]
    if voltage:   panel_lines.append(f"Voltage  : {voltage} V")
    if resistor:  panel_lines.append(f"Resistor : {resistor} Ω")
    if capacitor: panel_lines.append(f"Capacitor: {capacitor} F")
    if inductor:  panel_lines.append(f"Inductor : {inductor} H")

    if summary:
        panel_lines += ["", "ANALYSIS", "─" * 16]
        panel_lines.append(f"Peak V   : {summary.get('peak_V', 'N/A')} V")
        panel_lines.append(f"Trough V : {summary.get('trough_V', 'N/A')} V")
        panel_lines.append(f"Ripple   : {summary.get('ripple_V', 'N/A')} V")
        panel_lines.append(f"Mean V   : {summary.get('mean_V', 'N/A')} V")
        freq = summary.get("frequency_Hz", 0)
        if freq:
            panel_lines.append(f"Freq     : {freq} Hz")
        rt = summary.get("rise_time_s")
        if rt:
            panel_lines.append(f"Rise time: {rt:.4f} s")
        panel_lines.append(f"Points   : {summary.get('data_points', 'N/A')}")

    panel_text = "\n".join(panel_lines)
    ax_panel.text(
        0.05, 0.95, panel_text,
        transform=ax_panel.transAxes,
        fontsize=8, verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5",
                  edgecolor="#cccccc", alpha=0.9)
    )

    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Graph saved → %s", path)
    print(f"[PLOT] Graph saved: {path}")
    return path

# ══════════════════════════════════════════════════════════════════════════════
# CONVERSATIONAL DESIGNER  (Mode 2)
# ══════════════════════════════════════════════════════════════════════════════

def mode2_ai_suggest_values(goal):
    """
    Sub-mode A: User describes a goal in plain English.
    AI suggests initial component values based on the goal.
    Returns a convo dict.
    """
    system = (
        "You are an electronics expert. "
        "Respond ONLY with a JSON object — no explanation, no markdown."
    )
    user = (
        f"Design goal: {goal}\n\n"
        "Suggest starting component values for an RLC circuit to achieve this goal.\n"
        "Return JSON with keys:\n"
        "  input_voltage   : e.g. '10'\n"
        "  load_resistance : e.g. '1k'\n"
        "  capacitor_value : e.g. '10u'\n"
        "  inductor_value  : e.g. '10m'\n"
        "  circuit_type    : one of 'rc', 'rl', 'rlc', 'diode'\n"
        "  reason          : one sentence why these values suit the goal"
    )
    result = ai_call(user, system_text=system, expect_json=True)
    if not result or "input_voltage" not in result:
        print("[AI] Could not suggest values — using defaults.")
        return {
            "input_voltage":   "10",
            "load_resistance": "1k",
            "capacitor_value": "10u",
            "inductor_value":  "10m",
            "circuit_type":    "rlc"
        }
    print(f"\n[AI] Suggested values for goal '{goal}':")
    print(f"     V={result['input_voltage']}, R={result['load_resistance']}, "
          f"C={result['capacitor_value']}, L={result['inductor_value']}")
    if "reason" in result:
        print(f"     Reason: {result['reason']}")
    return result

def mode2_manual_values():
    """
    Sub-mode B: User enters values manually.
    """
    print("\n--- Manual Component Entry ---")
    def ask(q, default):
        val = input(f"  {q} [{default}]: ").strip()
        return val if val else default
    return {
        "input_voltage":   ask("Input voltage (V)", "10"),
        "load_resistance": ask("Load resistance (ohms)", "1k"),
        "capacitor_value": ask("Capacitor value (F)", "10u"),
        "inductor_value":  ask("Inductor value (H)", "10m"),
        "circuit_type":    "rlc"
    }

def build_netlist_from_conversation(convo):
    """Build netlist based on circuit_type in convo."""
    ctype = convo.get("circuit_type", "rlc").lower()
    v  = convo["input_voltage"]
    r  = convo["load_resistance"]
    c  = convo["capacitor_value"]
    l  = convo["inductor_value"]

    if ctype == "rc":
        return (
            f"* RC Circuit (Conversational)\n"
            f"V1 1 0 PULSE(0 {v} 0 1u 1u 5m 10m)\n"
            f"R1 1 2 {r}\n"
            f"C1 2 0 {c}\n"
            f".tran 10u 50m\n"
            f".print tran v(2)\n"
            f".end\n"
        )
    elif ctype == "rl":
        return (
            f"* RL Circuit (Conversational)\n"
            f"V1 1 0 SIN(0 {v} 50)\n"
            f"R1 1 2 {r}\n"
            f"L1 2 0 {l}\n"
            f".tran 1u 10m\n"
            f".print tran v(2)\n"
            f".end\n"
        )
    elif ctype == "diode":
        return (
            f"* Diode Rectifier (Conversational)\n"
            f"V1 1 0 SIN(0 {v} 50)\n"
            f"D1 1 2 D\n"
            f"R1 2 0 {r}\n"
            f"C1 2 0 {c}\n"
            f".model D D\n"
            f".tran 1u 20m\n"
            f".print tran v(2)\n"
            f".end\n"
        )
    else:  # default: rlc
        return (
            f"* RLC Circuit (Conversational)\n"
            f"V1 1 0 SIN(0 {v} 50)\n"
            f"R1 1 2 {r}\n"
            f"C1 2 0 {c}\n"
            f"L1 2 0 {l}\n"
            f".tran 1u 20m\n"
            f".print tran v(2)\n"
            f".end\n"
        )

def ai_optimize_circuit(goal, convo, iterations=3):
    """
    Iterative AI optimization loop.
    Each iteration: simulate → analyze → AI suggests better values → repeat.
    Shows before/after comparison after all iterations.
    """
    print(f"\n{'='*50}")
    print(f"  AI Optimization Loop  |  Goal: {goal}")
    print(f"{'='*50}")

    history = []   # store summary of each iteration

    for i in range(iterations):
        print(f"\n[OPT] ── Iteration {i+1}/{iterations} ──")
        print(f"[OPT] Values: V={convo['input_voltage']}, "
              f"R={convo['load_resistance']}, "
              f"C={convo['capacitor_value']}, "
              f"L={convo['inductor_value']}")

        netlist = build_netlist_from_conversation(convo)
        valid, reason = validate_netlist(netlist)
        if not valid:
            print(f"[OPT] Invalid netlist: {reason}")
            break

        save_netlist(netlist)
        result = run_simulation()
        if not result or result.returncode != 0:
            print("[OPT] Simulation failed.")
            break

        x, y = parse_output()
        if not x or not y:
            print("[OPT] No output data.")
            break

        summary = analyze_waveform(x, y)
        history.append({
            "iteration": i + 1,
            "values":    dict(convo),
            "summary":   summary
        })
        print(f"[OPT] Result: peak={summary.get('peak_V')}V, "
              f"ripple={summary.get('ripple_V')}V, "
              f"mean={summary.get('mean_V')}V")

        # Save graph for each iteration
        plot_graph(x, y, f"Iteration {i+1} — {goal}")

        if i < iterations - 1:
            # Ask AI for better values
            system = (
                "You are a circuit optimization expert. "
                "Respond ONLY with a JSON object — no explanation, no markdown."
            )
            user = (
                f"Design goal: {goal}\n"
                f"Current waveform: {json.dumps(summary)}\n"
                f"Current values: {json.dumps(convo)}\n\n"
                "Suggest improved component values to better achieve the goal.\n"
                "Return JSON with keys:\n"
                "  input_voltage, load_resistance, capacitor_value, inductor_value\n"
                "  improvement: one sentence explaining what you changed and why\n"
                "Only return the JSON."
            )
            suggestions = ai_call(user, system_text=system, expect_json=True)
            if isinstance(suggestions, dict) and "load_resistance" in suggestions:
                improvement = suggestions.pop("improvement", "")
                for key in ["input_voltage", "load_resistance",
                            "capacitor_value", "inductor_value"]:
                    if key in suggestions and suggestions[key]:
                        convo[key] = str(suggestions[key])
                print(f"[AI] Updated values: R={convo['load_resistance']}, "
                      f"C={convo['capacitor_value']}, L={convo['inductor_value']}")
                if improvement:
                    print(f"[AI] Reason: {improvement}")
            else:
                print("[OPT] AI returned no valid suggestions — keeping current values.")

    # ── Before / After comparison ──
    if len(history) >= 2:
        print(f"\n{'='*50}")
        print("  OPTIMIZATION RESULT — Before vs After")
        print(f"{'='*50}")
        first = history[0]
        last  = history[-1]
        metrics = ["peak_V", "ripple_V", "mean_V"]
        for m in metrics:
            before = first["summary"].get(m, "N/A")
            after  = last["summary"].get(m, "N/A")
            if isinstance(before, float) and isinstance(after, float):
                change = after - before
                arrow  = "▲" if change > 0 else "▼" if change < 0 else "─"
                print(f"  {m:15s}: {before:.4f}V  →  {after:.4f}V  {arrow}")
            else:
                print(f"  {m:15s}: {before}  →  {after}")
        print(f"\n  Start values : {first['values']}")
        print(f"  Final values : {last['values']}")
        print(f"{'='*50}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN SIMULATION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def simulate(prompt, voltage, resistor, capacitor, inductor):
    """
    Full pipeline:
      1. Template check FIRST (fast, reliable, 19 circuits)
      2. If no template → AI Intent Classification
      3. AI generates netlist
      4. Validate → fallback if invalid
      5. Run NGSpice
    """
    netlist = None

    # ── Step 1: Template check FIRST (fast + 100% reliable) ──
    netlist = template_fallback(prompt, voltage, resistor, capacitor, inductor)

    if netlist:
        logger.info("Template matched — skipping AI netlist generation")
    else:
        # ── Step 2: AI Intent Classification (only if no template) ──
        intent = ai_classify_intent(prompt)

        # ── Step 3: AI Netlist Generation ──
        if OLLAMA_AVAILABLE:
            raw = ai_generate_netlist(prompt, voltage, resistor, capacitor, inductor, intent)
            if raw:
                raw = ai_tune_sim_params(raw, intent)
                valid, reason = validate_netlist(raw)
                if valid:
                    netlist = raw
                    logger.info("AI netlist validated successfully")
                else:
                    logger.warning("AI netlist failed validation (%s)", reason)
        else:
            logger.warning("Ollama not available — no AI netlist generation")

    if not netlist:
        print("\n[SIM] Could not generate a netlist for this circuit.")
        print("      Supported circuits:")
        print("      RC/RL filters, RLC, Diode, Bandpass, Zener, CE Amplifier,")
        print("      Full wave rectifier, Wheatstone bridge, Schmitt trigger...")
        return None, None, None

    print("\n--- NETLIST ---")
    print(netlist)

    save_netlist(netlist)
    result = run_simulation()
    if not result or result.returncode != 0:
        print("\n[SIM] Simulation failed. Possible reasons:")
        print("  - Complex circuit needs manual node checking")
        print("  - Missing .model statement for semiconductors")
        print("  - NGSpice not installed or not in PATH")
        return None, None, netlist

    x, y = parse_output()
    return x, y, netlist

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args    = parse_args()
    session = SimulationSession()   # Step 7: session memory

    print("\n" + "="*52)
    print("   AI Circuit Simulator  —  Final Version")
    print("   19 circuit types  |  Ollama + NGSpice")
    print("="*52 + "\n")

    # ── CLI direct mode ────────────────────────────────────────────────────────
    if args.circuit:
        logger.info("CLI mode — '%s'", args.circuit)
        voltage   = args.voltage   or DEFAULT_VALUES["voltage"]
        resistor  = args.resistor  or DEFAULT_VALUES["resistor"]
        capacitor = args.capacitor or DEFAULT_VALUES["capacitor"]
        inductor  = args.inductor  or DEFAULT_VALUES["inductor"]
        try:
            sanitize_input(args.circuit)
        except ValueError as e:
            logger.error("Unsafe input: %s", e)
            return
        x, y, netlist = simulate(args.circuit, voltage, resistor, capacitor, inductor)
        if x and y:
            summary   = analyze_waveform(x, y)
            diagnosis = ai_fault_diagnosis(summary, args.circuit,
                                           voltage, resistor, capacitor, inductor)
            saved     = save_results(args.circuit, netlist, x, y, summary)
            session.add(args.circuit, netlist, summary, diagnosis)
            print(f"\n[SAVED] → {saved}")
            if not args.no_plot:
                plot_graph(x, y, args.circuit.title(),
                           voltage=voltage, resistor=resistor,
                           capacitor=capacitor, inductor=inductor,
                           summary=summary)
            print("\n[RESULT]", json.dumps(summary, indent=2))
        elif netlist:
            logger.warning("Netlist generated but no output")
        else:
            logger.error("Failed to generate netlist")

        if session.history:
            print("\n" + "="*52)
            print("[SESSION SUMMARY]")
            print("="*52)
            print(session.ai_session_summary())
            session.save()
        return

    # ── Interactive mode ───────────────────────────────────────────────────────
    mode = args.mode or input(
        "Mode:\n"
        "  1 — Describe any circuit\n"
        "  2 — Conversational designer\n"
        "  3 — Compare multiple circuits\n"
        "Choice [1]: "
    ).strip()

    # ── Mode 3: Compare circuits ───────────────────────────────────────────────
    if mode == "3":
        print("\n" + "="*52)
        print("  Mode 3 — Multi-Circuit Comparison")
        print("="*52)
        print("\nEnter circuit names separated by commas:")
        print("Example: RC low pass filter, RC high pass filter, RL series circuit")
        raw      = input("> ").strip()
        circuits = [c.strip() for c in raw.split(",") if c.strip()]

        if len(circuits) < 2:
            print("[ERROR] Please enter at least 2 circuits to compare.")
            return

        voltage, resistor, capacitor, inductor = get_user_parameters()
        compare_circuits(circuits, voltage, resistor, capacitor, inductor, session)

    # ── Mode 2: Conversational Designer ───────────────────────────────────────
    elif mode == "2":
        print("\n" + "="*52)
        print("  Mode 2 — Conversational Circuit Designer")
        print("="*52)
        print("\n  2a — Describe your goal, AI suggests values")
        print("  2b — Enter component values manually")
        sub = input("Choice [2a]: ").strip().lower()

        if sub in ("2b", "b", "manual"):
            convo  = mode2_manual_values()
            goal   = input("\nOptimization goal (e.g. 'ripple < 0.5V') or ENTER to skip: ").strip()
            prompt = input("Circuit name (for AI): ").strip() or "RLC circuit"
        else:
            goal = input(
                "\nDescribe your design goal:\n"
                "(e.g. 'low pass filter', 'reduce ripple', 'RC charging circuit')\n> "
            ).strip()
            if not goal:
                goal = "general RLC circuit"
            convo  = mode2_ai_suggest_values(goal)
            prompt = goal
            print("\nPress ENTER to accept AI values, or type new value.")
            def confirm(label, key):
                val = input(f"  {label} [{convo[key]}]: ").strip()
                if val:
                    convo[key] = val
            confirm("Input voltage (V)",      "input_voltage")
            confirm("Load resistance (ohms)", "load_resistance")
            confirm("Capacitor value (F)",    "capacitor_value")
            confirm("Inductor value (H)",     "inductor_value")

        if goal:
            _, best_convo = ai_optimize_circuit(goal, convo, prompt)
            convo = best_convo

        print(f"\n[MODE2] Final simulation with: {convo}")
        netlist = build_netlist_from_conversation(convo)
        valid, reason = validate_netlist(netlist)
        if not valid:
            logger.error("Invalid netlist: %s", reason)
            return

        save_netlist(netlist)
        run_simulation()
        x, y = parse_output()

        if x and y:
            summary   = analyze_waveform(x, y)
            diagnosis = ai_fault_diagnosis(summary, prompt,
                                           convo["input_voltage"], convo["load_resistance"],
                                           convo["capacitor_value"], convo["inductor_value"])
            saved = save_results(prompt, netlist, x, y, summary)
            session.add(prompt, netlist, summary, diagnosis)
            print(f"\n[RESULT] {len(x)} data points | Saved → {saved}")
            plot_graph(x, y, f"Final — {goal}",
                       voltage=convo["input_voltage"],
                       resistor=convo["load_resistance"],
                       capacitor=convo["capacitor_value"],
                       inductor=convo["inductor_value"],
                       summary=summary)

            print("\n" + "="*52)
            print("[AI] FINAL FAULT DIAGNOSIS")
            print("="*52)
            print(diagnosis)
            print("\n" + "="*52)
            print("[AI] WHAT TO TRY NEXT")
            print("="*52)
            print(ai_recommend_circuits(goal))
        else:
            logger.warning("No simulation output — check NGSpice installation.")

    # ── Mode 1: Describe any circuit ──────────────────────────────────────────
    else:
        try:
            raw    = input("\nDescribe the circuit you want to simulate:\n> ")
            prompt = sanitize_input(raw)
        except ValueError as e:
            logger.error("Input error: %s", e)
            return

        voltage, resistor, capacitor, inductor = get_user_parameters()
        x, y, netlist = simulate(prompt, voltage, resistor, capacitor, inductor)

        if x and y:
            summary   = analyze_waveform(x, y)
            diagnosis = ai_fault_diagnosis(summary, prompt,
                                           voltage, resistor, capacitor, inductor)
            saved = save_results(prompt, netlist, x, y, summary)
            session.add(prompt, netlist, summary, diagnosis)

            print(f"\n[RESULT] {len(x)} data points collected.")
            print(f"[SAVED]  → {saved}")
            plot_graph(x, y, prompt.title(),
                       voltage=voltage, resistor=resistor,
                       capacitor=capacitor, inductor=inductor,
                       summary=summary)

            print("\n" + "="*52)
            print("[AI] FAULT DIAGNOSIS")
            print("="*52)
            print(diagnosis)

            print("\n" + "="*52)
            print("[AI] CIRCUIT EXPLANATION")
            print("="*52)
            print(ai_explain_circuit(netlist, prompt))

            print("\n" + "="*52)
            print("[AI] WHAT TO TRY NEXT")
            print("="*52)
            print(ai_recommend_circuits(prompt))

        elif netlist:
            logger.warning("Netlist generated but simulation produced no output.")
        else:
            logger.error("Failed to generate netlist.")

    # ── Session Summary (end of every run) ────────────────────────────────────
    if session.history:
        print("\n" + "="*52)
        print("[AI] SESSION LEARNING SUMMARY")
        print("="*52)
        print(session.ai_session_summary())
        path = session.save()
        print(f"[SAVED] Session → {path}")


if __name__ == "__main__":
    main()
