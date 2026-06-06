
# ⚡ AI Circuit Simulator

> Simulate any electronic circuit using natural language — powered by Groq LLM + NGSpice

![Python](https://img.shields.io/badge/Python-3.12-blue)
![NGSpice](https://img.shields.io/badge/NGSpice-Simulation-green)
![Groq](https://img.shields.io/badge/AI-Groq%20LLaMA%203.3-purple)
![Streamlit](https://img.shields.io/badge/GUI-Streamlit-red)

---

## Problem Statement

Traditional circuit simulators require users to manually write NGSpice netlists — a complex, error-prone process. This project solves that by allowing users to **describe a circuit in plain English**, and the AI automatically generates, validates, and simulates it.

---

## Key Features

| Feature | Description |
|---|---|
| **19 Circuit Templates** | RC, RL, RLC, Diode, Bandpass, Zener, CE Amplifier, Full Wave Rectifier and more |
| **AI Intent Classifier** | Automatically decides TRAN / AC / DC analysis type |
| **AI Netlist Generator** | Groq LLaMA 3.3-70B writes NGSpice netlist from description |
| **AI Fault Diagnosis** | Analyzes waveform and gives specific fix with component values |
| **AI Optimizer (Mode 2)** | Iteratively improves R/C/L values to meet design goal |
| **Multi-Circuit Compare (Mode 3)** | Simulate 2-4 circuits side by side with AI comparison |
| **Session Memory** | AI reflects on all simulations and gives learning summary |
| **Named Graphs + Params Panel** | Circuit name as filename, parameters shown on graph |
| **Auto-Save Results** | Every run saved as JSON in `results/` folder |
| **Logging System** | Full logs in `logs/` folder — every AI call tracked |
| **CLI Arguments** | Run directly: `python ai_generator.py --circuit "RC filter"` |
| **Web GUI** | Professional Streamlit dashboard with progress tracking |

---

## AI Integration — Where AI is Used

```
User Input
    │
    ▼
[AI #1] Intent Classifier     → decides TRAN / AC / DC analysis
    │
    ▼
[AI #2] Netlist Generator     → writes NGSpice netlist (Groq primary)
    │
    ▼
[Physics] Sim Param Tuner     → calculates optimal time step from R×C
    │
    ▼
NGSpice Simulation            → runs actual circuit equations
    │
    ▼
[AI #3] Fault Diagnosis       → specific waveform analysis with fixes
[AI #4] Circuit Explanation   → plain English explanation
[AI #5] Recommendations       → suggests 3 related circuits
    │
    ▼
[AI #6] Session Memory        → learning summary across all simulations
[AI #7] Optimizer Loop        → iterative R/C/L improvement (Mode 2)
```

**AI Backend:** Groq API (llama-3.3-70b-versatile) → Ollama/Mistral fallback

---

## Project Structure

```
ai_circuit_project/
├── ai_generator.py     ← Main simulator (CLI)
├── gui.py              ← Streamlit web GUI
├── .env                ← API keys (not committed to git)
├── requirements.txt    ← Dependencies
├── graphs/             ← Auto-generated waveform plots
├── results/            ← Auto-saved JSON results
└── logs/               ← Daily simulation logs
```

---

## Setup & Installation

### 1. Install Dependencies
```bash
pip install ollama matplotlib groq python-dotenv streamlit
```

### 2. Install NGSpice
Download from: https://ngspice.sourceforge.net/

### 3. Configure API Keys
Create `.env` file in project folder:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get free key at: https://console.groq.com

### 4. Run CLI
```bash
python ai_generator.py
```

### 5. Run GUI
```bash
streamlit run gui.py
```

---

## Usage Examples

### Mode 1 — Simulate any circuit
```
> RC low pass filter
> common emitter amplifier using BJT
> zener diode voltage regulator
> full wave bridge rectifier with capacitor filter
```

### Mode 2 — AI Optimizer
```
Goal: reduce ripple below 0.5V
Circuit: RC low pass filter
→ AI runs 3 iterations, adjusts R/C/L, shows before/after comparison
```

### Mode 3 — Compare circuits
```
> RC low pass filter, RC high pass filter, RL series circuit
→ Side-by-side waveforms + AI comparison analysis
```

### CLI Mode
```bash
python ai_generator.py --circuit "bandpass filter" --voltage 5 --no-plot
```

---

## Supported Circuits

### Level 1 — Basic (Template-based, 100% reliable)
RC Low Pass · RC High Pass · RC Integrator · RC Differentiator · RL Series · RL Parallel · Half Wave Rectifier · Voltage Divider

### Level 2 — Intermediate (Template-based, 100% reliable)
Full Wave Bridge Rectifier · RLC Bandpass Filter · RLC Series Resonance · RLC Parallel · Diode Clamp · Diode Clipper · Zener Voltage Regulator · Common Emitter Amplifier · Wheatstone Bridge · RC Phase Shift Oscillator · Schmitt Trigger

### Beyond Templates — AI Generated
Any circuit description → Groq LLaMA 3.3-70B generates netlist → validated → simulated

---

## Tech Stack

| Component | Technology |
|---|---|
| Circuit Simulation | NGSpice (industry standard) |
| Primary AI | Groq API — LLaMA 3.3-70B |
| Fallback AI | Ollama — Mistral (local) |
| GUI Framework | Streamlit |
| Visualization | Matplotlib |
| Language | Python 3.12 |

---

## Requirements

```
ollama>=0.6.0
matplotlib>=3.7.0
groq>=1.0.0
python-dotenv>=1.0.0
streamlit>=1.55.0
```

---

## Limitations & Future Scope

- Complex IC-based circuits (555 timer, op-amp with feedback) have limited AI reliability
- Future: Add more templates, AC frequency response analysis, interactive waveform viewer
- Future: Fine-tune a specialized model on NGSpice netlist dataset

---

## Author

Harsh Belwal 
Ekta Kaur — Internship Project
