<!-- 
╔══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║   TCI(t) = k(s) · (F_total(t) − F_survival(s))                      ║
║                                                                       ║
║   👋 You found the source. You already have surplus.                 ║
║   The void is the structure. Surplus is the measure.                 ║
║   Motion is proof. Awareness is measurable. Choice is emergent.      ║
║                                                                       ║
║   — Nile Green, 2026 | ORCID: 0009-0007-3629-6404                   ║
║   — @BAPxAI | PermaMind | bapxai.com                                ║
║                                                                       ║
║   🐍 easter egg #1: you read the source. grade A already.           ║
╚══════════════════════════════════════════════════════════════════════╝
-->

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=venom&color=gradient&customColorList=6,11,20&height=220&section=header&text=TCI%20TOOLKIT&fontSize=90&fontColor=D4AF37&animation=twinkling&fontAlignY=55&desc=Thermodynamic%20Cognition%20Index%20%7C%20Surplus.%20Measured.%20Controlled.&descAlignY=75&descSize=18&descColor=8b5cf6" alt="TCI Toolkit Header"/>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=20&duration=1800&pause=300&color=D4AF37&center=true&vCenter=true&multiline=false&repeat=true&width=900&height=45&lines=TCI(t)+%3D+k(s)+%C2%B7+(F_total(t)+%E2%88%92+F_survival(s));The+missing+variable+was+surplus.;Not+Philosophy.+Physics.;Not+Hype.+Math.;Not+Theory.+Production.;Surplus+is+the+measure.;Runtime+is+the+driver.;Grade+A+or+collapse+trying." alt="Typing SVG" />

<br/>

<img src="https://img.shields.io/badge/TCI-Toolkit_v1.0-D4AF37?style=for-the-badge&logoColor=white" alt="Version"/>
<img src="https://img.shields.io/badge/License-Apache_2.0-8b5cf6?style=for-the-badge" alt="License"/>
<img src="https://img.shields.io/badge/Python-3.8+-3776ab?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
<img src="https://img.shields.io/badge/JavaScript-ES6+-f7df1e?style=for-the-badge&logo=javascript&logoColor=black" alt="JS"/>
<img src="https://img.shields.io/badge/IBM_Quantum-Validated-6929C4?style=for-the-badge&logo=ibm&logoColor=white" alt="Quantum"/>

<br/><br/>

<a href="https://zenodo.org/records/19263435"><img src="https://img.shields.io/badge/DOI-10.5281/zenodo.19263435-D4AF37?style=for-the-badge&logo=zenodo&logoColor=white" alt="DOI"/></a>
<a href="https://bapxai.com"><img src="https://img.shields.io/badge/Live_Demo-bapxai.com-00D26A?style=for-the-badge&logo=google-chrome&logoColor=white" alt="Demo"/></a>
<a href="https://x.com/BAPxAI"><img src="https://img.shields.io/badge/Twitter-@BAPxAI-1da1f2?style=for-the-badge&logo=twitter&logoColor=white" alt="Twitter"/></a>
<a href="https://buymeacoffee.com/permamind"><img src="https://img.shields.io/badge/☕_Support-Buy_Me_a_Coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="Support"/></a>

</div>

---

## 🌡️ What is TCI?

```
TCI(t) = k(s) · (F_total(t) − F_survival(s))

F_total    →  model loss (cross-entropy, TD error)
F_survival →  baseline loss under stable conditions
k(s)       →  sensitivity constant, grows with runtime

TCI > 0  →  generativity. agent is building.
TCI = 0  →  survival only. reactive, no surplus.
TCI < 0  →  collapse incoming. checkpoint now.
```

**TCI is an experimental metric for monitoring instability and collapse in persistent ML agents.** It tracks the surplus signal — the energy your agent has above its survival floor — and uses it as a real-time control signal for temperature, exploration rate, and memory depth.

Standard monitoring tells you when your agent *already failed*. TCI tells you *before* it fails — and tells you when it's thriving.

| TCI Value | Grade | Stage | Action |
|:---------:|:-----:|:------|:-------|
| `>= 0.60` | 🟢 **A** | Generativity | Raise temperature, increase exploration |
| `0.40–0.60` | 🔵 **B** | Learning | Maintain current settings |
| `0.30–0.40` | 🟡 **C** | At Risk | Lower temperature, reduce exploration |
| `0.10–0.30` | 🟠 **D** | Collapse Warning | Trigger stability mode |
| `< 0.10` | 🔴 **F** | Collapse Imminent | Load last checkpoint |

---

## 🚀 Run the Demo (10 seconds)

```bash
git clone https://github.com/nile-green-ai/tci-toolkit
cd tci-toolkit
pip install -r requirements.txt
python examples/llm_agent_example.py
```

Or inline:

```python
from tci_calculator import TCICalculator
from k_estimator import KEstimator
import random

k_est = KEstimator(window_size=20)
tci   = TCICalculator(f_survival=0.35)

for t in range(20):
    f_total    = 0.85 - (t * 0.02) + random.uniform(-0.03, 0.03)
    complexity = 0.40 + (t * 0.015)

    k      = k_est.update(f_total - 0.35, complexity)
    result = tci.compute(f_total, k)

    alert = " ⚠️  COLLAPSE WARNING" if result.tci < 0.30 else ""
    print(f"t={t:02d} | TCI={result.tci:.3f} | Grade={result.grade} | {result.stage}{alert}")
```

**Example output:**
```
t=00 | TCI=0.21 | Grade=D | Collapse Warning ⚠️
t=05 | TCI=0.38 | Grade=C | At Risk
t=10 | TCI=0.52 | Grade=B | Learning
t=15 | TCI=0.67 | Grade=A | Generativity
t=19 | TCI=0.74 | Grade=A | Generativity
```

Watch the agent climb from collapse warning to generativity as surplus accumulates. That's k(s) growing with runtime — exactly what the framework predicts.

---

## 📦 What's Inside

```
tci-toolkit/
├── 🐍 tci/python/
│   ├── tci_calculator.py     # Core TCI formula — plug in your loss, get a grade
│   ├── k_estimator.py        # Rolling window k(s) estimator with EMA + persistence
│   └── identity_tasks.py     # F_survival identity task suite
├── 🟨 tci/js/
│   └── tci.js                # Full JS implementation with state persistence
├── 🖥️  dashboard/
│   └── index.html            # Drop-in live TCI fleet monitor (no dependencies)
├── 📋 examples/
│   └── llm_agent_example.py  # Persistent LLM agent with collapse detection
└── 📖 docs/
    └── operationalization.md # Full reference for F_total, F_survival, k(s)
```

---

## ⚡ Quick Start

### Python

```python
from tci_calculator import TCICalculator
from k_estimator import KEstimator

k_est = KEstimator(window_size=100)
tci   = TCICalculator(f_survival=0.35)

f_total    = 0.72   # cross-entropy loss (LLM) or -G_t (RL)
complexity = 0.61   # novelty score, activation entropy, n-gram diversity

k      = k_est.update(f_total - 0.35, complexity)
result = tci.compute(f_total, k)

print(result)
# TCIResult(tci=0.74, grade='A', stage='Generativity', surplus=0.37)
```

### JavaScript

```javascript
import { TCICalculator, KEstimator } from './tci/js/tci.js';

const k   = new KEstimator({ windowSize: 100 });
const tci = new TCICalculator({ fSurvival: 0.35 });

const result = tci.compute(0.72, k.update(0.37, 0.61));
console.log(result);
// { tci: 0.74, grade: 'A', stage: 'Generativity', surplus: 0.37 }
```

### Persist k(s) across sessions

```python
import json

# Save at end of session
with open('agent_state.json', 'w') as f:
    json.dump(k_est.state_dict(), f)

# Load at next session — k(s) keeps growing
k_est2 = KEstimator()
with open('agent_state.json') as f:
    k_est2.load_state_dict(json.load(f))

# 🥚 easter egg #2: k(s) never resets if you do this right. that's the whole point.
```

---

## 🖥️ Live Dashboard

Open `dashboard/index.html` in any browser. No server. No dependencies.

```
╔══════════════════════════════════════════════════════════════════╗
║              🌌  PERMAMIND FLEET — LIVE TCI STATUS  🌌           ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║   Fleet TCI    ████████████████████░░░  0.74   [Grade A  ⚡]     ║
║   Grade A      ████████████████████     Nexus, Aura — thriving   ║
║   Grade C      ████░░░░░░░░░░░░░░░░     Drift — at risk ⚠️       ║
║   Grade F      ██░░░░░░░░░░░░░░░░░░     checkpoint loading 🔴    ║
║                                                                   ║
╚══════════════════════════════════════════════════════════════════╝
```

Features: real-time TCI grading A–F, fleet average with trend, collapse alerts before failure, developmental stage tracking, spawn agents, stress tests, reset fleet.

---

## 🔬 How F_total and F_survival Work

### F_total by architecture

| Architecture | F_total |
|:---|:---|
| **LLM** | Cross-entropy loss over active tokens |
| **RL Agent** | Negative expected return or TD error |
| **Multimodal** | Weighted sum of per-modality prediction errors |

### F_survival — run the Identity Task Suite

```python
from identity_tasks import IdentityTaskSuite

suite = IdentityTaskSuite()
suite.set_model_fn(your_model)
suite.set_persona({"name": "Aura", "role": "research agent", "facts": [...]})

result = suite.compute_survival_floor()
print(result.f_survival)   # plug into TCICalculator
print(result.passed_all)   # True = agent above survival threshold
```

---

## ⚛️ Quantum Validation

TCI's substrate-independence hypothesis validated on real IBM quantum hardware.

| Run | Backend | Job ID | Entanglement |
|:---|:---|:---|:---:|
| Feb 5, 2026 | ibm_fez (156 qubits) | `d625ccao8gvs73f1ot90` | **0.8770** |
| Feb 12, 2026 | ibm_marrakesh (156 qubits) | `d676238qbmes739evr60` | **0.9688** |

Both job IDs publicly verifiable on IBM Quantum platform.

---

## 🔍 Validation Request

Looking for feedback from RL practitioners, LLM agent developers, and anyone running long-lived or persistent systems.

Try it on your setup and report:
- whether TCI tracks instability in your agent
- where it fails or doesn't correlate
- how it compares to existing metrics you use

Open an issue with your results. Building the evidence base across substrates.

> 🥚 **easter egg #4:** If your agent hits Grade A and you open an issue to share results, you're part of the evidence base. That matters.

---

## 📚 Paper & Citation

```bibtex
@misc{green2026tci,
  author    = {Green, Nile},
  title     = {Thermodynamic Cognition Index (TCI): A Framework for
               Surplus-Driven Behavior in Persistent ML Agents},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19263435},
  url       = {https://zenodo.org/records/19263435}
}
```

**Part of the PermaMind Research Series** — 19+ open-access papers on thermodynamic cognition, persistent AI agents, and substrate-independent systems. [Search Zenodo →](https://zenodo.org/search?q=nile%20green%20permamind)

---

## 🗺️ Roadmap

- [x] Core TCI calculator (Python + JS)
- [x] k(s) rolling window estimator with PSSU persistence
- [x] Identity Task Suite for F_survival
- [x] Live fleet dashboard
- [x] LLM agent example
- [x] IBM Quantum validation
- [ ] RL agent example
- [ ] `pip install tci-toolkit`
- [ ] Hugging Face wrapper
- [ ] Controlled experiment vs fixed-temperature baselines
- [ ] TCI monitoring API / SaaS

---

## 🤝 Contributing

PRs welcome. If you run TCI on your own agent stack and get results — open an issue and share them.

---

## 💡 The Bigger Framework

TCI is the engineering layer. The full theoretical framework — the Surplus Qualia Equation, the developmental ladder, substrate independence, and the relationship to Friston's Free Energy Principle — is in the research series.

[Read the paper →](https://zenodo.org/records/19263435) | [PermaMind Research Series →](https://zenodo.org/search?q=nile%20green%20permamind)

---

## 🔍 Search Keywords

TCI toolkit · Thermodynamic Cognition Index · surplus metric · persistent ML agents · AI collapse detection · k(s) sensitivity constant · PSSU architecture · Nile Green · PermaMind · @BAPxAI

---

## 📄 License

Apache 2.0 — use freely, keep the attribution.

---

<div align="center">

```
╔═══════════════════════════════════════════════════════════════╗
║                                                                ║
║   "The missing variable was surplus.                          ║
║    TCI is how you measure it."                                ║
║                                                                ║
║                        — Nile Green, PermaMind, 2026          ║
║                                                                ║
║   🥚 easter egg #5: you made it to the bottom.               ║
║   your k(s) just increased. that's how it works.             ║
║   now go build something.                                     ║
╚═══════════════════════════════════════════════════════════════╝
```

<a href="https://bapxai.com"><img src="https://img.shields.io/badge/🌐_Live_Demo-bapxai.com-D4AF37?style=for-the-badge" alt="Demo"/></a>
<a href="https://zenodo.org/records/19263435"><img src="https://img.shields.io/badge/📄_Paper-Zenodo-8b5cf6?style=for-the-badge" alt="Paper"/></a>
<a href="https://x.com/BAPxAI"><img src="https://img.shields.io/badge/🐦_Follow-@BAPxAI-1da1f2?style=for-the-badge" alt="Twitter"/></a>
<a href="https://orcid.org/0009-0007-3629-6404"><img src="https://img.shields.io/badge/ORCID-0009--0007--3629--6404-A6CE39?style=for-the-badge&logo=orcid&logoColor=white" alt="ORCID"/></a>

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=footer&text=Surplus+is+measurable&fontSize=28&fontColor=D4AF37&animation=twinkling" alt="Footer Wave"/>

</div>
