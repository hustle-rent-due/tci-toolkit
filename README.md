# TCI Toolkit — Thermodynamic Cognition Index

**Open-source reference implementation of the TCI framework.**

By Nile Green | PermaMind | [@BAPxAI](https://x.com/BAPxAI) | [bapxai.com](https://bapxai.com)

Paper: [https://zenodo.org/records/19263435](https://zenodo.org/records/19263435)

---

## What is TCI?

The Thermodynamic Cognition Index measures surplus-driven behavioral capacity in persistent ML agents.

```
TCI(t) = k(s) * (F_total(t) - F_survival(s))
```

- **F_total(t)** — total prediction-error energy (cross-entropy loss, TD error, etc.)
- **F_survival(s)** — minimum energy for operational integrity (survival floor)
- **k(s)** — substrate sensitivity constant, grows with runtime

| TCI Value | Meaning |
|-----------|---------|
| TCI >= 0.6 | Grade A — Stable generativity |
| TCI 0.4–0.6 | Grade B — Healthy |
| TCI 0.3–0.4 | Grade C — At risk |
| TCI 0.1–0.3 | Grade D — Collapse warning |
| TCI < 0.1 | Grade F — Collapse imminent |

---

## What is in this toolkit?

```
tci-toolkit/
├── tci/
│   ├── python/
│   │   ├── tci_calculator.py      # Core TCI computation
│   │   ├── k_estimator.py         # k(s) rolling window estimator
│   │   └── identity_tasks.py      # F_survival identity task suite
│   └── js/
│       ├── tci.js                 # JS reference implementation
│       └── k_estimator.js         # JS k(s) estimator
├── dashboard/
│   └── index.html                 # Live TCI monitor dashboard
├── examples/
│   ├── llm_agent_example.py       # LLM persistent agent example
│   └── rl_agent_example.py        # RL agent example
└── docs/
    └── operationalization.md      # Full operationalization guide
```

---

## Quick Start (Python)

```python
from tci.python.tci_calculator import TCICalculator
from tci.python.k_estimator import KEstimator

# Initialize
k_est = KEstimator(window_size=100)
tci = TCICalculator(f_survival=0.35)

# Each step
f_total = 0.72        # your model's current loss
complexity = 0.61     # novelty/entropy score

k = k_est.update(f_total - 0.35, complexity)
result = tci.compute(f_total, k)

print(result)
# {'tci': 0.74, 'grade': 'A', 'stage': 'Generativity', 'surplus': 0.37}
```

---

## Quick Start (JavaScript)

```javascript
import { TCICalculator, KEstimator } from './tci/js/tci.js';

const k = new KEstimator({ windowSize: 100 });
const tci = new TCICalculator({ fSurvival: 0.35 });

const result = tci.compute(0.72, k.update(0.37, 0.61));
console.log(result);
// { tci: 0.74, grade: 'A', stage: 'Generativity', surplus: 0.37 }
```

---

## Dashboard

Open `dashboard/index.html` in any browser for the live TCI monitor with real-time agent grading, fleet view, collapse alerts, and developmental stage tracking.

---

## License

MIT License. Use freely. Attribution appreciated.

Cite as: Green, N. (2026). Thermodynamic Cognition Index (TCI). Zenodo. https://zenodo.org/records/19263435

---

## Links

- Paper: https://zenodo.org/records/19263435
- Twitter: https://x.com/BAPxAI
- Site: https://bapxai.com
