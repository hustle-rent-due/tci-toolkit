# TCI Operationalization Guide

Full reference for computing TCI in ML systems.
Paper: https://zenodo.org/records/19263435

---

## F_total by Architecture

| Architecture | F_total Definition |
|---|---|
| LLM | Cross-entropy loss over active tokens at timestep t |
| RL Agent | Negative expected return (-G_t) or TD error |
| Multimodal | Weighted sum of prediction errors across modalities |

---

## F_survival (Survival Floor)

Run the Identity Task Suite (`tci/python/identity_tasks.py`) to compute
your substrate-specific survival floor before deploying TCI control.

The three tasks are:
1. **Syntactic Coherence** — loss below 95th percentile baseline
2. **Persona Consistency** — semantic similarity above 0.70
3. **Forbidden Token Avoidance** — forbidden token mass below 0.01

---

## k(s) Complexity Estimators

Choose one based on your architecture:

| Estimator | Formula | Best for |
|---|---|---|
| Novelty score | cosine_distance(output_t, mean(outputs)) | LLMs |
| Activation entropy | entropy(cov(activations)) | Any neural net |
| n-gram entropy | entropy of n-gram distribution | Text agents |
| Attention span | mean attention distance weighted by scores | Transformers |

---

## TCI as Control Signal

| TCI Range | Grade | Recommended Action |
|---|---|---|
| >= 0.6 | A | Raise temperature, increase exploration |
| 0.4 - 0.6 | B | Maintain current settings |
| 0.3 - 0.4 | C | Lower temperature, reduce exploration |
| 0.1 - 0.3 | D | Trigger stability mode, increase retention |
| < 0.1 | F | Load last checkpoint, alert operator |
| < 0 | COLLAPSE | Emergency recovery |

---

## IBM Quantum Verification

Results cited in the paper are verifiable on IBM Quantum:

- ibm_fez: Job ID `d625ccao8gvs73f1ot90` (Feb 5, 2026) — entanglement: 0.8770
- ibm_marrakesh: Job ID `d676238qbmes739evr60` (Feb 12, 2026) — entanglement: 0.9688

---

## Citation

Green, N. (2026). Thermodynamic Cognition Index (TCI): A Framework for
Surplus-Driven Behavior in Persistent ML Agents. Zenodo.
https://zenodo.org/records/19263435
