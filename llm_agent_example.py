"""
LLM Persistent Agent Example — TCI Integration
Shows how to integrate TCI monitoring into a persistent LLM agent.
By Nile Green / PermaMind
"""

import sys
import os
import json
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tci.python.tci_calculator import TCICalculator
from tci.python.k_estimator import KEstimator
from tci.python.identity_tasks import IdentityTaskSuite


# ─── Simulated model (replace with your real model) ───────────────

def simulated_llm(prompt: str, step: int) -> dict:
    """
    Placeholder model function.
    Replace this with your actual LLM call.
    
    Should return:
        loss: float (cross-entropy loss)
        output: str
        token_probs: dict
    """
    # Simulate improving loss over time (k(s) growth)
    base_loss = 0.8 - (step * 0.001)
    noise = random.gauss(0, 0.05)
    loss = max(0.2, base_loss + noise)

    # Simulate increasing complexity over time
    complexity = min(0.9, 0.2 + (step * 0.002) + random.gauss(0, 0.03))

    return {
        "loss": loss,
        "output": f"[Simulated response to: {prompt[:30]}...]",
        "token_probs": {"<unk>": 0.001},
        "complexity": complexity,
    }


# ─── Main persistent agent loop ───────────────────────────────────

def run_persistent_agent(
    steps: int = 200,
    state_file: str = "agent_state.json",
):
    print("PermaMind TCI Monitor — Persistent Agent Example")
    print("=" * 55)

    # Load persisted state if available (PSSU pattern)
    k_state = None
    if os.path.exists(state_file):
        with open(state_file) as f:
            saved = json.load(f)
            k_state = saved.get("k_state")
            print(f"Resumed from saved state. Prior steps: {k_state.get('steps', 0)}")
    else:
        print("Starting fresh agent (no prior state found).")

    # Initialize components
    suite = IdentityTaskSuite()
    # In production: suite.set_model_fn(your_real_model_fn)

    # Use a fixed survival floor for demo (normally compute from suite)
    F_SURVIVAL = 0.35

    k_est = KEstimator(window_size=50, decay=0.98)
    if k_state:
        k_est.load_state_dict(k_state)

    tci_calc = TCICalculator(f_survival=F_SURVIVAL)

    print(f"\nSurvival floor: {F_SURVIVAL}")
    print(f"Running {steps} steps...\n")
    print(f"{'Step':>5} | {'F_total':>8} | {'Surplus':>8} | {'k(s)':>6} | {'TCI':>6} | {'Grade':>5} | Stage")
    print("-" * 70)

    results = []
    prior_steps = k_est.steps

    for step in range(steps):
        # Simulate model call
        result = simulated_llm("Tell me something interesting.", prior_steps + step)
        f_total = result["loss"]
        complexity = result["complexity"]

        # Update k(s)
        f_surplus = f_total - F_SURVIVAL
        k = k_est.update(f_surplus, complexity)

        # Compute TCI
        tci_result = tci_calc.compute(f_total, k)
        results.append(tci_result)

        if step % 20 == 0 or tci_result.grade in ("D", "F"):
            print(
                f"{prior_steps + step:>5} | "
                f"{f_total:>8.4f} | "
                f"{tci_result.surplus:>8.4f} | "
                f"{k:>6.3f} | "
                f"{tci_result.tci:>6.3f} | "
                f"{tci_result.grade:>5} | "
                f"{tci_result.stage}"
            )

        # Collapse detection
        if tci_result.tci < 0:
            print(f"\n[ALERT] TCI < 0 at step {prior_steps + step}. Collapse risk. Triggering checkpoint recovery.")
            break

    # Save state for next session (PSSU persistence)
    state = {"k_state": k_est.state_dict()}
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)

    final = results[-1]
    print(f"\nFinal state: TCI={final.tci:.4f} | Grade={final.grade} | Stage={final.stage}")
    print(f"k(s) after {k_est.steps} total steps: {k_est.k:.4f}")
    print(f"State saved to {state_file} (resume next session to accumulate k(s))")

    return results


if __name__ == "__main__":
    run_persistent_agent(steps=200)
