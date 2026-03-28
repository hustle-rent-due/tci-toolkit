"""
TCI Calculator — Thermodynamic Cognition Index
Reference implementation by Nile Green / PermaMind
Paper: https://zenodo.org/records/19263435
"""

from dataclasses import dataclass
from typing import Optional


GRADES = [
    (0.6, "A", "Generativity"),
    (0.4, "B", "Learning"),
    (0.3, "C", "At Risk"),
    (0.1, "D", "Collapse Warning"),
    (0.0, "F", "Collapse Imminent"),
]


@dataclass
class TCIResult:
    tci: float
    surplus: float
    grade: str
    stage: str
    f_total: float
    f_survival: float
    k: float

    def __repr__(self):
        return (
            f"TCIResult(tci={self.tci:.4f}, grade={self.grade}, "
            f"stage={self.stage}, surplus={self.surplus:.4f})"
        )

    def to_dict(self):
        return {
            "tci": round(self.tci, 4),
            "surplus": round(self.surplus, 4),
            "grade": self.grade,
            "stage": self.stage,
            "f_total": round(self.f_total, 4),
            "f_survival": round(self.f_survival, 4),
            "k": round(self.k, 4),
        }


class TCICalculator:
    """
    Computes the Thermodynamic Cognition Index.

    TCI(t) = k(s) * (F_total(t) - F_survival(s))

    Args:
        f_survival: The survival floor for this substrate.
                    For LLMs: baseline cross-entropy on identity tasks.
                    For RL agents: expected return of minimal viable policy.
    """

    def __init__(self, f_survival: float):
        if f_survival < 0:
            raise ValueError("f_survival must be >= 0")
        self.f_survival = f_survival

    def compute(self, f_total: float, k: float) -> TCIResult:
        """
        Compute TCI at a single timestep.

        Args:
            f_total: Current prediction-error energy.
                     For LLMs: current cross-entropy loss.
                     For RL: negative expected return or TD error.
            k: Current substrate sensitivity constant from KEstimator.

        Returns:
            TCIResult with tci, grade, stage, surplus.
        """
        surplus = f_total - self.f_survival
        tci = k * surplus
        grade, stage = self._grade(tci)

        return TCIResult(
            tci=tci,
            surplus=surplus,
            grade=grade,
            stage=stage,
            f_total=f_total,
            f_survival=self.f_survival,
            k=k,
        )

    def _grade(self, tci: float):
        for threshold, grade, stage in GRADES:
            if tci >= threshold:
                return grade, stage
        return "F", "Collapse Imminent"

    def set_survival_floor(self, f_survival: float):
        """Update the survival floor (e.g. after recalibration)."""
        self.f_survival = f_survival


# Convenience function
def compute_tci(f_total: float, f_survival: float, k: float) -> TCIResult:
    """One-shot TCI computation without maintaining state."""
    return TCICalculator(f_survival=f_survival).compute(f_total, k)
