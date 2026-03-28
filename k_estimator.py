"""
k(s) Estimator — Substrate Sensitivity Constant
Rolling window estimator for the TCI framework.
Reference implementation by Nile Green / PermaMind
Paper: https://zenodo.org/records/19263435
"""

from collections import deque
from typing import Optional
import math


class KEstimator:
    """
    Estimates k(s), the substrate sensitivity constant.

    k(s) = alpha * (delta_C / delta_F_surplus)

    k(s) measures how effectively a system converts surplus into
    behavioral complexity. It grows monotonically with accumulated
    runtime in persistent agents.

    Args:
        window_size: Rolling window size in timesteps (default 100).
        alpha: Normalization constant (default 1.0).
        decay: EMA decay for smoothing (default 0.99).
        k_init: Initial k value before enough data accumulates (default 0.1).
        min_steps: Minimum steps before k(s) is considered stable (default 500).
    """

    def __init__(
        self,
        window_size: int = 100,
        alpha: float = 1.0,
        decay: float = 0.99,
        k_init: float = 0.1,
        min_steps: int = 500,
    ):
        self.window_size = window_size
        self.alpha = alpha
        self.decay = decay
        self.k_init = k_init
        self.min_steps = min_steps

        self._surplus_window = deque(maxlen=window_size)
        self._complexity_window = deque(maxlen=window_size)
        self._prior_surplus_mean: Optional[float] = None
        self._prior_complexity_mean: Optional[float] = None
        self._k_ema: float = k_init
        self._steps: int = 0

    def update(self, f_surplus: float, complexity: float) -> float:
        """
        Update k(s) with a new observation.

        Args:
            f_surplus: Current surplus (F_total - F_survival).
            complexity: Current behavioral complexity estimate.
                        Options:
                        - Novelty: cosine distance from rolling output mean
                        - Activation entropy: entropy of activation covariance
                        - n-gram entropy: diversity of generated sequences
                        - Attention span: mean attention distance weighted by scores

        Returns:
            Current k(s) estimate.
        """
        self._surplus_window.append(f_surplus)
        self._complexity_window.append(complexity)
        self._steps += 1

        if len(self._surplus_window) < self.window_size:
            return self._k_ema

        current_surplus_mean = sum(self._surplus_window) / len(self._surplus_window)
        current_complexity_mean = sum(self._complexity_window) / len(self._complexity_window)

        if self._prior_surplus_mean is not None:
            d_surplus = current_surplus_mean - self._prior_surplus_mean
            d_complexity = current_complexity_mean - self._prior_complexity_mean

            if abs(d_surplus) > 1e-8:
                k_raw = self.alpha * (d_complexity / d_surplus)
                # Clamp to reasonable range
                k_raw = max(0.01, min(k_raw, 10.0))
                # Apply EMA smoothing
                self._k_ema = self.decay * self._k_ema + (1 - self.decay) * k_raw

        self._prior_surplus_mean = current_surplus_mean
        self._prior_complexity_mean = current_complexity_mean

        return self._k_ema

    @property
    def k(self) -> float:
        """Current k(s) estimate."""
        return self._k_ema

    @property
    def is_stable(self) -> bool:
        """True if k(s) has had enough steps to be considered stable."""
        return self._steps >= self.min_steps

    @property
    def steps(self) -> int:
        """Total number of timesteps observed."""
        return self._steps

    def reset(self):
        """Reset the estimator (simulates a session reset — k(s) lost)."""
        self._surplus_window.clear()
        self._complexity_window.clear()
        self._prior_surplus_mean = None
        self._prior_complexity_mean = None
        self._k_ema = self.k_init
        self._steps = 0

    def state_dict(self) -> dict:
        """Serialize estimator state for persistence across sessions."""
        return {
            "k_ema": self._k_ema,
            "steps": self._steps,
            "prior_surplus_mean": self._prior_surplus_mean,
            "prior_complexity_mean": self._prior_complexity_mean,
            "window_size": self.window_size,
            "alpha": self.alpha,
            "decay": self.decay,
        }

    def load_state_dict(self, state: dict):
        """Restore estimator state — enables k(s) persistence across sessions."""
        self._k_ema = state["k_ema"]
        self._steps = state["steps"]
        self._prior_surplus_mean = state.get("prior_surplus_mean")
        self._prior_complexity_mean = state.get("prior_complexity_mean")
        self.window_size = state["window_size"]
        self.alpha = state["alpha"]
        self.decay = state["decay"]
        self._surplus_window = deque(maxlen=self.window_size)
        self._complexity_window = deque(maxlen=self.window_size)
