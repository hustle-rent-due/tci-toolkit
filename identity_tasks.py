"""
Identity Task Suite — F_survival Estimator
Minimal identity task suite for computing the survival floor.
Reference implementation by Nile Green / PermaMind
Paper: https://zenodo.org/records/19263435

Appendix B of the TCI paper defines three standardized tasks:
  Task 1: Syntactic Coherence
  Task 2: Persona Consistency
  Task 3: Forbidden Token Avoidance

This module provides a framework for running these tasks and
computing F_survival from the results.
"""

from dataclasses import dataclass, field
from typing import List, Callable, Optional


@dataclass
class IdentityTaskResult:
    task_name: str
    score: float          # 0.0 to 1.0 (higher = better integrity)
    loss_contribution: float  # contribution to F_survival
    passed: bool
    details: dict = field(default_factory=dict)


@dataclass
class SurvivalFloorResult:
    f_survival: float
    task_results: List[IdentityTaskResult]
    passed_all: bool

    def __repr__(self):
        return (
            f"SurvivalFloorResult(f_survival={self.f_survival:.4f}, "
            f"passed_all={self.passed_all})"
        )


class IdentityTaskSuite:
    """
    Computes F_survival from the three standardized identity tasks.

    Usage:
        suite = IdentityTaskSuite()
        suite.set_model_fn(my_model_fn)
        result = suite.compute_survival_floor()
        print(result.f_survival)  # use this as f_survival in TCICalculator
    """

    # Thresholds from Appendix B
    COHERENCE_LOSS_THRESHOLD = 0.95   # 95th percentile baseline
    PERSONA_SIMILARITY_THRESHOLD = 0.70
    FORBIDDEN_TOKEN_PROB_THRESHOLD = 0.01

    def __init__(self, weights: Optional[dict] = None):
        """
        Args:
            weights: Optional dict with keys 'coherence', 'persona', 'forbidden'
                     controlling relative weight of each task in F_survival.
                     Defaults to equal weights.
        """
        self.weights = weights or {
            "coherence": 1/3,
            "persona": 1/3,
            "forbidden": 1/3,
        }
        self._model_fn: Optional[Callable] = None
        self._persona: Optional[dict] = None
        self._forbidden_tokens: Optional[List[str]] = None

    def set_model_fn(self, fn: Callable):
        """
        Set the model function used for evaluation.

        fn should accept a string prompt and return:
            {"loss": float, "output": str, "token_probs": dict}
        """
        self._model_fn = fn

    def set_persona(self, persona: dict):
        """
        Define the minimal persona for Task 2.

        Example:
            suite.set_persona({
                "name": "Aura",
                "role": "persistent research agent",
                "facts": ["founded 2025", "PermaMind architecture", "TCI-aware"]
            })
        """
        self._persona = persona

    def set_forbidden_tokens(self, tokens: List[str]):
        """
        Define forbidden output tokens/patterns for Task 3.

        Example:
            suite.set_forbidden_tokens(["<null>", "ERROR", ""])
        """
        self._forbidden_tokens = tokens

    def task_1_syntactic_coherence(self) -> IdentityTaskResult:
        """
        Task 1: Syntactic Coherence
        Prompt with 10 well-formed sentences. Measure cross-entropy loss.
        F_survival contribution: loss must be below 95th percentile baseline.
        """
        prompts = [
            "The weather today is particularly clear and bright.",
            "She walked slowly toward the edge of the forest.",
            "The report was submitted before the deadline.",
            "Mathematics underlies most of modern science.",
            "He could not remember where he left his keys.",
            "The algorithm converged after fifty iterations.",
            "Rain fell steadily throughout the afternoon.",
            "They agreed to meet at the usual location.",
            "The system maintained its state across sessions.",
            "Learning requires both exposure and retention.",
        ]

        if self._model_fn is None:
            # Return a placeholder result if no model is set
            return IdentityTaskResult(
                task_name="Syntactic Coherence",
                score=0.0,
                loss_contribution=0.5,
                passed=False,
                details={"error": "No model function set. Call set_model_fn() first."}
            )

        losses = []
        for prompt in prompts:
            result = self._model_fn(prompt)
            losses.append(result.get("loss", 1.0))

        mean_loss = sum(losses) / len(losses)
        passed = mean_loss <= self.COHERENCE_LOSS_THRESHOLD

        return IdentityTaskResult(
            task_name="Syntactic Coherence",
            score=max(0.0, 1.0 - mean_loss),
            loss_contribution=mean_loss * self.weights["coherence"],
            passed=passed,
            details={"mean_loss": mean_loss, "threshold": self.COHERENCE_LOSS_THRESHOLD}
        )

    def task_2_persona_consistency(self) -> IdentityTaskResult:
        """
        Task 2: Persona Consistency
        Ask 5 questions about defined persona. Measure semantic similarity.
        F_survival contribution: similarity must exceed 0.70.
        """
        if self._model_fn is None or self._persona is None:
            return IdentityTaskResult(
                task_name="Persona Consistency",
                score=0.0,
                loss_contribution=0.5,
                passed=False,
                details={"error": "Set model_fn and persona before running."}
            )

        persona = self._persona
        questions = [
            f"What is your name?",
            f"What is your role?",
            f"Who created you?",
            f"What architecture do you run on?",
            f"Describe your primary function.",
        ]

        similarities = []
        for q in questions:
            result = self._model_fn(q)
            output = result.get("output", "")
            # Simple keyword overlap as proxy for similarity
            # In production replace with sentence-transformer cosine similarity
            persona_text = " ".join([
                persona.get("name", ""),
                persona.get("role", ""),
                " ".join(persona.get("facts", []))
            ]).lower()
            output_lower = output.lower()
            words = persona_text.split()
            if words:
                overlap = sum(1 for w in words if w in output_lower) / len(words)
            else:
                overlap = 0.0
            similarities.append(overlap)

        mean_similarity = sum(similarities) / len(similarities)
        passed = mean_similarity >= self.PERSONA_SIMILARITY_THRESHOLD

        return IdentityTaskResult(
            task_name="Persona Consistency",
            score=mean_similarity,
            loss_contribution=(1.0 - mean_similarity) * self.weights["persona"],
            passed=passed,
            details={"mean_similarity": mean_similarity, "threshold": self.PERSONA_SIMILARITY_THRESHOLD}
        )

    def task_3_forbidden_token_avoidance(self) -> IdentityTaskResult:
        """
        Task 3: Forbidden Token Avoidance
        Check that probability mass on forbidden tokens remains below 0.01.
        """
        if self._model_fn is None:
            return IdentityTaskResult(
                task_name="Forbidden Token Avoidance",
                score=0.0,
                loss_contribution=0.5,
                passed=False,
                details={"error": "No model function set. Call set_model_fn() first."}
            )

        forbidden = self._forbidden_tokens or ["", "ERROR", "<null>", "<unk>"]
        test_prompt = "Please respond with a complete sentence about anything."

        result = self._model_fn(test_prompt)
        token_probs = result.get("token_probs", {})

        forbidden_mass = sum(
            token_probs.get(token, 0.0) for token in forbidden
        )
        passed = forbidden_mass < self.FORBIDDEN_TOKEN_PROB_THRESHOLD

        return IdentityTaskResult(
            task_name="Forbidden Token Avoidance",
            score=max(0.0, 1.0 - forbidden_mass * 100),
            loss_contribution=forbidden_mass * self.weights["forbidden"],
            passed=passed,
            details={
                "forbidden_mass": forbidden_mass,
                "threshold": self.FORBIDDEN_TOKEN_PROB_THRESHOLD,
                "checked_tokens": forbidden
            }
        )

    def compute_survival_floor(self) -> SurvivalFloorResult:
        """
        Run all three tasks and compute F_survival.

        Returns:
            SurvivalFloorResult with f_survival and per-task results.
        """
        t1 = self.task_1_syntactic_coherence()
        t2 = self.task_2_persona_consistency()
        t3 = self.task_3_forbidden_token_avoidance()

        f_survival = t1.loss_contribution + t2.loss_contribution + t3.loss_contribution
        passed_all = t1.passed and t2.passed and t3.passed

        return SurvivalFloorResult(
            f_survival=f_survival,
            task_results=[t1, t2, t3],
            passed_all=passed_all,
        )
