"""Typed yes/no judgments.

The shape follows a Jev-style noul: one named question, one probability that
the answer is yes. Ranking is that probability sorted over candidates.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class NoulJudgment:
    """Calibrated-style yes/no. `noul` is P(yes), in (0, 1)."""

    question: str
    noul: float

    def as_dict(self) -> dict:
        return {"type": "noul", "question": self.question, "noul": float(self.noul)}


# Kept under the historical constant name; the question is now well posed:
# "is candidate perturbation P the one whose effect produced target pseudobulk T?"
MOVES_TOWARD = "is_candidate_perturbation_the_source_of_target_pseudobulk"
