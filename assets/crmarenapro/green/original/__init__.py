"""
Original CRMArena-Pro Compatibility Mode

This module provides evaluation logic compatible with Salesforce's original
CRMArena-Pro benchmark, enabling direct score comparison.

By default, both original and entropic scores are computed for each task.
Use `skip_original: true` in config to disable original scoring.

Reference: https://github.com/SalesforceAIResearch/CRMArena
"""

from .evaluator import OriginalEvaluator
from .scorer import OriginalScorer
from .metrics import get_all_metrics, exact_match_score, f1_score

__all__ = [
    "OriginalEvaluator",
    "OriginalScorer", 
    "get_all_metrics",
    "exact_match_score",
    "f1_score",
]
