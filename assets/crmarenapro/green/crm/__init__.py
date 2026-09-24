"""
CRM Evaluation Modules for Entropic CRMArena Green Agent.

Provides:
- Task loading from HuggingFace CRMArenaPro dataset
- Entropy Engine (Schema Drift + Context Rot)
- Answer evaluation with CRMArena-compatible scoring
- 7-Dimension scoring for comprehensive agent assessment
"""

from crm.tasks import TaskLoader, CRMTask, TASK_CATEGORIES
from crm.entropy import EntropyEngine, DriftLevel, RotLevel
from crm.evaluator import CRMArenaEvaluator
from crm.scorer import SevenDimensionScorer, AgentMetrics

__all__ = [
    "TaskLoader",
    "CRMTask",
    "TASK_CATEGORIES",
    "EntropyEngine",
    "DriftLevel",
    "RotLevel",
    "CRMArenaEvaluator",
    "SevenDimensionScorer",
    "AgentMetrics",
]
