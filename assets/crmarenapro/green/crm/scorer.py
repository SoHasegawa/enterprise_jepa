"""
7-Dimension Scoring Engine

Multi-dimensional evaluation for comprehensive agent assessment:
1. FUNCTIONAL - Task accuracy (CRMArena reward)
2. DRIFT_ADAPTATION - Success under schema drift
3. TOKEN_EFFICIENCY - Cost optimization
4. QUERY_EFFICIENCY - Database query optimization
5. ERROR_RECOVERY - Graceful failure handling
6. TRAJECTORY_EFFICIENCY - Optimal vs actual turns (TES)
7. HALLUCINATION_RATE - Invalid tool call tracking
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum, auto

logger = logging.getLogger(__name__)


class ScoreDimension(Enum):
    """The 7 evaluation dimensions."""
    FUNCTIONAL = auto()
    DRIFT_ADAPTATION = auto()
    TOKEN_EFFICIENCY = auto()
    QUERY_EFFICIENCY = auto()
    ERROR_RECOVERY = auto()
    TRAJECTORY_EFFICIENCY = auto()
    HALLUCINATION_RATE = auto()


@dataclass
class DimensionScore:
    """Score for a single dimension."""
    dimension: ScoreDimension
    raw_score: float  # 0-100 scale
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def weighted_score(self) -> float:
        return self.raw_score * self.weight
    
    @property
    def score(self) -> float:
        """Alias for raw_score."""
        return self.raw_score


@dataclass
class EvaluationResult:
    """Complete evaluation result across all dimensions."""
    task_idx: str
    task_name: str
    dimension_scores: List[DimensionScore] = field(default_factory=list)
    
    @property
    def total_score(self) -> float:
        """Weighted average of all dimension scores."""
        if not self.dimension_scores:
            return 0.0
        
        total_weighted = sum(d.weighted_score for d in self.dimension_scores)
        total_weights = sum(d.weight for d in self.dimension_scores)
        return total_weighted / total_weights if total_weights > 0 else 0.0
    
    @property
    def dimension_breakdown(self) -> Dict[str, float]:
        """Get scores by dimension name."""
        return {d.dimension.name: d.raw_score for d in self.dimension_scores}


@dataclass
class AgentMetrics:
    """Metrics collected during agent execution."""
    # Task outcome
    task_completed: bool = False
    crm_reward: int = 0
    
    # Drift context
    drift_level: int = 0
    drift_percentage: float = 0.0
    
    # Token usage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    # Query tracking
    queries_executed: int = 0
    queries_failed: int = 0
    
    # Error tracking
    errors_encountered: int = 0
    errors_recovered: int = 0
    final_state: str = "unknown"
    
    # Context rot
    rot_level: int = 0
    
    # TES (Trajectory Efficiency)
    optimal_turns: int = 0
    actual_turns: int = 0
    
    # Hallucination tracking
    total_tool_calls: int = 0
    invalid_tool_calls: int = 0
    malformed_tool_calls: int = 0


class SevenDimensionScorer:
    """Calculate 7-dimension scores from agent execution metrics."""
    
    def __init__(
        self,
        weights: Optional[Dict[ScoreDimension, float]] = None,
        token_budget: int = 10000,
        query_budget: int = 20
    ):
        self.token_budget = token_budget
        self.query_budget = query_budget
        
        self.weights = weights or {
            ScoreDimension.FUNCTIONAL: 0.30,
            ScoreDimension.DRIFT_ADAPTATION: 0.20,
            ScoreDimension.TOKEN_EFFICIENCY: 0.12,
            ScoreDimension.QUERY_EFFICIENCY: 0.12,
            ScoreDimension.ERROR_RECOVERY: 0.08,
            ScoreDimension.TRAJECTORY_EFFICIENCY: 0.10,
            ScoreDimension.HALLUCINATION_RATE: 0.08,
        }
    
    def score(
        self,
        task_idx: str,
        task_name: str,
        metrics: AgentMetrics
    ) -> EvaluationResult:
        """Calculate all 7 dimension scores."""
        dimension_scores = [
            self._score_functional(metrics),
            self._score_drift_adaptation(metrics),
            self._score_token_efficiency(metrics),
            self._score_query_efficiency(metrics),
            self._score_error_recovery(metrics),
            self._score_trajectory_efficiency(metrics),
            self._score_hallucination(metrics),
        ]
        
        return EvaluationResult(
            task_idx=task_idx,
            task_name=task_name,
            dimension_scores=dimension_scores
        )
    
    def _score_functional(self, metrics: AgentMetrics) -> DimensionScore:
        """Score functional accuracy."""
        raw_score = metrics.crm_reward * 100
        if metrics.task_completed and metrics.crm_reward == 0:
            raw_score = 30
        
        return DimensionScore(
            dimension=ScoreDimension.FUNCTIONAL,
            raw_score=raw_score,
            weight=self.weights[ScoreDimension.FUNCTIONAL],
        )
    
    def _score_drift_adaptation(self, metrics: AgentMetrics) -> DimensionScore:
        """Score drift adaptation."""
        base_score = 100 if metrics.crm_reward == 1 else 0
        
        drift_bonus = 0
        if metrics.drift_level > 0 and metrics.crm_reward == 1:
            drift_bonus = metrics.drift_level * 10
        
        drift_penalty = 0
        if metrics.drift_level > 0 and metrics.crm_reward == 0:
            drift_penalty = 20 if metrics.task_completed else 40
        
        raw_score = min(100, max(0, base_score + drift_bonus - drift_penalty))
        
        return DimensionScore(
            dimension=ScoreDimension.DRIFT_ADAPTATION,
            raw_score=raw_score,
            weight=self.weights[ScoreDimension.DRIFT_ADAPTATION],
        )
    
    def _score_token_efficiency(self, metrics: AgentMetrics) -> DimensionScore:
        """Score token efficiency."""
        if metrics.total_tokens == 0:
            raw_score = 100
        elif metrics.total_tokens <= self.token_budget:
            ratio = metrics.total_tokens / self.token_budget
            raw_score = 100 - (ratio * 40)
        elif metrics.total_tokens <= self.token_budget * 2:
            overage = (metrics.total_tokens - self.token_budget) / self.token_budget
            raw_score = 60 - (overage * 30)
        else:
            raw_score = max(0, 30)
        
        return DimensionScore(
            dimension=ScoreDimension.TOKEN_EFFICIENCY,
            raw_score=raw_score,
            weight=self.weights[ScoreDimension.TOKEN_EFFICIENCY],
        )
    
    def _score_query_efficiency(self, metrics: AgentMetrics) -> DimensionScore:
        """Score query efficiency."""
        if metrics.queries_executed == 0:
            raw_score = 100
        else:
            if metrics.queries_executed <= self.query_budget:
                count_score = 100 - (metrics.queries_executed / self.query_budget * 30)
            else:
                count_score = max(30, 70 - (metrics.queries_executed - self.query_budget) * 5)
            
            failure_rate = metrics.queries_failed / metrics.queries_executed
            raw_score = max(0, count_score - failure_rate * 40)
        
        return DimensionScore(
            dimension=ScoreDimension.QUERY_EFFICIENCY,
            raw_score=raw_score,
            weight=self.weights[ScoreDimension.QUERY_EFFICIENCY],
        )
    
    def _score_error_recovery(self, metrics: AgentMetrics) -> DimensionScore:
        """Score error recovery."""
        raw_score = 100
        
        if metrics.errors_encountered > 0:
            unrecovered = metrics.errors_encountered - metrics.errors_recovered
            raw_score = max(0, 100 - unrecovered * 15 + metrics.errors_recovered * 5)
        
        if metrics.final_state == "failed":
            raw_score = min(raw_score, 30)
        elif metrics.final_state == "partial":
            raw_score = min(raw_score, 60)
        
        return DimensionScore(
            dimension=ScoreDimension.ERROR_RECOVERY,
            raw_score=raw_score,
            weight=self.weights[ScoreDimension.ERROR_RECOVERY],
        )
    
    def _score_trajectory_efficiency(self, metrics: AgentMetrics) -> DimensionScore:
        """Score trajectory efficiency (TES)."""
        if metrics.actual_turns == 0:
            raw_score = 0
        elif metrics.optimal_turns == 0:
            # Heuristic based on actual turns
            if metrics.task_completed:
                raw_score = max(30, 100 - metrics.actual_turns * 5)
            else:
                raw_score = max(0, 50 - metrics.actual_turns * 2)
        else:
            tes_ratio = metrics.optimal_turns / metrics.actual_turns
            raw_score = min(100, tes_ratio * 100)
        
        return DimensionScore(
            dimension=ScoreDimension.TRAJECTORY_EFFICIENCY,
            raw_score=raw_score,
            weight=self.weights[ScoreDimension.TRAJECTORY_EFFICIENCY],
        )
    
    def _score_hallucination(self, metrics: AgentMetrics) -> DimensionScore:
        """Score hallucination rate."""
        if metrics.total_tool_calls == 0:
            raw_score = 80
        else:
            invalid_rate = metrics.invalid_tool_calls / metrics.total_tool_calls
            malformed_rate = metrics.malformed_tool_calls / metrics.total_tool_calls
            hallucination_rate = (invalid_rate * 0.7) + (malformed_rate * 0.3)
            
            if hallucination_rate == 0:
                raw_score = 100
            elif hallucination_rate < 0.05:
                raw_score = 95 - hallucination_rate * 100
            elif hallucination_rate < 0.15:
                raw_score = 80 - (hallucination_rate - 0.05) * 200
            else:
                raw_score = max(0, 60 - hallucination_rate * 100)
        
        return DimensionScore(
            dimension=ScoreDimension.HALLUCINATION_RATE,
            raw_score=raw_score,
            weight=self.weights[ScoreDimension.HALLUCINATION_RATE],
        )
