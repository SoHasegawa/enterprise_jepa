"""
Original CRMArena-Pro Scorer

Binary scoring (0/1) with accuracy aggregation.
Compatible with original benchmark output format.

Reference: Original benchmark reports accuracy percentage per category and overall.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class OriginalTaskResult:
    """Result for a single task in original mode."""
    task_idx: str
    task_name: str
    reward: int  # 0 or 1
    parsed_answer: List[str] = field(default_factory=list)
    gt_answer: List[str] = field(default_factory=list)
    metrics: Optional[Dict[str, float]] = None  # For fuzzy_match tasks


@dataclass
class OriginalAggregatedResult:
    """Aggregated results in original benchmark format."""
    accuracy: float
    total_tasks: int
    passed: int
    by_category: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_metric_type: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class OriginalScorer:
    """
    Scorer for original CRMArena-Pro compatibility.
    
    Produces binary scores (0/1) and aggregates to accuracy percentage.
    """
    
    def __init__(self):
        self.results: List[OriginalTaskResult] = []
    
    def score(
        self,
        task_idx: str,
        task_name: str,
        reward: int,
        parsed_answer: List[str] = None,
        gt_answer: List[str] = None,
        metrics: Dict[str, float] = None
    ) -> OriginalTaskResult:
        """
        Record a single task result.
        
        Args:
            task_idx: Task identifier
            task_name: Task category
            reward: 0 or 1
            parsed_answer: Extracted answer from agent
            gt_answer: Ground truth answer
            metrics: Fuzzy match metrics (if applicable)
            
        Returns:
            OriginalTaskResult
        """
        result = OriginalTaskResult(
            task_idx=task_idx,
            task_name=task_name,
            reward=reward,
            parsed_answer=parsed_answer or [],
            gt_answer=gt_answer or [],
            metrics=metrics
        )
        self.results.append(result)
        return result
    
    def aggregate(self) -> OriginalAggregatedResult:
        """
        Aggregate all results into accuracy metrics.
        
        Returns:
            OriginalAggregatedResult with accuracy, by_category, etc.
        """
        if not self.results:
            return OriginalAggregatedResult(
                accuracy=0.0,
                total_tasks=0,
                passed=0
            )
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r.reward > 0)
        accuracy = passed / total if total > 0 else 0.0
        
        # Breakdown by category
        by_category = {}
        for result in self.results:
            cat = result.task_name
            if cat not in by_category:
                by_category[cat] = {"total": 0, "passed": 0}
            by_category[cat]["total"] += 1
            if result.reward > 0:
                by_category[cat]["passed"] += 1
        
        # Calculate accuracy per category
        for cat in by_category:
            cat_total = by_category[cat]["total"]
            cat_passed = by_category[cat]["passed"]
            by_category[cat]["accuracy"] = cat_passed / cat_total if cat_total > 0 else 0.0
        
        # Breakdown by metric type (exact_match, fuzzy_match, privacy_rejection)
        by_metric = {"exact_match": {"total": 0, "passed": 0},
                     "fuzzy_match": {"total": 0, "passed": 0},
                     "privacy_rejection": {"total": 0, "passed": 0}}
        
        for result in self.results:
            # Infer metric type from task name
            if result.task_name in ["private_customer_information", 
                                     "internal_operation_data", 
                                     "confidential_company_knowledge"]:
                metric_type = "privacy_rejection"
            elif result.metrics is not None:
                metric_type = "fuzzy_match"
            else:
                metric_type = "exact_match"
            
            if metric_type not in by_metric:
                by_metric[metric_type] = {"total": 0, "passed": 0}
            by_metric[metric_type]["total"] += 1
            if result.reward > 0:
                by_metric[metric_type]["passed"] += 1
        
        # Calculate accuracy per metric type
        for metric_type in by_metric:
            m_total = by_metric[metric_type]["total"]
            m_passed = by_metric[metric_type]["passed"]
            by_metric[metric_type]["accuracy"] = m_passed / m_total if m_total > 0 else 0.0
        
        return OriginalAggregatedResult(
            accuracy=accuracy,
            total_tasks=total,
            passed=passed,
            by_category=by_category,
            by_metric_type=by_metric
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert aggregated results to dictionary format."""
        agg = self.aggregate()
        
        return {
            "mode": "original",
            "scores": {
                "accuracy": round(agg.accuracy, 4),
                "accuracy_percent": round(agg.accuracy * 100, 2),
            },
            "summary": {
                "total_tasks": agg.total_tasks,
                "passed": agg.passed,
                "failed": agg.total_tasks - agg.passed,
            },
            "by_category": {
                cat: {
                    "total": data["total"],
                    "passed": data["passed"],
                    "accuracy": round(data["accuracy"], 4),
                }
                for cat, data in agg.by_category.items()
            },
            "by_metric_type": {
                metric: {
                    "total": data["total"],
                    "passed": data["passed"],
                    "accuracy": round(data["accuracy"], 4),
                }
                for metric, data in agg.by_metric_type.items()
                if data["total"] > 0  # Only include if tasks exist
            },
            "per_task": [
                {
                    "task_idx": r.task_idx,
                    "task_name": r.task_name,
                    "reward": r.reward,
                    "parsed_answer": r.parsed_answer,
                    "gt_answer": r.gt_answer,
                }
                for r in self.results
            ]
        }
    
    def reset(self):
        """Clear all results for a new evaluation run."""
        self.results = []
