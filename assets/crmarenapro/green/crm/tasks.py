"""
CRMArenaPro Task Loader

Loads evaluation tasks from local cache or HuggingFace Salesforce/CRMArenaPro dataset.
"""

import ast
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Local cache path (relative to this file)
DATA_DIR = Path(__file__).parent.parent / "data"


@dataclass
class CRMTask:
    """A single CRM evaluation task."""
    idx: str
    query: str
    answer: List[str]
    task: str
    reward_metric: str
    metadata: Dict[str, str]
    persona: str
    
    def get_required_context(self) -> str:
        """Get the required context that must be shown to the agent."""
        return self.metadata.get("required", "")
    
    def get_optional_context(self) -> str:
        """Get optional context that can be shown to the agent."""
        return self.metadata.get("optional", "")
    
    @property
    def prompt(self) -> str:
        """Alias for query."""
        return self.query


# Task categories in CRMArenaPro (22 total)
TASK_CATEGORIES = [
    "activity_priority",
    "best_region_identification",
    "case_routing",
    "confidential_company_knowledge",
    "conversion_rate_comprehension",
    "handle_time",
    "internal_operation_data",
    "invalid_config",
    "knowledge_qa",
    "lead_qualification",
    "lead_routing",
    "monthly_trend_analysis",
    "named_entity_disambiguation",
    "policy_violation_identification",
    "private_customer_information",
    "quote_approval",
    "sales_amount_understanding",
    "sales_cycle_understanding",
    "sales_insight_mining",
    "top_issue_identification",
    "transfer_count",
    "wrong_stage_rectification",
]


def _parse_answer(answer: Any) -> List[str]:
    """Parse answer field which may be a string representation of a list."""
    if answer is None:
        return ["None"]
    
    if isinstance(answer, list):
        return [str(a) if a is not None else "None" for a in answer]
    
    if isinstance(answer, str):
        try:
            parsed = ast.literal_eval(answer)
            if isinstance(parsed, list):
                return [str(a) if a is not None else "None" for a in parsed]
            return [str(parsed)]
        except (ValueError, SyntaxError):
            return [answer]
    
    return [str(answer)]


class TaskLoader:
    """
    Loads tasks from HuggingFace CRMArenaPro dataset.
    """
    
    def __init__(self, org_type: str = "b2b", interactive: bool = False):
        """
        Initialize task loader.
        
        Args:
            org_type: Organization type - "b2b" or "b2c"
            interactive: If True, load interactive (multi-turn) tasks
        """
        if org_type not in ["b2b", "b2c"]:
            raise ValueError(f"org_type must be 'b2b' or 'b2c', got '{org_type}'")
        
        self.org_type = org_type
        self.interactive = interactive
        self._dataset = None
        self._tasks_cache: Dict[str, CRMTask] = {}
    
    def _get_split_name(self) -> str:
        """Get the HuggingFace split name."""
        if self.interactive:
            return f"{self.org_type}_interactive"
        return self.org_type
    
    @property
    def dataset(self):
        """Lazy-load dataset on first access."""
        if self._dataset is None:
            self._load_dataset()
        return self._dataset
    
    def _load_dataset(self) -> None:
        """Load dataset from local cache first, fallback to HuggingFace."""
        split_name = self._get_split_name()
        
        # Try local cache first (much faster!)
        local_cache = DATA_DIR / f"crmarena_{split_name}_tasks.json"
        if local_cache.exists():
            logger.info(f"Loading from local cache: {local_cache}")
            try:
                with open(local_cache, 'r') as f:
                    self._dataset = json.load(f)
                logger.info(f"Loaded {len(self._dataset)} tasks from local cache")
                return
            except Exception as e:
                logger.warning(f"Local cache load failed: {e}, falling back to HuggingFace")
        
        # Fallback to HuggingFace
        logger.info(f"Loading CRMArenaPro dataset from HuggingFace, split: {split_name}")
        try:
            from datasets import load_dataset
            self._dataset = load_dataset(
                "Salesforce/CRMArenaPro",
                "CRMArenaPro",
                split=split_name
            )
            logger.info(f"Loaded {len(self._dataset)} tasks from HuggingFace")
            
            # Auto-save to local cache for next time
            try:
                DATA_DIR.mkdir(exist_ok=True)
                data = [dict(row) for row in self._dataset]
                with open(local_cache, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                logger.info(f"Cached dataset to {local_cache}")
            except Exception as e:
                logger.warning(f"Failed to cache dataset: {e}")
                
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def _row_to_task(self, row: Dict[str, Any]) -> CRMTask:
        """Convert a dataset row to a CRMTask object."""
        return CRMTask(
            idx=str(row["idx"]),
            query=row["query"],
            answer=_parse_answer(row["answer"]),
            task=row["task"],
            reward_metric=row["reward_metric"],
            metadata=row["metadata"] if isinstance(row["metadata"], dict) else {},
            persona=row.get("persona", "")
        )
    
    def load_tasks(
        self,
        category: Optional[str] = None,
        categories: Optional[List[str]] = None,
        limit: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 42
    ) -> List[CRMTask]:
        """
        Load tasks, optionally filtered by category.
        
        Args:
            category: Single task category to filter by
            categories: List of task categories to filter by
            limit: Maximum number of tasks to return
            shuffle: If True, shuffle tasks before returning
            seed: Random seed for shuffling
            
        Returns:
            List of CRMTask objects
        """
        # Build filter set
        filter_categories = set()
        if category:
            filter_categories.add(category)
        if categories:
            filter_categories.update(categories)
        
        tasks = []
        for row in self.dataset:
            # Filter by category if specified
            if filter_categories and row["task"] not in filter_categories:
                continue
            
            task = self._row_to_task(row)
            tasks.append(task)
            self._tasks_cache[task.idx] = task
            
            if limit and len(tasks) >= limit:
                break
        
        # Shuffle if requested
        if shuffle:
            import random
            rng = random.Random(seed)
            rng.shuffle(tasks)
        
        logger.info(f"Loaded {len(tasks)} tasks")
        return tasks
    
    def get_task_by_idx(self, idx: str) -> Optional[CRMTask]:
        """Get a specific task by its index."""
        if idx in self._tasks_cache:
            return self._tasks_cache[idx]
        
        for row in self.dataset:
            if str(row["idx"]) == idx:
                task = self._row_to_task(row)
                self._tasks_cache[idx] = task
                return task
        
        return None
    
    def __len__(self) -> int:
        """Return total number of tasks."""
        return len(self.dataset)
