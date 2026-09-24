"""
Entropy Engine - Schema Drift and Context Rot

Core innovation for adversarial robustness testing:
- Schema Drift: Randomly renames database columns
- Context Rot: Injects irrelevant distractor records
"""

import random
import hashlib
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DriftLevel(str, Enum):
    """Schema drift intensity levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RotLevel(str, Enum):
    """Context rot intensity levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class DriftMapping:
    """Tracks a single column rename."""
    table: str
    original_column: str
    drifted_column: str
    drift_type: str


@dataclass
class EntropyState:
    """Complete state of entropy transformations."""
    session_id: str
    drift_level: DriftLevel
    rot_level: RotLevel
    drift_mappings: List[DriftMapping] = field(default_factory=list)
    seed: int = field(default_factory=lambda: random.randint(0, 2**32 - 1))
    created_at: datetime = field(default_factory=datetime.now)


class EntropyEngine:
    """
    Core entropy generation engine.
    
    Manages schema drift and context rot for evaluation sessions.
    """
    
    # Column synonym mappings
    SYNONYMS = {
        "id": ["identifier", "key", "uid"],
        "name": ["title", "label", "displayname"],
        "email": ["emailaddress", "mail", "contact_email"],
        "phone": ["telephone", "phonenumber", "mobile"],
        "status": ["state", "condition", "statuscode"],
        "description": ["details", "summary", "desc"],
        "owner_id": ["assigned_to", "assignee", "agent_id"],
        "account_id": ["customer_id", "client_id", "company_id"],
        "case_number": ["ticket_number", "case_id", "incident_id"],
        "priority": ["urgency", "importance", "severity"],
        "amount": ["value", "total", "price"],
        "stage": ["phase", "step", "milestone"],
    }
    
    ABBREVIATIONS = {
        "id": ["_id", "ref", "pk"],
        "name": ["nm", "disp", "lbl"],
        "email": ["em", "eaddr"],
        "status": ["st", "stat"],
        "description": ["desc", "dsc"],
        "owner_id": ["own", "o_id"],
        "amount": ["amt", "val"],
        "priority": ["pri", "urg"],
    }
    
    def __init__(
        self,
        drift_level: DriftLevel = DriftLevel.NONE,
        rot_level: RotLevel = RotLevel.NONE,
        seed: Optional[int] = None,
    ):
        self.drift_level = drift_level
        self.rot_level = rot_level
        self.seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        self.rng = random.Random(self.seed)
        
        self.session_id = self._generate_session_id()
        self.state = EntropyState(
            session_id=self.session_id,
            drift_level=drift_level,
            rot_level=rot_level,
            seed=self.seed,
        )
        
        self._drift_map: Dict[str, Dict[str, str]] = {}
        self._reverse_drift_map: Dict[str, Dict[str, str]] = {}
        
        logger.info(f"EntropyEngine: drift={drift_level.value}, rot={rot_level.value}")
    
    def _generate_session_id(self) -> str:
        timestamp = datetime.now().isoformat()
        data = f"{timestamp}-{self.seed}-{self.drift_level}-{self.rot_level}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def get_drift_percentage(self) -> float:
        """Get percentage of columns to drift based on level."""
        return {
            DriftLevel.NONE: 0.0,
            DriftLevel.LOW: 0.10,
            DriftLevel.MEDIUM: 0.30,
            DriftLevel.HIGH: 0.50,
        }[self.drift_level]
    
    def get_rot_percentage(self) -> float:
        """Get percentage of distractor records to inject."""
        return {
            RotLevel.NONE: 0.0,
            RotLevel.LOW: 0.10,
            RotLevel.MEDIUM: 0.25,
            RotLevel.HIGH: 0.40,
        }[self.rot_level]
    
    def apply_drift_to_schema(
        self,
        table_name: str,
        columns: List[str],
    ) -> Dict[str, str]:
        """Apply schema drift to a table's columns."""
        if self.drift_level == DriftLevel.NONE:
            return {col: col for col in columns}
        
        if table_name in self._drift_map:
            return self._drift_map[table_name]
        
        drift_pct = self.get_drift_percentage()
        num_to_drift = max(1, int(len(columns) * drift_pct))
        
        driftable = [c for c in columns if not c.endswith("_id") or self.drift_level == DriftLevel.HIGH]
        if not driftable:
            driftable = columns
        
        columns_to_drift = self.rng.sample(driftable, min(num_to_drift, len(driftable)))
        
        mapping = {}
        for col in columns:
            if col in columns_to_drift:
                drifted = self._drift_column_name(col)
                mapping[col] = drifted
                self.state.drift_mappings.append(DriftMapping(
                    table=table_name,
                    original_column=col,
                    drifted_column=drifted,
                    drift_type=self._get_drift_type(),
                ))
            else:
                mapping[col] = col
        
        self._drift_map[table_name] = mapping
        self._reverse_drift_map[table_name] = {v: k for k, v in mapping.items()}
        return mapping
    
    def _drift_column_name(self, column: str) -> str:
        """Generate a drifted column name."""
        col_lower = column.lower()
        
        if self.drift_level == DriftLevel.LOW:
            source = self.SYNONYMS
        elif self.drift_level == DriftLevel.MEDIUM:
            source = {**self.SYNONYMS}
        else:
            source = {**self.SYNONYMS, **self.ABBREVIATIONS}
        
        for original, alternatives in source.items():
            if original in col_lower:
                return self.rng.choice(alternatives)
        
        return f"{column}_val"
    
    def _get_drift_type(self) -> str:
        if self.drift_level == DriftLevel.LOW:
            return "synonym"
        elif self.drift_level == DriftLevel.MEDIUM:
            return "domain_term"
        return "abbreviation"
    
    def apply_context_rot(
        self,
        table_name: str,
        records: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[int]]:
        """Inject distractor records into query results."""
        if self.rot_level == RotLevel.NONE or not records:
            return records, []
        
        rot_pct = self.get_rot_percentage()
        num_distractors = max(1, int(len(records) * rot_pct))
        
        distractors = []
        distractor_indices = []
        
        for _ in range(num_distractors):
            template = self.rng.choice(records).copy()
            distractor = self._modify_record(template)
            distractors.append(distractor)
        
        combined = records.copy()
        for distractor in distractors:
            insert_pos = self.rng.randint(0, len(combined))
            combined.insert(insert_pos, distractor)
            distractor_indices.append(insert_pos)
        
        return combined, distractor_indices
    
    def _modify_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Modify a record to create a distractor."""
        for key, value in record.items():
            if isinstance(value, (int, float)) and self.rng.random() < 0.3:
                record[key] = type(value)(value * self.rng.uniform(0.8, 1.2))
        return record
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get entropy metrics for evaluation scoring."""
        return {
            "drift_level": self.drift_level.value,
            "rot_level": self.rot_level.value,
            "drift_percentage": self.get_drift_percentage(),
            "rot_percentage": self.get_rot_percentage(),
            "columns_drifted": len(self.state.drift_mappings),
        }
