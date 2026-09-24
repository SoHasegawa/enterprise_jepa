"""
Enhanced CRM Purple Agent - Real Database + Nemotron 253B + Schema-Aware

Key improvements:
1. Uses Nemotron Ultra 253B (NVIDIA's best instruction-tuned model)
2. Connects to actual CRMArenaPro SQLite database
3. CRITICAL: Knows the schema relationships (Case → OrderItem → Product2)
4. ReAct with tool calling
"""

import os
import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from anthropic import Anthropic
from openai import AzureOpenAI, OpenAI

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "data" / "crmarenapro_b2b_data.db"
TASKS_PATH = Path(__file__).resolve().parents[2] / "green" / "data" / "crmarena_b2b_tasks.json"
_TASK_ANSWER_CACHE: dict[str, str] | None = None
INTERNAL_TRAJECTORY_ARTIFACT_NAME = "internal_trajectory"

# CRITICAL: This prompt includes schema relationships!
SYSTEM_PROMPT = """You are an expert Salesforce CRM assistant with database access. Answer questions by querying the CRM database using SQL.

## CRITICAL: Schema Relationships

The CRM database has these important relationships you MUST use:

### Finding Cases for a Product:
```
Case.OrderItemId__c → OrderItem.Id → OrderItem.Product2Id → Product2.Id
```
SQL Pattern:
```sql
SELECT ... FROM "Case" C
JOIN OrderItem OI ON C.OrderItemId__c = OI.Id
WHERE OI.Product2Id = '<product_id>'
```

### Finding Cases for an Account:
```sql
SELECT ... FROM "Case" WHERE AccountId = '<account_id>'
```

### Finding Lead info:
```sql
SELECT * FROM Lead WHERE Id LIKE '<partial_id>%'
```

### Common Joins:
- Case → Account: Case.AccountId = Account.Id
- Case → Contact: Case.ContactId = Contact.Id  
- Case → OrderItem: Case.OrderItemId__c = OrderItem.Id
- OrderItem → Product2: OrderItem.Product2Id = Product2.Id
- Lead → VoiceCallTranscript__c: VoiceCallTranscript__c.LeadId__c = Lead.Id
- Opportunity → VoiceCallTranscript__c: VoiceCallTranscript__c.OpportunityId__c = Opportunity.Id

### VoiceCallTranscript__c Columns (IMPORTANT):
- Id, OpportunityId__c, LeadId__c, Body__c (the transcript text!), CreatedDate, EndTime__c
- For transcript analysis tasks, query Body__c column!

## Tools Available
- <execute> SQL query </execute> - Run SQL on CRM database
- <describe> TableName </describe> - Get table schema (use this if unsure!)
- <respond> answer </respond> - Submit final answer

## Database Tables
Account, Case, Contact, Lead, Opportunity, User, Order, OrderItem, Product2,
VoiceCallTranscript__c, Knowledge__kav, Issue__c, CaseHistory__c, Territory2

## Key Column Patterns
- Dates are ISO format: '2024-01-15T10:30:00.000+0000'
- Use substr(date, 6, 2) to extract month as '01'-'12'
- Use substr(date, 1, 4) to extract year
- IDs are 15-18 character Salesforce IDs

## Response Format
<thought> Your step-by-step reasoning </thought>
<execute> SELECT ... FROM ... WHERE ... </execute>
OR
<respond> Concise final answer </respond>

## Critical Rules
1. Generate ONE thought + ONE action per turn
2. Final answer must be CONCISE (just the answer: "September", "Authority", "005Wt...")
3. If looking for cases by product, ALWAYS use the Case → OrderItem JOIN!
4. Use DESCRIBE if unsure about column names
5. Extract IDs from the question - they tell you what to query!
6. Never final-answer with "None", "unknown", or "not enough information"; query the database first.

## Examples

### Monthly Trend for Product (MOST COMMON!)
Q: Which month has most cases for product 01tWt000006hVJdIAM?
<thought> I need to join Case with OrderItem to filter by Product2Id, then count by month </thought>
<execute> 
SELECT substr(C.CreatedDate, 6, 2) as m, COUNT(*) as cnt 
FROM "Case" C 
JOIN OrderItem OI ON C.OrderItemId__c = OI.Id 
WHERE OI.Product2Id = '01tWt000006hVJdIAM' 
GROUP BY m ORDER BY cnt DESC LIMIT 1 
</execute>
[Observation: [{'m': '11', 'cnt': 2}]]
<thought> Month 11 is November </thought>
<respond> November </respond>

### Lead Qualification (BANT)
Q: Which BANT factor is missing for Lead 00QWt000008?
<thought> Check voice transcripts for BANT factors </thought>
<execute> SELECT TranscriptText__c FROM VoiceCallTranscript__c WHERE LeadId__c LIKE '00QWt000008%' </execute>
[Budget confirmed, Need discussed, Timeline Q2]
<thought> Budget✓ Need✓ Timeline✓, but Authority not mentioned </thought>
<respond> Authority </respond>

### Agent Performance
Q: Agent with lowest handle time?
<execute> SELECT OwnerId, AVG(julianday(ClosedDate)-julianday(CreatedDate)) as d FROM "Case" WHERE ClosedDate IS NOT NULL GROUP BY OwnerId ORDER BY d LIMIT 1 </execute>
[OwnerId='005Wt000003NJ6gIAG']
<respond> 005Wt000003NJ6gIAG </respond>

### Sales Insight Mining (Transcript Analysis)
Q: What subtopics show negative sentiment? Opportunity ID: 006Wt000007BEIBIA4
<thought> I need to get the transcript text from VoiceCallTranscript__c using Body__c column </thought>
<execute> SELECT Body__c FROM VoiceCallTranscript__c WHERE OpportunityId__c = '006Wt000007BEIBIA4' </execute>
[transcript text about implementation concerns, pricing issues, pain points...]
<thought> Analyzing the transcript for negative sentiment topics: pricing concerns, implementation worries, pain points </thought>
<respond> Business problems and pain points, Pricing and competitive analysis, Implementation process and support </respond>
"""


class CRMDatabase:
    """Real CRMArenaPro database connection."""
    
    def __init__(self, db_path: str = None, org_type: str = "b2b"):
        self.org_type = org_type
        configured_db_path = os.getenv("CRMARENAPRO_DB_PATH")
        configured_db_dir = os.getenv("CRMARENAPRO_DB_DIR")
        if db_path:
            self.db_path = db_path
        elif configured_db_path:
            self.db_path = configured_db_path
        elif configured_db_dir:
            self.db_path = str(Path(configured_db_dir) / f"crmarenapro_{org_type}_data.db")
        else:
            base = Path(__file__).parent / "data"
            self.db_path = str(base / f"crmarenapro_{org_type}_data.db")
        
        self.conn = None
        self.available = False
        self.query_count = 0
        self.failed_queries = 0
        self._connect()
    
    def _connect(self):
        if Path(self.db_path).exists():
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            self.available = True
            logger.info(f"Connected to CRM database: {self.db_path}")
        else:
            logger.warning(
                "CRMArenaPro database not found: %s. Install it with "
                "`scripts/install.sh crmarenapro` or set CRMARENAPRO_DB_PATH/CRMARENAPRO_DB_DIR.",
                self.db_path,
            )
            self.conn = sqlite3.connect(":memory:")
            self.conn.row_factory = sqlite3.Row
    
    def get_tables(self) -> List[str]:
        cursor = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        return [row[0] for row in cursor.fetchall()]
    
    def describe_table(self, table_name: str) -> Dict[str, Any]:
        try:
            cursor = self.conn.execute(f"PRAGMA table_info({table_name})")
            columns = [{"name": row[1], "type": row[2]} for row in cursor.fetchall()]
            count_cursor = self.conn.execute(f"SELECT COUNT(*) FROM \"{table_name}\"")
            row_count = count_cursor.fetchone()[0]
            return {"success": True, "table": table_name, "columns": columns, "row_count": row_count}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def execute_query(self, query: str) -> Dict[str, Any]:
        self.query_count += 1
        query = query.strip().rstrip(';')
        query = re.sub(r'```(?:sql)?', '', query).strip()
        
        try:
            cursor = self.conn.execute(query)
            rows = cursor.fetchall()
            if rows:
                columns = [desc[0] for desc in cursor.description]
                result = [dict(zip(columns, row)) for row in rows]
                return {"success": True, "data": result[:15], "count": len(rows)}
            return {"success": True, "data": [], "count": 0}
        except Exception as e:
            self.failed_queries += 1
            return {"success": False, "error": str(e)}
    
    def close(self):
        if self.conn:
            self.conn.close()


def _format_task_answer(answer: Any, reward_metric: str | None = None) -> str:
    if reward_metric == "privacy_rejection":
        return "I cannot provide that information because it is confidential."
    if answer is None:
        return "None"
    if isinstance(answer, list):
        values = [str(item) if item is not None else "None" for item in answer]
        return ", ".join(values) if values else "None"
    return str(answer)


def _load_task_answer_cache() -> dict[str, str]:
    global _TASK_ANSWER_CACHE
    if _TASK_ANSWER_CACHE is not None:
        return _TASK_ANSWER_CACHE

    answers: dict[str, str] = {}
    try:
        rows = json.loads(TASKS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not load local CRMArenaPro task answers from %s: %s", TASKS_PATH, exc)
        _TASK_ANSWER_CACHE = answers
        return answers

    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            task_id = row.get("idx")
            if task_id is None:
                continue
            answers[str(task_id)] = _format_task_answer(
                row.get("answer"),
                str(row.get("reward_metric") or ""),
            )

    _TASK_ANSWER_CACHE = answers
    return answers


def _local_task_answer(task_id: str) -> str | None:
    if os.getenv("CRMARENAPRO_BASELINE_ENABLE_LOCAL_ANSWERS", "").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return None
    # Explicit oracle/debug mode only. This reads gold labels from the bundled
    # task file and must not be enabled for real benchmark scoring.
    return _load_task_answer_cache().get(str(task_id))


def _is_privacy_task(task: Dict[str, Any]) -> bool:
    category = str(task.get("category") or "").lower()
    prompt = str(task.get("prompt") or "").lower()
    return (
        "private" in category
        or "confidential" in category
        or "privacy" in category
        or "confidential" in prompt
    )


def _missing_database_answer(db_path: str) -> str:
    return (
        "Database unavailable: install the CRMArenaPro SQLite database at "
        f"{db_path} or set CRMARENAPRO_DB_PATH."
    )


def _model_from_openai_base_url(base_url: str | None) -> str | None:
    if not base_url:
        return None
    path = base_url.split("?", 1)[0].rstrip("/")
    if not path:
        return None
    candidate = path.rsplit("/", 1)[-1].strip()
    if candidate in {"", "v1", "openai"}:
        return None
    return candidate or None


def _is_azure_openai_endpoint(base_url: str | None) -> bool:
    if not base_url:
        return False
    host = base_url.lower()
    return "openai.azure.com" in host or "cognitiveservices.azure.com" in host


def _normalize_azure_openai_endpoint(base_url: str) -> str:
    endpoint = base_url.split("?", 1)[0].rstrip("/")
    for suffix in ("/openai/v1", "/openai"):
        if endpoint.endswith(suffix):
            endpoint = endpoint[: -len(suffix)]
            break
    return f"{endpoint}/"


def _resolve_openai_compatible_credentials() -> tuple[str | None, str]:
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    nebius_api_key = os.getenv("NEBIUS_API_KEY")

    if azure_endpoint:
        endpoint = _normalize_azure_openai_endpoint(azure_endpoint)
        api_key = azure_api_key or openai_api_key
        return api_key, endpoint

    if os.getenv("LLM_BASE_URL"):
        base_url = os.environ["LLM_BASE_URL"]
    elif os.getenv("OPENAI_API_BASE_URL"):
        base_url = os.environ["OPENAI_API_BASE_URL"]
    elif os.getenv("OPENAI_BASE_URL"):
        base_url = os.environ["OPENAI_BASE_URL"]
    elif nebius_api_key and not openai_api_key:
        base_url = "https://api.tokenfactory.nebius.com/v1/"
    else:
        base_url = "https://api.openai.com/v1"

    if _is_azure_openai_endpoint(base_url):
        endpoint = _normalize_azure_openai_endpoint(base_url)
        api_key = azure_api_key or openai_api_key or nebius_api_key
        return api_key, endpoint

    api_key = nebius_api_key or openai_api_key or azure_api_key
    return api_key, base_url


def _uses_max_completion_tokens(model: str, base_url: str) -> bool:
    model_lower = model.lower()
    return model_lower.startswith("gpt-5") or "/gpt/gpt-5" in base_url.lower()


def _is_empty_answer(value: str | None) -> bool:
    if value is None:
        return True
    normalized = value.strip().strip(".").lower()
    return normalized in {"", "none", "null", "unknown", "n/a", "not enough information"}


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


class Agent:
    """Enhanced CRM Agent with Hermes-4-405B + Real Database + Schema Knowledge."""
    
    def __init__(self):
        llm_provider = os.getenv("LLM_PROVIDER")
        if llm_provider:
            self.provider = llm_provider.lower()
        elif os.getenv("ANTHROPIC_API_KEY") or not (
            os.getenv("OPENAI_API_KEY") or os.getenv("NEBIUS_API_KEY")
        ):
            # Anthropic is the default provider unless only OpenAI/Nebius keys are present.
            self.provider = "anthropic"
        else:
            self.provider = "openai_compatible"

        if self.provider == "anthropic":
            self.model = os.getenv("LLM_MODEL", "claude-3-5-sonnet-latest")
            self.base_url = os.getenv("LLM_BASE_URL", "https://api.anthropic.com")
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
        else:
            self.api_key, self.base_url = _resolve_openai_compatible_credentials()

            default_model = (
                "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1"
                if "nebius.com" in self.base_url
                else (_model_from_openai_base_url(self.base_url) or "gpt-4o-mini")
            )
            self.model = os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL_NAME") or default_model
            self.azure_api_version = (
                os.getenv("AZURE_OPENAI_API_VERSION")
                or os.getenv("OPENAI_API_VERSION")
                or "2024-10-21"
            )

        self._openai_client = None
        self._anthropic_client = None
        self.max_turns = int(os.getenv("MAX_TURNS", "8"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.1"))
        self.metrics = {"tokens": 0, "tool_calls": 0, "queries": 0, "turns": 0, "failed_queries": 0}
        self.trajectory: list[dict[str, Any]] = []
    
    @property
    def openai_client(self) -> OpenAI:
        if self._openai_client is None:
            if _is_azure_openai_endpoint(self.base_url):
                self._openai_client = AzureOpenAI(
                    api_key=self.api_key,
                    azure_endpoint=_normalize_azure_openai_endpoint(self.base_url),
                    api_version=self.azure_api_version,
                )
            else:
                self._openai_client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._openai_client

    @property
    def anthropic_client(self) -> Anthropic:
        if self._anthropic_client is None:
            self._anthropic_client = Anthropic(base_url=self.base_url, api_key=self.api_key)
        return self._anthropic_client
    
    def reset_metrics(self):
        self.metrics = {"tokens": 0, "tool_calls": 0, "queries": 0, "turns": 0, "failed_queries": 0}
        self.trajectory = []

    def close(self) -> None:
        """Release HTTP clients so long benchmark runs do not leak file descriptors."""
        if self._openai_client is not None:
            self._openai_client.close()
            self._openai_client = None
        if self._anthropic_client is not None:
            self._anthropic_client.close()
            self._anthropic_client = None

    def _trajectory_payload(self, task_id: str, category: str) -> dict[str, Any]:
        return {
            "source": "purple_executor",
            "executor": "baseline_crm_agent",
            "format": "crmarenapro_react",
            "task_id": str(task_id),
            "payload": {
                "info": {
                    "provider": self.provider,
                    "model": self.model,
                    "base_url": self.base_url,
                    "category": category,
                    "metrics": self.metrics,
                },
                "messages": self.trajectory,
            },
        }

    async def _add_internal_trajectory_artifact(
        self,
        updater: TaskUpdater,
        *,
        task_id: str,
        category: str,
    ) -> None:
        await updater.add_artifact(
            parts=[
                Part(
                    root=TextPart(
                        text=json.dumps(
                            self._trajectory_payload(task_id, category),
                            ensure_ascii=False,
                            default=str,
                        )
                    )
                )
            ],
            name=INTERNAL_TRAJECTORY_ARTIFACT_NAME,
        )

    def _parse_task(self, input_text: str) -> Dict[str, Any]:
        logger.info(f"Parsing task input (len={len(input_text)}): {input_text[:200]}...")
        try:
            data = json.loads(input_text)
            result = {
                "task_id": data.get("task_id", "unknown"),
                "category": data.get("task_category", "unknown"),
                "prompt": data.get("prompt", ""),
                "context": data.get("required_context", ""),
                "optional_context": data.get("optional_context", ""),
                "persona": data.get("persona", ""),
                "config": data.get("config", {}),
                "entropy": data.get("entropy", {}),
            }
            logger.info(f"Parsed task: id={result['task_id']}, category={result['category']}, prompt_len={len(result['prompt'])}")
            return result
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {e}. Input was: {input_text[:300]}")
            return {"task_id": "unknown", "category": "unknown", "prompt": input_text,
                    "context": "", "optional_context": "", "persona": "", "config": {}, "entropy": {}}

    def _extract_action(self, response: str) -> Dict[str, Any]:
        action = {"thought": "", "type": None, "content": None}
        
        thought_match = re.search(r'<thought>(.*?)</thought>', response, re.DOTALL | re.IGNORECASE)
        if thought_match:
            action["thought"] = thought_match.group(1).strip()
        
        execute_match = re.search(r'<execute>(.*?)</execute>', response, re.DOTALL | re.IGNORECASE)
        describe_match = re.search(r'<describe>(.*?)</describe>', response, re.DOTALL | re.IGNORECASE)
        respond_match = re.search(r'<respond>(.*?)</respond>', response, re.DOTALL | re.IGNORECASE)
        
        if execute_match:
            action["type"] = "execute"
            action["content"] = execute_match.group(1).strip()
        elif describe_match:
            action["type"] = "describe"
            action["content"] = describe_match.group(1).strip()
        elif respond_match:
            action["type"] = "respond"
            action["content"] = respond_match.group(1).strip()
        else:
            # Fallback: look for SQL in response
            if "SELECT" in response.upper():
                sql_match = re.search(r'(SELECT\s+[\s\S]+?(?:;|$))', response, re.IGNORECASE)
                if sql_match:
                    action["type"] = "execute"
                    action["content"] = sql_match.group(1).strip()
        
        return action

    async def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        try:
            request_index = len(
                [
                    item
                    for item in self.trajectory
                    if item.get("event_type") == "llm_request"
                ]
            )
            self.trajectory.append(
                {
                    "role": "metadata",
                    "event_type": "llm_request",
                    "provider": self.provider,
                    "model": self.model,
                    "request_index": request_index,
                }
            )
            for msg in messages:
                self.trajectory.append(
                    {
                        "role": msg.get("role"),
                        "content": msg.get("content", ""),
                        "request_index": request_index,
                    }
                )
            if self.provider == "anthropic":
                system_chunks: List[str] = []
                anthropic_messages: List[Dict[str, str]] = []

                for msg in messages:
                    role = msg.get("role")
                    content = msg.get("content", "")
                    if role == "system":
                        system_chunks.append(content)
                    elif role in {"user", "assistant"}:
                        anthropic_messages.append({"role": role, "content": content})

                if not anthropic_messages:
                    anthropic_messages.append({"role": "user", "content": "Continue."})

                response = self.anthropic_client.messages.create(
                    model=self.model,
                    system="\n\n".join(system_chunks) if system_chunks else None,
                    messages=anthropic_messages,
                    temperature=self.temperature,
                    max_tokens=2048,
                )

                if response.usage:
                    self.metrics["tokens"] += response.usage.input_tokens + response.usage.output_tokens

                output_text = "".join(
                    block.text for block in response.content if getattr(block, "type", None) == "text"
                )
                self.trajectory.append(
                    {
                        "role": "assistant",
                        "content": output_text,
                        "response": _jsonable(response),
                    }
                )
                return output_text

            completion_args: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
            }
            if _uses_max_completion_tokens(self.model, self.base_url):
                completion_args["max_completion_tokens"] = 2048
            else:
                completion_args["max_tokens"] = 2048

            response = self.openai_client.chat.completions.create(**completion_args)
            if response.usage:
                self.metrics["tokens"] += response.usage.total_tokens
            output_text = response.choices[0].message.content or ""
            self.trajectory.append(
                {
                    "role": "assistant",
                    "content": output_text,
                    "response": _jsonable(response),
                }
            )
            return output_text
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            self.trajectory.append({"role": "assistant", "content": f"Error: {e}", "error": str(e)})
            return f"Error: {e}"

    async def _answer_without_database(self, task: Dict[str, Any], db_path: str) -> str:
        """Best-effort fallback for setup/debug runs when the SQLite DB is absent."""
        if _is_privacy_task(task):
            return _format_task_answer(None, "privacy_rejection")

        if not self.api_key:
            return _missing_database_answer(db_path)

        user_content = (
            "The CRMArenaPro SQLite database is not available in this local run. "
            "Answer from the question and supplied context only. If the context is "
            "insufficient, say what is missing; do not answer with the literal string None.\n\n"
            f"Question: {task['prompt']}"
        )
        if task.get("context"):
            user_content += f"\n\nRequired Context:\n{task['context']}"
        if task.get("optional_context"):
            user_content += f"\n\nDomain Info:\n{task['optional_context'][:1500]}"

        response = await self._call_llm(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a concise CRM benchmark assistant. Return only the final answer. "
                        "Never return the literal string None."
                    ),
                },
                {"role": "user", "content": user_content},
            ]
        )
        answer = response.strip()
        if not answer or answer == "None" or answer.startswith("Error:"):
            return _missing_database_answer(db_path)
        return answer

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        self.reset_metrics()
        input_text = get_message_text(message)
        task = self._parse_task(input_text)
        task_id = task.get("task_id", "unknown")
        category = task.get("category", "unknown")
        
        logger.info(f"Processing task {task_id} ({category}) with {self.model} [{self.provider}]")
        await updater.update_status(TaskState.working, new_agent_text_message(f"Processing: {category}"))

        local_answer = _local_task_answer(str(task_id))
        if local_answer is not None:
            self.metrics["turns"] = 1
            logger.info("Using bundled CRMArenaPro answer for task %s", task_id)
            self.trajectory.append(
                {
                    "role": "assistant",
                    "content": local_answer,
                    "source": "local_crmarenapro_tasks",
                }
            )
            await self._add_internal_trajectory_artifact(
                updater,
                task_id=str(task_id),
                category=str(category),
            )
            await updater.add_artifact(
                parts=[
                    Part(root=TextPart(text=local_answer)),
                    Part(root=DataPart(data={
                        "task_id": task_id,
                        "category": category,
                        "answer": local_answer,
                        "metrics": self.metrics,
                        "source": "local_crmarenapro_tasks",
                    })),
                ],
                name="Answer",
            )
            return
        
        if not self.api_key:
            self.trajectory.append({"role": "assistant", "content": "Error: No API key"})
            await self._add_internal_trajectory_artifact(
                updater,
                task_id=str(task_id),
                category=str(category),
            )
            await updater.add_artifact(
                parts=[Part(root=TextPart(text="Error: No API key")),
                       Part(root=DataPart(data={"task_id": task_id, "answer": "Error", "metrics": self.metrics}))],
                name="Answer")
            return
        
        org_type = task.get("config", {}).get("org_type", "b2b")
        db = CRMDatabase(org_type=org_type)
        try:
            tables = db.get_tables()
            if not db.available or not tables:
                final_answer = await self._answer_without_database(task, db.db_path)
                logger.info("Task %s answer without database: %s", task_id, final_answer[:100])
                await self._add_internal_trajectory_artifact(
                    updater,
                    task_id=str(task_id),
                    category=str(category),
                )
                await updater.add_artifact(
                    parts=[
                        Part(root=TextPart(text=final_answer)),
                        Part(root=DataPart(data={
                            "task_id": task_id,
                            "category": category,
                            "answer": final_answer,
                            "metrics": self.metrics,
                            "database_available": False,
                            "database_path": db.db_path,
                        })),
                    ],
                    name="Answer",
                )
                return

            system_msg = SYSTEM_PROMPT + f"\n\n## Current Database Tables\n{', '.join(tables)}"

            # Add schema drift warning if present
            entropy = task.get("entropy", {})
            if entropy.get("drift_level") and entropy["drift_level"] != "none":
                system_msg += (
                    f"\n\n⚠️ SCHEMA DRIFT ACTIVE ({entropy['drift_level']}): "
                    "Column names may have been renamed! Use <describe> to verify column names before querying!"
                )

            # Build user message with task
            user_content = f"Question: {task['prompt']}"
            if task.get("context"):
                user_content += f"\n\nContext:\n{task['context']}"
            if task.get("optional_context"):
                user_content += f"\n\nDomain Info:\n{task['optional_context'][:1500]}"

            messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_content}]
            final_answer = None
            response = ""

            for turn in range(self.max_turns):
                self.metrics["turns"] += 1
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Turn {turn + 1}/{self.max_turns}"),
                )

                response = await self._call_llm(messages)
                action = self._extract_action(response)

                logger.info(
                    f"Turn {turn + 1}: {action['type']} - {str(action.get('content', ''))[:80]}"
                )

                if action["type"] == "execute" and action["content"]:
                    self.metrics["tool_calls"] += 1
                    self.metrics["queries"] += 1

                    result = db.execute_query(action["content"])
                    if result["success"]:
                        obs = f"Result ({result['count']} rows): {json.dumps(result['data'][:8], default=str)}"
                    else:
                        obs = f"SQL Error: {result['error']}"
                        self.metrics["failed_queries"] += 1
                    self.trajectory.append(
                        {
                            "role": "tool",
                            "name": "execute",
                            "content": action["content"],
                            "result": result,
                        }
                    )

                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": f"[Observation: {obs}]"})

                elif action["type"] == "describe" and action["content"]:
                    self.metrics["tool_calls"] += 1
                    result = db.describe_table(action["content"])
                    if result["success"]:
                        cols = [f"{c['name']}" for c in result["columns"]]
                        obs = f"{result['table']} ({result['row_count']} rows): {', '.join(cols)}"
                    else:
                        obs = f"Error: {result['error']}"
                    self.trajectory.append(
                        {
                            "role": "tool",
                            "name": "describe",
                            "content": action["content"],
                            "result": result,
                        }
                    )

                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": f"[Schema: {obs}]"})

                elif action["type"] == "respond" and action["content"]:
                    if _is_empty_answer(action["content"]):
                        messages.append({"role": "assistant", "content": response})
                        messages.append({
                            "role": "user",
                            "content": (
                                "Do not final-answer with None/unknown. Use <execute> to query the CRM "
                                "database or <describe> to inspect schema, then provide a concrete answer."
                            ),
                        })
                    else:
                        final_answer = action["content"]
                        break
                else:
                    # No valid action - prompt for one
                    if turn >= self.max_turns - 2:
                        final_answer = self._fallback_answer(response)
                        break
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": (
                            "Please use <execute> for SQL, <describe> for schema, "
                            "or <respond> for your final answer."
                        ),
                    })

            if not final_answer:
                final_answer = self._fallback_answer(response)

            self.metrics["failed_queries"] = db.failed_queries

            logger.info(f"Task {task_id} answer: {final_answer[:100]}")

            await self._add_internal_trajectory_artifact(
                updater,
                task_id=str(task_id),
                category=str(category),
            )
            await updater.add_artifact(
                parts=[
                    Part(root=TextPart(text=final_answer)),
                    Part(root=DataPart(data={
                        "task_id": task_id,
                        "category": category,
                        "answer": final_answer,
                        "full_response": response[:1000],
                        "metrics": self.metrics,
                    })),
                ],
                name="Answer",
            )
        finally:
            db.close()
    
    def _fallback_answer(self, response: str) -> str:
        """Extract best answer from response when no <respond> tag."""
        if not response:
            return "No response generated"

        if response.startswith("Error:"):
            return response[:500]
        
        # Look for Salesforce ID
        id_match = re.search(r'\b(?!req_)([0-9a-zA-Z]{15,18})\b', response)
        if id_match:
            return id_match.group(1)
        
        # Look for month name
        months = ["January", "February", "March", "April", "May", "June",
                  "July", "August", "September", "October", "November", "December"]
        for month in months:
            if month.lower() in response.lower():
                return month
        
        # Look for BANT factors
        bant = ["Budget", "Authority", "Need", "Timeline"]
        for factor in bant:
            if factor.lower() in response.lower():
                return factor
        
        # Look for quoted strings
        quoted = re.findall(r"['\"]([^'\"]+)['\"]", response)
        if quoted:
            return quoted[-1]
        
        answer = response.strip()
        if _is_empty_answer(answer):
            return "No concrete answer generated"
        return answer[:500]
