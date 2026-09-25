import argparse
import copy
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ENTERPRISE_LAB_PATH = (
    Path.home() / "program" / "tools" / "EnterpriseLab" / "Data" / "task_tool_names.json"
)
DEFAULT_ENTERPRISE_OPS_GYM_PATH = (
    Path.home()
    / "program"
    / "tools"
    / "EnterpriseOps-Gym"
    / "results"
    / "task_gym_tool_combinations.json"
)
DEFAULT_THE_AGENT_COMPANY_PATH = Path("/data/user/TheAgentCompany/task_function_combinations.json")
DEFAULT_TOUCAN_PATH = (
    Path.home() / "program" / "tools" / "Toucan" / "enterprise_trajectories_multi_turn_e5.jsonl"
)
DEFAULT_OUTPUT_JSONL = ROOT / "trajectories" / "imported_benchmark_seeds.jsonl"
DEFAULT_PILOT_OUTPUT_JSONL = ROOT / "trajectories" / "enterpriseops_gym_pilot_seeds.jsonl"
TOUCAN_CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1}

NORMALIZATION_VERSION = "v1"
PILOT_DOMAIN_TARGETS = {
    "itsm": 3,
    "csm": 3,
    "email": 2,
    "drive": 2,
    "teams": 2,
}

READ_PREFIXES = ("get_", "list_", "find_", "search_", "retrieve_", "browser", "web_read")
CREATE_PREFIXES = ("create_", "add_", "register_", "enlist_", "copy_", "fork_", "upload_")
UPDATE_PREFIXES = (
    "update_",
    "patch_",
    "modify_",
    "delete_",
    "remove_",
    "archive_",
    "cancel_",
    "soft_delete_",
    "link_",
    "resolve_",
    "publish_",
    "verify_",
    "transfer_",
)
DELIVER_PREFIXES = ("send_", "create_draft", "create_call")

STOP_TOOL_TOKENS = {
    "add",
    "archive",
    "batch",
    "cancel",
    "copy",
    "create",
    "delete",
    "disable",
    "enable",
    "enlist",
    "find",
    "finish",
    "fork",
    "get",
    "given",
    "link",
    "list",
    "modify",
    "new",
    "patch",
    "publish",
    "register",
    "remove",
    "resolve",
    "retrieve",
    "search",
    "send",
    "soft",
    "think",
    "transfer",
    "update",
    "using",
    "web",
}

PRECONDITION_MARKERS = (
    "already",
    "currently",
    "existing",
    "needs",
    "not yet",
    "still",
    "waiting",
    "without",
)
BLOCKER_MARKERS = (
    "if ",
    "if it",
    "must",
    "needs to",
    "require",
    "should",
    "ensure",
    "only if",
)
TARGET_STATE_MARKERS = (
    "active",
    "approved",
    "archived",
    "assigned",
    "canceled",
    "cancelled",
    "confirmed",
    "critical",
    "enabled",
    "high priority",
    "in progress",
    "in_progress",
    "in use",
    "published",
    "ready",
    "resolved",
    "under maintenance",
    "work in progress",
)
SEQUENCING_MARKERS = (
    "first",
    "then",
    "next",
    "after",
    "before",
    "once",
    "finally",
    "while",
    "begin by",
    "immediately",
)
SLA_MARKERS = (
    "sla",
    "urgent",
    "priority",
    "critical",
    "24/7",
    "response",
    "deadline",
    "business impact",
    "urgency",
    "effective date",
)

TOOL_CATEGORY_RULES = {
    "communication": (
        "message",
        "notification",
        "chat",
        "channel",
        "draft",
        "forwarding",
        "pop_settings",
        "imap_settings",
        "send_as",
        "email",
        "reply",
    ),
    "repository": (
        "repository",
        "branch",
        "merge_request",
        "file_contents",
        "push_files",
    ),
    "document_management": (
        "file",
        "folder",
        "drive",
        "comment",
        "reply",
        "revision",
        "permission",
        "accessproposal",
        "label",
    ),
    "customer_support": (
        "account",
        "contact",
        "case",
        "contract",
        "entitlement",
        "installed_product",
        "knowledge",
        "portal_user",
        "product",
    ),
    "itsm": (
        "incident",
        "configuration_item",
        "configuration_items",
        "service",
        "location",
        "sla",
        "group",
    ),
    "project_management": (
        "project",
        "issue",
        "module",
        "cycle",
        "worklog",
        "state",
        "label",
    ),
    "collaboration_workspace": (
        "team",
        "channel",
        "virtual_event",
        "webinar",
        "townhall",
        "teamwork_tag",
        "tabs",
        "call",
    ),
    "browser_workspace": (
        "browser",
        "web_read",
    ),
    "computation": (
        "execute_bash",
        "execute_ipython_cell",
        "str_replace_editor",
    ),
}

SOURCE_TO_FAMILY = {
    "EnterpriseLab": "enterprise_lab",
    "EnterpriseOps-Gym": "enterprise_ops_gym",
    "TheAgentCompany": "the_agent_company",
    "Toucan": "toucan",
}

DOMAIN_SURFACES = {
    "csm": ["crm", "support_case_management", "knowledge_base"],
    "itsm": ["service_management", "cmdb", "incident_response"],
    "email": ["mailbox", "message_routing", "labeling"],
    "drive": ["cloud_documents", "sharing_permissions", "document_comments"],
    "teams": ["team_workspace", "channel_messaging", "virtual_events"],
    "repository_ops": ["repository", "source_code", "merge_workflow"],
    "project_management": ["project_tracker", "workflow_board", "team_updates"],
    "document_ops": ["cloud_storage", "document_lookup", "message_updates"],
    "mixed_enterprise_ops": ["enterprise_systems"],
    "workspace_coordination": ["browser", "chat", "local_filesystem"],
}

DOMAIN_ARTIFACT_KEYWORDS = {
    "csm": [
        "account",
        "contact",
        "contract",
        "entitlement",
        "case",
        "installed product",
        "knowledge article",
        "sla",
    ],
    "itsm": [
        "incident",
        "configuration item",
        "service offering",
        "location",
        "knowledge article",
        "sla",
        "group",
    ],
    "email": [
        "message",
        "thread",
        "label",
        "filter",
        "draft",
        "forwarding address",
        "mailbox setting",
    ],
    "drive": [
        "file",
        "document",
        "permission",
        "comment",
        "reply",
        "metadata",
        "shared drive",
    ],
    "teams": [
        "team",
        "channel",
        "member",
        "webinar",
        "townhall",
        "chat",
        "tag",
        "call",
        "tab",
    ],
    "repository_ops": [
        "repository",
        "file",
        "issue",
        "merge request",
        "branch",
    ],
    "project_management": [
        "project",
        "issue",
        "module",
        "cycle",
        "state",
        "label",
        "worklog",
    ],
    "document_ops": [
        "file",
        "folder",
        "message",
        "company",
        "invoice",
        "product",
    ],
    "workspace_coordination": [
        "chat thread",
        "workspace file",
        "browser session",
        "report",
    ],
}

AGENT_BLUEPRINTS = {
    "csm": [
        (
            "customer_contact",
            "Customer Contact",
            "stakeholder",
            [
                "Describe the customer impact and validate affected products or contacts.",
                "Receive progress updates once account or case actions are complete.",
            ],
        ),
        (
            "csm_operations_agent",
            "CSM Operations Agent",
            "operator",
            [
                "Create or update accounts, contracts, entitlements, and cases.",
                "Coordinate the main CRM-side workflow changes requested by the task.",
            ],
        ),
        (
            "support_case_owner",
            "Support Case Owner",
            "specialist",
            [
                "Own case progression, SLA linkage, and knowledge alignment.",
                "Drive the requested support state after the record setup is complete.",
            ],
        ),
    ],
    "itsm": [
        (
            "service_requester",
            "Service Requester",
            "stakeholder",
            [
                "Report the operational problem and validate the affected service or asset.",
                "Confirm the urgency or impact conditions that should guide the workflow.",
            ],
        ),
        (
            "service_desk_agent",
            "Service Desk Agent",
            "operator",
            [
                "Update incidents, configuration items, or service records in the workflow system.",
                "Coordinate notifications and operational record changes needed to stabilize the task.",
            ],
        ),
        (
            "incident_manager",
            "Incident Manager",
            "reviewer",
            [
                "Own escalation, SLA alignment, or post-update review of the incident path.",
                "Ensure the requested incident state is consistent with the broader support process.",
            ],
        ),
    ],
    "email": [
        (
            "mailbox_owner",
            "Mailbox Owner",
            "stakeholder",
            [
                "Define the desired mailbox organization and delivery rules.",
                "Confirm which messages, labels, or forwarding settings should be changed.",
            ],
        ),
        (
            "messaging_admin",
            "Messaging Administrator",
            "operator",
            [
                "Apply mailbox configuration changes, labels, drafts, filters, and routing updates.",
                "Carry the main execution workload across the email task.",
            ],
        ),
        (
            "review_owner",
            "Review Owner",
            "reviewer",
            [
                "Validate that mailbox settings and message classifications satisfy the requested policy.",
                "Catch configuration choices that may need a second pass before rollout.",
            ],
        ),
    ],
    "drive": [
        (
            "document_requester",
            "Document Requester",
            "stakeholder",
            [
                "Specify the target file, audience, and access policy for the document workflow.",
                "Confirm the document-level business purpose after updates are complete.",
            ],
        ),
        (
            "document_operator",
            "Document Operations Agent",
            "operator",
            [
                "Create or update files, comments, metadata, and permissions.",
                "Apply the durable document changes requested by the task.",
            ],
        ),
        (
            "access_reviewer",
            "Access Reviewer",
            "reviewer",
            [
                "Check whether permission and metadata changes match the requested governance policy.",
                "Review external-sharing or audit-facing changes before sign-off.",
            ],
        ),
    ],
    "teams": [
        (
            "team_owner",
            "Team Owner",
            "stakeholder",
            [
                "Define the workspace restructure, target members, and collaboration outcomes.",
                "Confirm the visibility and purpose of the created collaboration space.",
            ],
        ),
        (
            "collaboration_operator",
            "Collaboration Operations Agent",
            "operator",
            [
                "Create or update teams, channels, tabs, tags, chats, or virtual events.",
                "Carry out the main collaboration-system actions in the task.",
            ],
        ),
        (
            "participant_contact",
            "Participant Contact",
            "member_representative",
            [
                "Represent the downstream members, co-organizers, or attendees affected by the workspace change.",
                "Receive rollout messages or event updates after the system-side changes land.",
            ],
        ),
    ],
    "repository_ops": [
        (
            "developer_requester",
            "Developer Requester",
            "stakeholder",
            [
                "Specify the repository or file-level information needed for the task.",
                "Clarify which code artifact or repository action matters to the request.",
            ],
        ),
        (
            "repository_operator",
            "Repository Operator",
            "operator",
            [
                "Search repositories, inspect files, and carry out source-control actions.",
                "Execute the primary repository workflow requested by the task.",
            ],
        ),
        (
            "team_contact",
            "Team Contact",
            "collaborator",
            [
                "Receive notifications or downstream repository updates once the work is complete.",
                "Represent the collaboration side of the repository workflow.",
            ],
        ),
    ],
    "project_management": [
        (
            "project_requester",
            "Project Requester",
            "stakeholder",
            [
                "Define the project, cycle, or issue tracking outcome required by the task.",
                "Clarify the reporting or workflow view needed after the system changes land.",
            ],
        ),
        (
            "project_operator",
            "Project Operations Agent",
            "operator",
            [
                "Work with project, issue, module, and cycle records.",
                "Execute the workflow-management side of the benchmark task.",
            ],
        ),
        (
            "team_contact",
            "Team Contact",
            "collaborator",
            [
                "Receive summaries or notifications once project-state updates are finished.",
                "Represent the collaboration endpoint affected by the project changes.",
            ],
        ),
    ],
    "document_ops": [
        (
            "business_requester",
            "Business Requester",
            "stakeholder",
            [
                "Specify the file, folder, or business record outcome needed from the task.",
                "Confirm that the operational artifact is available after execution.",
            ],
        ),
        (
            "document_operator",
            "Document Operator",
            "operator",
            [
                "Carry out file, folder, or record updates across the operational workspace.",
                "Handle the durable artifact changes implied by the task request.",
            ],
        ),
        (
            "team_contact",
            "Team Contact",
            "collaborator",
            [
                "Receive the operational update after the system work completes.",
                "Represent the downstream consumer of the changed artifact.",
            ],
        ),
    ],
    "mixed_enterprise_ops": [
        (
            "enterprise_requester",
            "Enterprise Requester",
            "stakeholder",
            [
                "Frame the desired enterprise outcome and provide the original ask.",
                "Confirm whether the resulting record or update satisfies the request.",
            ],
        ),
        (
            "enterprise_operator",
            "Enterprise Operations Agent",
            "operator",
            [
                "Execute the primary tool-backed workflow represented by the source task.",
                "Coordinate the main system changes across the involved surfaces.",
            ],
        ),
        (
            "review_contact",
            "Review Contact",
            "reviewer",
            [
                "Review the changed state before it is treated as complete.",
                "Represent the secondary human check that makes the workflow multi-agent.",
            ],
        ),
    ],
    "workspace_coordination": [
        (
            "task_requester",
            "Task Requester",
            "stakeholder",
            [
                "Set the end goal and supply the raw office-work request.",
                "Confirm that the final deliverable reaches the right destination.",
            ],
        ),
        (
            "workspace_operator",
            "Workspace Operator",
            "operator",
            [
                "Coordinate browser, chat, filesystem, or computation actions across the task.",
                "Carry the primary long-horizon work needed to complete the request.",
            ],
        ),
        (
            "response_contact",
            "Response Contact",
            "collaborator",
            [
                "Represent the people contacted during collection, verification, or handoff steps.",
                "Anchor the downstream communication side of the workflow.",
            ],
        ),
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Normalize external benchmark task/tool inventories into imported EWM seed records.",
    )
    parser.add_argument("--enterprise-lab-path", type=Path, default=DEFAULT_ENTERPRISE_LAB_PATH)
    parser.add_argument(
        "--enterprise-ops-gym-path",
        type=Path,
        default=DEFAULT_ENTERPRISE_OPS_GYM_PATH,
    )
    parser.add_argument(
        "--the-agent-company-path",
        type=Path,
        default=DEFAULT_THE_AGENT_COMPANY_PATH,
    )
    parser.add_argument("--toucan-path", type=Path, default=DEFAULT_TOUCAN_PATH)
    parser.add_argument(
        "--toucan-min-confidence",
        choices=("high", "medium", "low"),
        default="high",
        help="Lowest TOUCAN enterprise-label confidence bucket to include.",
    )
    parser.add_argument(
        "--skip-toucan",
        action="store_true",
        help="Skip TOUCAN ingestion entirely (for fast EnterpriseLab/EnterpriseOps-Gym-only runs).",
    )
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--pilot-output-jsonl", type=Path, default=DEFAULT_PILOT_OUTPUT_JSONL)
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_jsonl(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def unique_ordered(values):
    seen = set()
    ordered = []
    for value in values:
        if not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalize_task_text(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    return value.strip()


def slugify(value: str) -> str:
    lowered = value.lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    lowered = re.sub(r"_+", "_", lowered)
    return lowered.strip("_")


def split_sentences(text: str) -> list[str]:
    rough_parts = re.split(r"(?:\n{2,}|(?<=[.!?])\s+)", text)
    sentences = []
    for part in rough_parts:
        normalized = normalize_space(part)
        if not normalized:
            continue
        sentences.append(normalized)
    return sentences


def extract_marker_sentences(text: str, markers: tuple[str, ...], limit: int = 4) -> list[str]:
    sentences = split_sentences(text)
    matches = []
    for sentence in sentences:
        lowered = sentence.lower()
        if any(marker in lowered for marker in markers):
            matches.append(sentence)
        if len(matches) >= limit:
            break
    return unique_ordered(matches)


def extract_emails(text: str) -> list[str]:
    return unique_ordered(re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text))


def extract_quoted_strings(text: str) -> list[str]:
    values = []
    for value in re.findall(r"\"([^\"]{3,160})\"", text):
        values.append(normalize_space(value))
    for value in re.findall(r"'([^'\n]{3,160})'", text):
        candidate = normalize_space(value)
        if any(char in candidate for char in (" ", ".", "-", "_")):
            values.append(candidate)
    return unique_ordered(values)


def extract_identifiers(text: str) -> list[str]:
    patterns = [
        r"\b(?:INC|CS)-?\d{6,}\b",
        r"\b[A-Z]{2,}(?:-[A-Z0-9]+){1,}\b",
        r"\b[A-Z]{2,}_[A-Z0-9-]{3,}\b",
        r"\bUSER_\d+\b",
        r"\b[a-z]{2,}__[a-z0-9_]+__task_[a-z0-9_]+\b",
    ]
    matches = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, text))
    return unique_ordered(matches)


def extract_absolute_dates(text: str) -> list[str]:
    patterns = [
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2}\s+[A-Z][a-z]+\s+\d{4}\b",
        r"\b[A-Z][a-z]+\s+\d{1,2},\s+\d{4}\b",
        r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+\d{1,2}\s+[A-Z][a-z]+\s+\d{4}\b",
    ]
    values = []
    for pattern in patterns:
        values.extend(re.findall(pattern, text))
    return unique_ordered(values)


def extract_sequencing_constraints(text: str) -> list[str]:
    lowered = text.lower()
    matches = [marker for marker in SEQUENCING_MARKERS if marker in lowered]
    return unique_ordered(matches)


def singularize(value: str) -> str:
    replacements = {
        "accessproposals": "access_proposal",
        "comments": "comment",
        "contacts": "contact",
        "cycles": "cycle",
        "drafts": "draft",
        "files": "file",
        "filters": "filter",
        "forwarding_addresses": "forwarding_address",
        "groups": "group",
        "histories": "history",
        "incidents": "incident",
        "issues": "issue",
        "labels": "label",
        "locations": "location",
        "members": "member",
        "messages": "message",
        "modules": "module",
        "permissions": "permission",
        "products": "product",
        "projects": "project",
        "replies": "reply",
        "repositories": "repository",
        "services": "service",
        "states": "state",
        "teams": "team",
        "threads": "thread",
        "townhalls": "townhall",
        "users": "user",
        "webinars": "webinar",
    }
    if value in replacements:
        return replacements[value]
    if value.endswith("ies") and len(value) > 4:
        return value[:-3] + "y"
    if value.endswith("s") and not value.endswith("ss") and len(value) > 3:
        return value[:-1]
    return value


def canonicalize_artifact_name(value: str) -> str:
    replacements = {
        "auto_forwarding": "auto_forwarding_setting",
        "configuration_item": "configuration_item",
        "configuration_items": "configuration_item",
        "case_sla": "case_sla_link",
        "files_label": "file_label",
        "group_member": "group_membership",
        "group_membership": "group_membership",
        "imap_settings": "imap_setting",
        "installed_product": "installed_product",
        "knowledge": "knowledge_article",
        "knowledge_articles": "knowledge_article",
        "mail_system": "email_thread",
        "new_account": "account",
        "new_case": "case",
        "new_group_member": "group_membership",
        "new_service_offering": "service_offering",
        "new_user": "user",
        "permission": "permission",
        "pop_settings": "pop_setting",
        "send_as_aliases": "send_as_alias",
        "sla_definitions": "sla_definition",
        "teamwork_tag": "teamwork_tag",
        "teams_apps": "teams_app",
        "user_group": "user_group",
        "virtual_event_townhall": "townhall",
        "virtual_event_webinar": "webinar",
    }
    if value in replacements:
        return replacements[value]
    parts = [singularize(part) for part in value.split("_")]
    return "_".join(parts)


def artifact_from_tool(tool_name: str) -> str:
    tokens = [token for token in tool_name.lower().split("_") if token not in STOP_TOOL_TOKENS]
    if not tokens:
        return "workspace_object"
    return canonicalize_artifact_name("_".join(tokens))


def categorize_tools(tool_names: list[str]) -> list[str]:
    if not tool_names:
        return ["no_observed_tools"]

    categories = set()
    for tool_name in tool_names:
        normalized = tool_name.lower()
        for category, keywords in TOOL_CATEGORY_RULES.items():
            if any(keyword in normalized for keyword in keywords):
                categories.add(category)
    if not categories:
        categories.add("generic_operations")
    return sorted(categories)


def infer_enterprise_lab_domain(tool_categories: list[str]) -> str:
    if "repository" in tool_categories:
        return "repository_ops"
    if "project_management" in tool_categories:
        return "project_management"
    if "document_management" in tool_categories:
        return "document_ops"
    return "mixed_enterprise_ops"


def infer_environment(
    source_dataset: str,
    source_domain: str | None,
    tool_categories: list[str],
) -> dict:
    domain = source_domain or ""
    if source_dataset == "EnterpriseLab":
        domain = infer_enterprise_lab_domain(tool_categories)
    elif source_dataset == "TheAgentCompany":
        domain = "workspace_coordination"

    surfaces = DOMAIN_SURFACES.get(domain, [])
    if not surfaces:
        surfaces = [category for category in tool_categories if category != "no_observed_tools"]
    if not surfaces:
        surfaces = ["generic_workspace"]

    return {
        "benchmark_family": SOURCE_TO_FAMILY[source_dataset],
        "domain": domain or "mixed_enterprise_ops",
        "surfaces": unique_ordered(surfaces),
    }


def build_agents_for_domain(domain: str) -> list[dict]:
    blueprints = AGENT_BLUEPRINTS.get(domain, AGENT_BLUEPRINTS["mixed_enterprise_ops"])
    agents = []
    for agent_id, display_name, role, responsibilities in blueprints:
        agents.append(
            {
                "agent_id": agent_id,
                "display_name": display_name,
                "role": role,
                "responsibilities": responsibilities,
            }
        )
    return agents


def extract_candidate_roles(candidate_agents: list[dict]) -> list[str]:
    values = []
    for agent in candidate_agents:
        values.append(agent["display_name"])
        values.append(agent["role"])
    return unique_ordered(values)


def infer_keyword_artifacts(task_text: str, domain: str) -> list[str]:
    lowered = task_text.lower()
    artifacts = []
    for raw_keyword in DOMAIN_ARTIFACT_KEYWORDS.get(domain, []):
        if raw_keyword in lowered:
            artifacts.append(canonicalize_artifact_name(raw_keyword.replace(" ", "_")))
    return unique_ordered(artifacts)


def infer_communication_channels(task_text: str, tool_names: list[str], tool_categories: list[str]) -> list[str]:
    lowered = task_text.lower()
    channels = []
    if "communication" in tool_categories:
        channels.append("direct_or_channel_message")
    if "collaboration_workspace" in tool_categories:
        channels.append("team_workspace")
    if any(tool.startswith("create_draft") or "forwarding" in tool for tool in tool_names):
        channels.append("email")
    if "phone" in lowered:
        channels.append("phone")
    if "webinar" in lowered or "townhall" in lowered:
        channels.append("virtual_event")
    if "comment" in lowered:
        channels.append("document_comment")
    if not channels:
        channels.append("tool_only_workflow")
    return unique_ordered(channels)


def build_artifact_hooks(task_text: str, tool_names: list[str], domain: str) -> dict:
    read = []
    create = []
    update = []
    deliver = []
    durable_objects = []

    for tool_name in tool_names:
        artifact = artifact_from_tool(tool_name)
        durable_objects.append(artifact)

        lowered = tool_name.lower()
        if lowered.startswith(READ_PREFIXES):
            read.append(artifact)
        if lowered.startswith(CREATE_PREFIXES):
            create.append(artifact)
        if lowered.startswith(UPDATE_PREFIXES):
            update.append(artifact)
        if lowered.startswith(DELIVER_PREFIXES) or artifact in {"comment", "draft", "message", "notification"}:
            deliver.append(artifact)

    keyword_artifacts = infer_keyword_artifacts(task_text, domain)
    durable_objects.extend(keyword_artifacts)

    return {
        "read": unique_ordered(read),
        "create": unique_ordered(create),
        "update": unique_ordered(update),
        "deliver": unique_ordered(deliver),
        "durable_objects": unique_ordered(durable_objects),
    }


def derive_transitions(tool_names: list[str]) -> list[str]:
    transitions = []
    for tool_name in tool_names:
        artifact = artifact_from_tool(tool_name).replace("_", " ")
        lowered = tool_name.lower()
        if lowered.startswith(CREATE_PREFIXES):
            transitions.append(f"Create or register {artifact}.")
        elif lowered.startswith(UPDATE_PREFIXES):
            transitions.append(f"Update, link, or retire {artifact}.")
        elif lowered.startswith(DELIVER_PREFIXES):
            transitions.append(f"Deliver or communicate {artifact}.")
        elif lowered.startswith(READ_PREFIXES):
            transitions.append(f"Inspect {artifact} state before acting.")
    return unique_ordered(transitions)[:6]


def build_target_states(task_text: str, tool_names: list[str]) -> list[str]:
    target_states = extract_marker_sentences(task_text, TARGET_STATE_MARKERS, limit=5)
    if target_states:
        return target_states

    fallback = []
    for transition in derive_transitions(tool_names)[:3]:
        fallback.append(f"Requested end state should reflect: {transition}")
    return unique_ordered(fallback)


def build_relational_hooks(task_text: str) -> dict:
    ownership = extract_marker_sentences(
        task_text,
        ("owner", "owned", "ownership", "primary contact", "reporting contact", "assigned to"),
    )
    permissions = extract_marker_sentences(
        task_text,
        ("access", "permission", "writer", "reader", "commenting", "member", "owner"),
    )
    dependencies = extract_marker_sentences(
        task_text,
        ("linked", "attach", "associate", "depends", "dependency", "same group", "under same"),
    )
    approvals = extract_marker_sentences(
        task_text,
        ("approve", "approval", "audit", "review", "compliance", "warranty"),
    )
    return {
        "ownership": ownership,
        "permissions": permissions,
        "dependencies": dependencies,
        "approvals": approvals,
    }


def build_temporal_hooks(task_text: str) -> dict:
    return {
        "absolute_dates": extract_absolute_dates(task_text),
        "relative_constraints": extract_marker_sentences(
            task_text,
            ("today", "tomorrow", "yesterday", "next week", "next month", "before", "after", "immediately", "asap", "currently"),
        ),
        "sla_signals": extract_marker_sentences(task_text, SLA_MARKERS),
        "sequencing_constraints": extract_sequencing_constraints(task_text),
    }


def build_state_hooks(
    task_text: str,
    tool_names: list[str],
    environment: dict,
    candidate_agents: list[dict],
    tool_categories: list[str],
) -> dict:
    named_entities = unique_ordered(
        extract_quoted_strings(task_text)
        + extract_emails(task_text)
        + extract_identifiers(task_text)
    )
    stakeholders = unique_ordered(extract_emails(task_text) + [agent["display_name"] for agent in candidate_agents])
    context_facts = split_sentences(task_text)[:4]

    return {
        "identify": {
            "candidate_roles": extract_candidate_roles(candidate_agents),
            "stakeholders": stakeholders,
            "named_entities": named_entities,
        },
        "artifacts": build_artifact_hooks(task_text, tool_names, environment["domain"]),
        "process": {
            "preconditions": extract_marker_sentences(task_text, PRECONDITION_MARKERS),
            "transitions": derive_transitions(tool_names),
            "target_states": build_target_states(task_text, tool_names),
            "blocking_conditions": extract_marker_sentences(task_text, BLOCKER_MARKERS),
        },
        "relational": build_relational_hooks(task_text),
        "context": {
            "facts": context_facts,
            "communication_channels": infer_communication_channels(task_text, tool_names, tool_categories),
            "domains": unique_ordered([environment["domain"]] + tool_categories),
        },
        "temporal": build_temporal_hooks(task_text),
    }


def infer_coordination_pattern(environment: dict) -> str:
    domain = environment["domain"]
    if domain == "csm":
        return "coordinator_specialist_stakeholder"
    if domain == "itsm":
        return "requester_operator_escalation_owner"
    if domain in {"email", "drive"}:
        return "requester_operator_reviewer"
    if domain == "teams":
        return "coordinator_operator_member"
    if domain == "workspace_coordination":
        return "multi_party_collection"
    return "requester_operator"


def score_conversion_readiness(
    source_dataset: str,
    tool_names: list[str],
    state_hooks: dict,
    environment: dict,
) -> dict:
    score = 20
    rationale = []
    blocking_issues = []
    tool_count = len(tool_names)

    if source_dataset == "EnterpriseOps-Gym":
        score += 35
        rationale.append("Domain-specific enterprise tools expose durable workflow state.")
    elif source_dataset == "EnterpriseLab":
        score += 18
        rationale.append("Enterprise coverage is broad, but many tasks are shallow operator chains.")
    elif source_dataset == "Toucan":
        score += 12
        rationale.append(
            "Toucan multi-turn trajectories expose multi-server tool chains, "
            "though enterprise domain typing is heuristically inferred."
        )
    else:
        score += 10
        rationale.append("The task is long horizon, but the observed function layer is generic.")

    if 3 <= tool_count <= 8:
        score += 15
        rationale.append("Observed tool count is in the target range for pilot decomposition.")
    elif tool_count == 0:
        score -= 25
        blocking_issues.append("No observed tools were extracted from the source record.")
    elif tool_count <= 2:
        score += 5
        blocking_issues.append("Observed workflow may need additional lift because the tool chain is short.")
    else:
        score += 8
        rationale.append("Observed workflow is multi-step, though it may need pruning before templating.")

    durable_count = len(state_hooks["artifacts"]["durable_objects"])
    if durable_count >= 3:
        score += 10
        rationale.append("Multiple durable business objects are available to anchor state deltas.")

    temporal_signals = (
        len(state_hooks["temporal"]["absolute_dates"])
        + len(state_hooks["temporal"]["relative_constraints"])
        + len(state_hooks["temporal"]["sla_signals"])
    )
    if temporal_signals >= 2:
        score += 8
        rationale.append("Temporal or SLA constraints are explicit in the source task.")

    relational_signals = (
        len(state_hooks["relational"]["ownership"])
        + len(state_hooks["relational"]["permissions"])
        + len(state_hooks["relational"]["dependencies"])
        + len(state_hooks["relational"]["approvals"])
    )
    if relational_signals >= 2:
        score += 7
        rationale.append("Ownership, permission, or dependency signals support multi-agent lifting.")

    if source_dataset == "TheAgentCompany":
        blocking_issues.append("Low-level browser or shell functions still need semantic abstraction.")
    if source_dataset == "EnterpriseLab" and environment["domain"] == "mixed_enterprise_ops":
        blocking_issues.append("Mixed-domain task needs stronger domain typing before templating.")

    score = max(0, min(100, score))

    if source_dataset == "EnterpriseOps-Gym" and score >= 80:
        status = "ready_for_pilot"
    elif source_dataset == "TheAgentCompany":
        status = "needs_state_abstraction"
    elif score >= 60:
        status = "needs_agent_lift"
    else:
        status = "low_signal"

    return {
        "status": status,
        "score": score,
        "blocking_issues": unique_ordered(blocking_issues),
        "rationale": unique_ordered(rationale),
    }


def build_selection_tags(source_dataset: str, environment: dict, conversion_readiness: dict) -> list[str]:
    return [
        f"dataset:{slugify(source_dataset)}",
        f"domain:{environment['domain']}",
        f"status:{conversion_readiness['status']}",
    ]


def build_notes(source_dataset: str, source_record: dict, tool_names: list[str]) -> list[str]:
    notes = [
        "Normalized heuristically from a source benchmark task/tool inventory.",
        "Candidate agents are inferred lifting roles rather than explicit source annotations.",
    ]
    if not tool_names:
        notes.append("The source record did not contain any extracted tools or functions.")
    if source_dataset == "EnterpriseOps-Gym":
        notes.append("This source is the strongest candidate for the first state-explicit EWM import wave.")
    elif source_dataset == "EnterpriseLab":
        notes.append("This source contributes broad enterprise tool coverage, but many tasks still need agent lift.")
    elif source_dataset == "Toucan":
        confidence = (source_record or {}).get("confidence")
        prob = (source_record or {}).get("probability_enterprise")
        prob_text = f", probability={prob:.3f}" if isinstance(prob, (int, float)) else ""
        notes.append(
            f"Toucan multi-turn enterprise trajectory (confidence={confidence}{prob_text}); "
            "task domain is inferred heuristically from observed tool categories."
        )
    else:
        notes.append("This source is valuable for long-horizon coordination, but it still needs semantic action abstraction.")
    if source_record.get("source_path"):
        notes.append(f"Source path: {source_record['source_path']}")
    return notes


def validate_string_list(values, label: str):
    if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
        raise ValueError(f"{label} must be a list of strings")


def validate_seed(seed: dict):
    required_top_level = {
        "normalization_version",
        "seed_id",
        "source_dataset",
        "source_task_id",
        "task_text",
        "environment",
        "tool_names",
        "tool_categories",
        "state_hooks",
        "candidate_agents",
        "coordination_pattern",
        "conversion_readiness",
        "selection_tags",
        "source_record",
        "notes",
    }
    missing = required_top_level - set(seed)
    if missing:
        raise ValueError(f"seed missing keys: {sorted(missing)}")

    if not isinstance(seed["task_text"], str):
        raise ValueError("task_text must be a string")
    validate_string_list(seed["tool_names"], "tool_names")
    validate_string_list(seed["tool_categories"], "tool_categories")
    validate_string_list(seed["selection_tags"], "selection_tags")
    validate_string_list(seed["notes"], "notes")

    environment = seed["environment"]
    for key in ("benchmark_family", "domain"):
        if not isinstance(environment.get(key), str):
            raise ValueError(f"environment.{key} must be a string")
    validate_string_list(environment.get("surfaces"), "environment.surfaces")

    state_hooks = seed["state_hooks"]
    required_state_keys = {"identify", "artifacts", "process", "relational", "context", "temporal"}
    if set(state_hooks) != required_state_keys:
        raise ValueError("state_hooks must contain all six aspects")

    for aspect, fields in (
        ("identify", ("candidate_roles", "stakeholders", "named_entities")),
        ("artifacts", ("read", "create", "update", "deliver", "durable_objects")),
        ("process", ("preconditions", "transitions", "target_states", "blocking_conditions")),
        ("relational", ("ownership", "permissions", "dependencies", "approvals")),
        ("context", ("facts", "communication_channels", "domains")),
        ("temporal", ("absolute_dates", "relative_constraints", "sla_signals", "sequencing_constraints")),
    ):
        payload = state_hooks[aspect]
        for field in fields:
            validate_string_list(payload.get(field), f"state_hooks.{aspect}.{field}")

    agents = seed["candidate_agents"]
    if not isinstance(agents, list) or len(agents) < 2:
        raise ValueError("candidate_agents must contain at least two entries")
    for agent in agents:
        for key in ("agent_id", "display_name", "role"):
            if not isinstance(agent.get(key), str):
                raise ValueError(f"candidate_agents.{key} must be a string")
        validate_string_list(agent.get("responsibilities"), "candidate_agents.responsibilities")

    readiness = seed["conversion_readiness"]
    if readiness.get("status") not in {
        "ready_for_pilot",
        "needs_agent_lift",
        "needs_state_abstraction",
        "low_signal",
    }:
        raise ValueError("conversion_readiness.status has an invalid value")
    if not isinstance(readiness.get("score"), int) or not 0 <= readiness["score"] <= 100:
        raise ValueError("conversion_readiness.score must be an integer between 0 and 100")
    validate_string_list(readiness.get("blocking_issues"), "conversion_readiness.blocking_issues")
    validate_string_list(readiness.get("rationale"), "conversion_readiness.rationale")

    source_record = seed["source_record"]
    if not isinstance(source_record.get("source_index"), int) or source_record["source_index"] < 0:
        raise ValueError("source_record.source_index must be a non-negative integer")


def build_seed(
    *,
    source_dataset: str,
    source_task_id: str,
    task_text: str,
    tool_names: list[str],
    source_record: dict,
    source_domain: str | None = None,
) -> dict:
    normalized_tools = unique_ordered(tool_names)
    tool_categories = categorize_tools(normalized_tools)
    environment = infer_environment(source_dataset, source_domain, tool_categories)
    candidate_agents = build_agents_for_domain(environment["domain"])
    state_hooks = build_state_hooks(task_text, normalized_tools, environment, candidate_agents, tool_categories)
    readiness = score_conversion_readiness(source_dataset, normalized_tools, state_hooks, environment)

    seed = {
        "normalization_version": NORMALIZATION_VERSION,
        "seed_id": f"import.{slugify(source_dataset)}.{slugify(source_task_id)}",
        "source_dataset": source_dataset,
        "source_task_id": source_task_id,
        "task_text": task_text,
        "environment": environment,
        "tool_names": normalized_tools,
        "tool_categories": tool_categories,
        "state_hooks": state_hooks,
        "candidate_agents": candidate_agents,
        "coordination_pattern": infer_coordination_pattern(environment),
        "conversion_readiness": readiness,
        "selection_tags": build_selection_tags(source_dataset, environment, readiness),
        "source_record": source_record,
        "notes": build_notes(source_dataset, source_record, normalized_tools),
    }
    validate_seed(seed)
    return seed


def normalize_enterprise_lab(path: Path) -> list[dict]:
    payload = load_json(path)
    records = []
    for index, item in enumerate(payload):
        task_text = normalize_task_text(item["task"])
        tool_names = item.get("tool_names") or []
        source_task_id = f"enterprise_lab_{index + 1:04d}"
        source_record = {
            "source_index": index,
            "source_path": str(path),
            "tool_count_observed": len(tool_names),
        }
        records.append(
            build_seed(
                source_dataset="EnterpriseLab",
                source_task_id=source_task_id,
                task_text=task_text,
                tool_names=tool_names,
                source_record=source_record,
            )
        )
    return records


def normalize_enterprise_ops_gym(path: Path) -> list[dict]:
    payload = load_json(path)
    records = []
    for index, item in enumerate(payload["records"]):
        task_text = normalize_task_text(item["task"])
        tool_names = item.get("tool_names") or []
        source_path = item.get("source_file", "")
        if source_path:
            source_task_id = Path(source_path).stem
        else:
            source_task_id = f"enterprise_ops_gym_{item.get('gym', 'unknown')}_{index + 1:04d}"
        source_record = {
            "source_index": index,
            "source_path": source_path or str(path),
            "gym": item.get("gym", ""),
            "tool_count_observed": len(tool_names),
        }
        records.append(
            build_seed(
                source_dataset="EnterpriseOps-Gym",
                source_task_id=source_task_id,
                task_text=task_text,
                tool_names=tool_names,
                source_record=source_record,
                source_domain=item.get("gym"),
            )
        )
    return records


def _normalize_toucan_tool_name(raw: str) -> str:
    """Convert TOUCAN's `server::tool` form into a canonical kebab-case identifier."""
    if not raw:
        return ""
    # `target_tools` uses `Server Name::tool_name`; the matching `relevant_tools[].name`
    # is kebab-cased. Lowercase + replace spaces and `::` with `-` to align them.
    cleaned = raw.strip().replace("::", "-").replace(" ", "-")
    return cleaned.lower()


def _toucan_user_turns(question_field: Any) -> list[str]:
    """TOUCAN stores multi-turn user prompts as a JSON-stringified array."""
    if isinstance(question_field, list):
        return [str(turn) for turn in question_field if turn]
    if isinstance(question_field, str):
        try:
            parsed = json.loads(question_field)
        except json.JSONDecodeError:
            return [question_field]
        if isinstance(parsed, list):
            return [str(turn) for turn in parsed if turn]
        if isinstance(parsed, str):
            return [parsed]
    return []


def normalize_toucan(path: Path, min_confidence: str = "high") -> list[dict]:
    """Normalize TOUCAN multi-turn enterprise trajectories into seed records.

    `min_confidence` filters by the classifier's enterprise-label confidence
    ("high" | "medium" | "low"). The TOUCAN export also exposes
    `probability_enterprise`, but we use the discrete bucket here for clarity.
    """
    threshold = TOUCAN_CONFIDENCE_RANK.get(min_confidence, 3)
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            label = item.get("enterprise_label") or {}
            confidence = label.get("confidence", "low")
            if TOUCAN_CONFIDENCE_RANK.get(confidence, 0) < threshold:
                continue

            user_turns = _toucan_user_turns(item.get("question"))
            task_text = normalize_task_text(" ".join(user_turns)) if user_turns else ""
            if not task_text:
                continue

            target_tools = item.get("target_tools") or []
            tool_names = unique_ordered(
                _normalize_toucan_tool_name(name) for name in target_tools if name
            )

            source_task_id = item.get("uuid") or f"toucan_{item.get('record_index', len(records)):08d}"
            source_record = {
                "source_index": item.get("record_index"),
                "source_path": str(path),
                "uuid": item.get("uuid"),
                "subset_name": item.get("subset_name", ""),
                "confidence": confidence,
                "probability_enterprise": label.get("probability_enterprise"),
                "requested_mcp_servers": item.get("requested_mcp_servers") or [],
                "matched_mcp_servers": item.get("matched_mcp_servers") or [],
                "user_turn_count": len(user_turns),
                "tool_count_observed": len(tool_names),
            }
            records.append(
                build_seed(
                    source_dataset="Toucan",
                    source_task_id=source_task_id,
                    task_text=task_text,
                    tool_names=tool_names,
                    source_record=source_record,
                )
            )
    return records


def normalize_the_agent_company(path: Path) -> list[dict]:
    payload = load_json(path)
    records = []
    for index, item in enumerate(payload["tasks"]):
        task_text = normalize_task_text(item["task_prompt"])
        tool_names = item.get("function_names") or []
        source_task_id = item.get("task_id") or f"the_agent_company_{index + 1:04d}"
        source_record = {
            "source_index": index,
            "source_path": str(path),
            "task_id": item.get("task_id", ""),
            "trajectory_file": item.get("trajectory_file", ""),
            "tool_count_observed": len(tool_names),
            "function_call_counts": item.get("function_call_counts", {}),
        }
        records.append(
            build_seed(
                source_dataset="TheAgentCompany",
                source_task_id=source_task_id,
                task_text=task_text,
                tool_names=tool_names,
                source_record=source_record,
            )
        )
    return records


def pilot_candidate_score(seed: dict) -> int:
    score = seed["conversion_readiness"]["score"]
    tool_count = len(seed["tool_names"])
    if 3 <= tool_count <= 8:
        score += 10
    elif tool_count > 8:
        score += 4

    if len(seed["state_hooks"]["artifacts"]["durable_objects"]) >= 4:
        score += 4
    if seed["state_hooks"]["temporal"]["absolute_dates"]:
        score += 3
    if seed["state_hooks"]["temporal"]["sla_signals"]:
        score += 4
    if seed["state_hooks"]["relational"]["permissions"]:
        score += 3
    if seed["state_hooks"]["relational"]["dependencies"]:
        score += 3
    if "communication" in seed["tool_categories"] and len(seed["tool_categories"]) > 1:
        score += 2
    if len(seed["task_text"]) > 3500:
        score -= 4
    return score


def select_enterprise_ops_gym_pilot(all_records: list[dict]) -> tuple[list[dict], set[str]]:
    selected_ids = set()

    for domain, target_count in PILOT_DOMAIN_TARGETS.items():
        domain_records = [
            record
            for record in all_records
            if record["source_dataset"] == "EnterpriseOps-Gym"
            and record["environment"]["domain"] == domain
            and record["tool_names"]
        ]
        ranked_records = sorted(
            domain_records,
            key=lambda record: (pilot_candidate_score(record), record["seed_id"]),
            reverse=True,
        )

        used_tool_signatures = set()
        picked = 0
        for record in ranked_records:
            signature = tuple(record["tool_names"])
            if signature in used_tool_signatures:
                continue
            selected_ids.add(record["seed_id"])
            used_tool_signatures.add(signature)
            picked += 1
            if picked == target_count:
                break

        if picked < target_count:
            raise ValueError(f"Could not satisfy EnterpriseOps-Gym pilot quota for domain {domain}")

    annotated_records = []
    for record in all_records:
        if record["seed_id"] in selected_ids:
            annotated = copy.deepcopy(record)
            score = pilot_candidate_score(record)
            domain = record["environment"]["domain"]
            annotated["selection_tags"] = unique_ordered(
                annotated["selection_tags"] + ["pilot.enterprise_ops_gym_v1", f"pilot_domain:{domain}"]
            )
            annotated["notes"] = unique_ordered(
                annotated["notes"]
                + [f"Selected for the EnterpriseOps-Gym pilot slice with score {score} in the {domain} quota."]
            )
            validate_seed(annotated)
            annotated_records.append(annotated)

    annotated_records.sort(key=lambda record: (record["environment"]["domain"], record["seed_id"]))
    return annotated_records, selected_ids


def annotate_all_records(all_records: list[dict], selected_ids: set[str]) -> list[dict]:
    annotated = []
    for record in all_records:
        updated = copy.deepcopy(record)
        if updated["seed_id"] in selected_ids:
            updated["selection_tags"] = unique_ordered(
                updated["selection_tags"] + ["selected_for:enterprise_ops_gym_pilot_v1"]
            )
            updated["notes"] = unique_ordered(
                updated["notes"]
                + ["This imported seed is included in the EnterpriseOps-Gym pilot slice."]
            )
        validate_seed(updated)
        annotated.append(updated)
    return annotated


def print_summary(all_records: list[dict], pilot_records: list[dict]):
    dataset_counts = Counter(record["source_dataset"] for record in all_records)
    readiness_counts = Counter(record["conversion_readiness"]["status"] for record in all_records)
    pilot_domain_counts = Counter(record["environment"]["domain"] for record in pilot_records)

    print("Normalized source records:")
    for dataset_name, count in sorted(dataset_counts.items()):
        print(f"  {dataset_name}: {count}")

    print("Conversion readiness:")
    for status, count in sorted(readiness_counts.items()):
        print(f"  {status}: {count}")

    print("EnterpriseOps-Gym pilot slice:")
    for domain, count in sorted(pilot_domain_counts.items()):
        print(f"  {domain}: {count}")


def main():
    args = parse_args()

    all_records = []
    all_records.extend(normalize_enterprise_lab(args.enterprise_lab_path))
    all_records.extend(normalize_enterprise_ops_gym(args.enterprise_ops_gym_path))
    all_records.extend(normalize_the_agent_company(args.the_agent_company_path))
    if not args.skip_toucan and args.toucan_path.exists():
        all_records.extend(
            normalize_toucan(args.toucan_path, min_confidence=args.toucan_min_confidence)
        )
    elif not args.skip_toucan:
        print(f"Skipping TOUCAN ingestion: {args.toucan_path} not found")

    pilot_records, selected_ids = select_enterprise_ops_gym_pilot(all_records)
    annotated_all_records = annotate_all_records(all_records, selected_ids)

    dump_jsonl(args.output_jsonl, annotated_all_records)
    dump_jsonl(args.pilot_output_jsonl, pilot_records)
    print_summary(annotated_all_records, pilot_records)


if __name__ == "__main__":
    main()
