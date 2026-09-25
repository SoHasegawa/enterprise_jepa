"""ADP (Agent Data Protocol) trajectory inventory, shared by both world-model trainers.

`src/finetuning_jepa.py` (JEPA / latent world model) and `src/finetuning.py` (causal-LM world
model) both need the same ADP benchmark list. This module is the single definition so the two
cannot drift; it deliberately imports nothing from either trainer, since finetuning_jepa
already imports finetuning at module scope and a back-import would be circular.

Each ADP benchmark ships as one combined file with no official train/test split, so a preset
built from these paths reuses the same file for train and eval -- pass explicit
--train-data-path / --eval-data-path when a held-out split matters.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAJECTORIES_DIR = REPO_ROOT / "trajectories"

ADP_TRAJECTORY_DATASETS: tuple[str, ...] = (
    "agenttuning_alfworld",
    "agenttuning_db",
    "agenttuning_kg",
    "agenttuning_mind2web",
    "agenttuning_os",
    "agenttuning_webshop",
    "code_feedback",
    "codeactinstruct",
    "coderforge_preview",
    "go-browse-wa",
    "mind2web",
    "mini-coder",
    "nebius_SWE-agent-trajectories",
    "nemotron_terminal_corpus",
    "nnetnav-live",
    "nnetnav-wa",
    "openhands",
    "orca_agentinstruct",
    "swe-gym_openhands_sampled_trajectories",
    "swe-play-trajectories",
    "swe-smith",
    "synatra",
    # TOUCAN, enterprise-filtered: 131,000 trajectories / 600,577 action-observation pairs,
    # converted from trajectories/toucan_enterprise.jsonl (single-turn chat records dropped).
    # This replaced the uncurated `toucan_1_5m_multiturn` (955k trajectories / 19.2 GB) in the
    # ADP aggregate -- same domain as the enterprise benchmarks, and a fifth of the load cost.
    # The uncurated extracts stay on disk and remain selectable as the standalone
    # `toucan_1_5m_multiturn` / `toucan_1_5m` presets; the legacy curated split is
    # `toucan_curated`.
    "toucan_enterprise",
    # Function-calling subsets from ADP v2 (neulab/adp-v2, ATIF-v1.7 schema; raw std files
    # under /data/user/adp_v2; regenerate with
    #   generate_adp_world_model_trajectories.py --snapshot-path /data/user/adp_v2 \
    #     --dataset dolci_instruct_sft_tool_use --dataset toolmind
    # ). Included to familiarize the model with generic tool-calling trajectories. The other
    # v2 tool-flagged subsets were inspected and excluded: cognitivekernel_pro_sft carries
    # tool CALLS but zero observations (no transitions to learn), and
    # CharlieDreemur_OpenManus-RL embeds actions in message text (2 structured calls per 300
    # records) -- both would be auto-dropped by the no-last_tool_output loader gate anyway.
    "dolci_instruct_sft_tool_use",
    "toolmind",
)

# Web-browsing ADP benchmarks. These trajectory files are large and expensive to load;
# --skip-web-trajectories drops them from any preset/explicit path list.
WEB_BROWSING_ADP_DATASETS: tuple[str, ...] = (
    "agenttuning_mind2web",
    "go-browse-wa",
    "mind2web",
    "nnetnav-live",
    "nnetnav-wa",
    "synatra",
    "mini-coder",
    "coderforge_preview",
    "nemotron_terminal_corpus",
)

# ADP subsets that are terminal/shell-centric, excluded alongside Terminal-Bench itself in the
# *_no_terminalbench presets so the ablation removes the terminal DOMAIN rather than one file.
TERMINAL_DOMAIN_ADP_DATASETS: tuple[str, ...] = ("nemotron_terminal_corpus",)


# Enterprise-filtered TOUCAN. Converted from trajectories/toucan_enterprise.jsonl (raw
# ShareGPT-style {id, system, conversations} records, 150,000 of them / 600,578
# action-observation pairs) by
#   src/generation/generate_toucan_world_model_trajectories.py --toucan-path <that file>
# The default --min-action-count=1 drops the 19,000 single-turn chat records that contain no
# tool call at all, leaving 131,000 trajectories that carry a real transition.
TOUCAN_ENTERPRISE_DATA_PATH = DEFAULT_TRAJECTORIES_DIR / "toucan_enterprise_world_model_trajectories.json"
# Kept selectable on its own after being dropped from the ADP aggregate above.
TOUCAN_1_5M_MULTITURN_DATA_PATH = (
    DEFAULT_TRAJECTORIES_DIR / "toucan_1_5m_multiturn_world_model_trajectories.json"
)
# The conversion also wrote a 60/40 split (toucan_enterprise_world_model_{train,test}_
# trajectories.json), but the presets deliberately do NOT use it: toucan_enterprise is loaded
# from the combined file above like every other ADP benchmark. Train and eval therefore see the
# same trajectories -- pass explicit --train-data-path / --eval-data-path (the split files are
# still on disk) when a held-out eval matters.


def adp_trajectory_path(name: str) -> Path:
    return DEFAULT_TRAJECTORIES_DIR / f"{name}_world_model_trajectories.json"


# 25k-trajectory seeded subsets of the three corpora that dominate extraction time, cache size
# and per-rank RAM (built by src/data_preparation/subsample_world_model_trajectories.py, seed
# 42). 11.6 GB -> 1.86 GB. Each exists as BOTH .json and .jsonl, so finetuning.py's load_json
# path and finetuning_jepa.py's streaming path both work; the JEPA loader's
# resolve_streamable_path additionally prefers the .jsonl sibling automatically.
SUBSAMPLED_25K: dict[str, str] = {
    # NOT 25k any more: the uniform 25k subsample reproduced TOUCAN's success skew (2.5%
    # explicit failures against EnterpriseOps-Gym's 15%), which is what limits the outcome and
    # canonical-event minority classes. This stem is the union of that subsample with an
    # outcome-stratified draw that takes every failure-bearing trajectory in the corpus, with
    # reasoning pseudo-tool steps collapsed throughout (see
    # src/data_preparation/build_toucan_outcome_stratified_subsample.py). 38.2k trajectories /
    # 154.6k steps at 9.5% failure. The key is kept in SUBSAMPLED_25K so every preset built with
    # subsampled=True picks it up with no other edits.
    "toucan_enterprise": "toucan_enterprise_25k_plus_failure15",
    "dolci_instruct_sft_tool_use": "dolci_instruct_sft_tool_use_25k",
    "toolmind": "toolmind_25k",
}


def subsampled_25k_path(name: str) -> Path:
    """The 25k subset for `name`, or its full file when no subset exists."""
    return adp_trajectory_path(SUBSAMPLED_25K.get(name, name))


ADP_TRAJECTORY_PATHS: dict[str, Path] = {
    name: adp_trajectory_path(name) for name in ADP_TRAJECTORY_DATASETS
}
# Every ADP benchmark -- toucan_enterprise now included -- ships as one combined file, so train
# and eval reuse it. The train/eval lists are kept separate (rather than one shared list) so a
# per-benchmark split can be reintroduced here without touching either trainer.
ADP_ALL_TRAIN_DATA_PATHS: list[Path] = [
    ADP_TRAJECTORY_PATHS[name] for name in ADP_TRAJECTORY_DATASETS
]
ADP_ALL_EVAL_DATA_PATHS: list[Path] = [
    ADP_TRAJECTORY_PATHS[name] for name in ADP_TRAJECTORY_DATASETS
]
WEB_BROWSING_TRAJECTORY_PATHS: set[Path] = {
    ADP_TRAJECTORY_PATHS[name] for name in WEB_BROWSING_ADP_DATASETS
}
TERMINAL_DOMAIN_ADP_PATHS: set[Path] = {
    ADP_TRAJECTORY_PATHS[name] for name in TERMINAL_DOMAIN_ADP_DATASETS if name in ADP_TRAJECTORY_PATHS
}
ADP_NO_TERMINAL_TRAIN_DATA_PATHS: list[Path] = [
    ADP_TRAJECTORY_PATHS[name]
    for name in ADP_TRAJECTORY_DATASETS
    if name not in TERMINAL_DOMAIN_ADP_DATASETS
]
ADP_NO_TERMINAL_EVAL_DATA_PATHS: list[Path] = [
    ADP_TRAJECTORY_PATHS[name]
    for name in ADP_TRAJECTORY_DATASETS
    if name not in TERMINAL_DOMAIN_ADP_DATASETS
]
# Same inventory with the three heavy corpora swapped for their 25k subsets. Used by the
# `*_25k` presets in both trainers.
ADP_ALL_25K_DATA_PATHS: list[Path] = [
    subsampled_25k_path(name) for name in ADP_TRAJECTORY_DATASETS
]
ADP_NO_TERMINAL_25K_DATA_PATHS: list[Path] = [
    subsampled_25k_path(name)
    for name in ADP_TRAJECTORY_DATASETS
    if name not in TERMINAL_DOMAIN_ADP_DATASETS
]
# --skip-web-trajectories filters by PATH, so the subset paths need their own membership set
# (the three subsampled corpora are not web-browsing, but the helper keeps this exact).
WEB_BROWSING_25K_PATHS: set[Path] = {
    subsampled_25k_path(name) for name in WEB_BROWSING_ADP_DATASETS
}


# --- downstream-targeted mixtures ---------------------------------------------------------
# The downstream benchmarks (EnterpriseOps-Gym, CRMArenaPro, and the planned Workspace-Bench /
# WorkBench / OdysseyBench) are all structured tool/API calls over business entities: tickets,
# CRM records, mail, calendar, files, spreadsheets. These two lists exist so the SWE/code share
# of the mixture can be A/B'd against dropping it, instead of carried unexamined.
#
# Excluded from BOTH (measured on the real corpora, see the [data] yield log): every dataset
# that extracts to ZERO examples because it carries no last_tool_output --
# orca_agentinstruct (4.76 GB), nebius_SWE-agent-trajectories (0.47 GB), codeactinstruct, and
# agenttuning_{alfworld,db,kg,webshop}. Those are parsed and then discarded, so keeping them in
# a mixture is pure load cost. Web/browser and terminal corpora are excluded too -- they are a
# different domain from every downstream target.
ENTERPRISE_TOOL_CALLING_ADP_DATASETS: tuple[str, ...] = (
    "toucan_enterprise",
    "dolci_instruct_sft_tool_use",
    "toolmind",
)
# The SWE/code arm: cheap in bytes (~2 GB) but ~495k examples, a quarter of the ADP example
# count. Bash/file-edit/patch actions over code repos -- generic action->observation dynamics,
# but a different tool vocabulary and outcome semantics from the downstream targets.
SWE_CODE_ADP_DATASETS: tuple[str, ...] = (
    "swe-smith",
    "code_feedback",
    "swe-play-trajectories",
    "swe-gym_openhands_sampled_trajectories",
    "openhands",
    "agenttuning_os",
)


def enterprise_tool_calling_paths(*, include_swe: bool, subsampled: bool) -> list[Path]:
    names = list(ENTERPRISE_TOOL_CALLING_ADP_DATASETS)
    if include_swe:
        names += list(SWE_CODE_ADP_DATASETS)
    resolve = subsampled_25k_path if subsampled else adp_trajectory_path
    return [resolve(name) for name in names]
