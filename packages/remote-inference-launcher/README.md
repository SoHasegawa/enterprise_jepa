# remote-inference-launcher

Workspace Python package for launching vLLM inference servers and exposing them
as OpenAI-compatible `/v1` endpoints.

The package is intentionally generic at the boundary: callers provide an
endpoint, local process settings, `ssh_target`, scheduler settings, model
settings, and environment paths. Benchmark-specific environment variable names
belong in `ejepa`, not in this package.

The current implementation supports:

- already-running OpenAI-compatible endpoints
- local vLLM processes
- local vLLM environment bootstrap
- SSH-launched remote vLLM processes without Slurm
- SSH-launched remote vLLM environment bootstrap without Slurm
- Slurm-backed vLLM jobs
- named fleets such as actor/critic endpoint groups
- first-ready endpoint races across multiple providers or resource targets
- explicit Ray-backed multi-node Slurm vLLM jobs
- Slurm-backed vLLM environment bootstrap jobs
- run registries, detached controllers, and reusable endpoint leases
- reusable endpoint pools with assignment, health, manifest, and reconnect commands
- config validation, advisory preflight, and Slurm runtime preflight checks
- an SSH setup helper for printing connection snippets

All launchers pass `ssh_target` directly to OpenSSH. It can be a normal SSH
alias from `~/.ssh/config`, a target such as `user@host`, or an SSH URI such as
`ssh://user@host:10022`, as long as the local `ssh` command can use it.

## Install

This package is vendored into the repository uv workspace:

```bash
uv sync
```

From the repository, run the CLI through the package environment:

```bash
uv run --project packages/remote-inference-launcher remote-inference-launcher --help
```

Installed script usage is simply `remote-inference-launcher ...`.

## Command Reference

The current CLI surface is:

| Command | Purpose |
| --- | --- |
| `start` | Start one strict YAML inference config and keep owned resources alive. |
| `check` | Validate an already-running OpenAI-compatible endpoint. |
| `validate` | Validate strict inference YAML and optional preflight checks. |
| `status`, `env`, `logs`, `cleanup-command`, `stop` | Inspect and clean up run registries created by `start` or detached controllers. |
| `lease start/status/env/attach/recover/stop` | Create, recover, and reuse a ready endpoint across benchmark runs. |
| `pool status/acquire/acquire-batch/release/release-batch/health/manifest/reconnect/recommend-partitions` | Coordinate exclusive claims on reusable lease-backed endpoints. |
| `bootstrap` | Create or verify local, SSH, or Slurm vLLM Python environments from strict bootstrap YAML. |
| `slurm-vllm` | Lower-level Slurm serving launcher with YAML plus scalar CLI overrides. |
| `local-vllm-bootstrap`, `ssh-vllm-bootstrap`, `slurm-vllm-bootstrap` | Bootstrap-specific lower-level commands with YAML plus scalar CLI overrides. |
| `ssh-setup` | Print SSH config, ssh-agent, and verification snippets. |

## Python Usage

Local vLLM:

```python
from remote_inference_launcher.local_vllm import LocalVllmConfig, LocalVllmLauncher

config = LocalVllmConfig(
    model="org/model-name",
    served_model_name="actor",
    python_bin=".venv-vllm/bin/python",
    port=8123,
    tensor_parallel_size=2,
)

with LocalVllmLauncher(config).running() as session:
    print(session.api_base)
```

Existing OpenAI-compatible endpoint:

```python
from remote_inference_launcher.existing_endpoint import (
    ExistingEndpointConfig,
    ExistingEndpointLauncher,
)

config = ExistingEndpointConfig(
    api_base="http://127.0.0.1:8000/v1",
    served_model_name="org/model-name",
)

with ExistingEndpointLauncher(config).running() as session:
    print(session.api_base)
```

Slurm vLLM:

```python
from remote_inference_launcher.slurm_vllm import SlurmVllmConfig, SlurmVllmLauncher

config = SlurmVllmConfig(
    ssh_target="slurm-login",
    model="org/model-name",
    served_model_name="org/model-name",
    python_bin="~/.venv-vllm/bin/python",
    partition="batch-8gpu",
    walltime="6:00:00",
    num_gpus=2,
    memory="384GB",
    cpus_per_task=20,
)

with SlurmVllmLauncher(config).running() as session:
    print(session.api_base)
    print(session.summary_path)
```

SSH vLLM without Slurm:

```python
from remote_inference_launcher.ssh_vllm import SshVllmConfig, SshVllmLauncher

config = SshVllmConfig(
    ssh_target="gpu-host",
    model="org/model-name",
    served_model_name="org/model-name",
    python_bin="~/.venv-vllm/bin/python",
)

with SshVllmLauncher(config).running() as session:
    print(session.api_base)
    print(session.logs)
```

Fleet:

Fleet startup launches named logical endpoints with optional concurrency,
handoff, failure, and lifetime controls. Slurm-backed endpoints are submitted as
separate Slurm jobs, so use generated ports for normal runs and explicit ports
only when a fixed integration requires them.

```python
from remote_inference_launcher.fleet import FleetConfig, InferenceFleetLauncher
from remote_inference_launcher.slurm_vllm import SlurmVllmConfig

launcher = InferenceFleetLauncher(
    FleetConfig(
        max_active_launches=2,
        endpoints={
            "actor": SlurmVllmConfig(
                ssh_target="slurm-login",
                model="org/actor-model",
                served_model_name="actor",
                python_bin="~/.venv-vllm/bin/python",
                partition="batch-8gpu",
                walltime="6:00:00",
                num_gpus=2,
                memory="384GB",
                cpus_per_task=20,
            ),
            "critic": SlurmVllmConfig(
                ssh_target="slurm-login",
                model="org/critic-model",
                served_model_name="critic",
                python_bin="~/.venv-vllm/bin/python",
                partition="batch-8gpu",
                walltime="6:00:00",
                num_gpus=2,
                memory="384GB",
                cpus_per_task=20,
            ),
        },
    )
)

with launcher.running() as sessions:
    print(sessions["actor"].api_base)
    print(sessions["critic"].api_base)
```

`model` is required. vLLM serving options such as `max_model_len`,
`max_num_batched_tokens`, `max_num_seqs`, and `gpu_memory_utilization` are
optional and are only passed to vLLM when set.

## YAML Usage

Strict inference YAML supports these `kind` values:

| Kind | What it launches |
| --- | --- |
| `existing_endpoint` | Validates and exposes an endpoint already managed elsewhere. |
| `local_vllm` | Starts vLLM as a local process. |
| `ssh_vllm` | Starts vLLM on a remote SSH host and opens a local tunnel. |
| `slurm_vllm` | Submits vLLM through Slurm, waits for compute-node readiness, and opens a local tunnel. |
| `fleet` | Starts multiple named logical endpoints under one lifecycle. |
| `endpoint_race` | Tries multiple candidates for one logical endpoint and keeps the first ready endpoint. |

```yaml
kind: slurm_vllm
ssh_target: slurm-login
model: org/model-name
served_model_name: org/model-name
python_bin: ~/.venv-vllm/bin/python
partition: batch-8gpu
walltime: "6:00:00"
num_gpus: 2
memory: 384GB
cpus_per_task: 20
```

For Slurm vLLM, omit collision-prone fields for normal benchmark runs:

- `job_name` becomes `<job_name_prefix>-<endpoint-label>-<run-suffix>`,
  defaulting to `ril-<endpoint-label>-<run-suffix>`.
- `out_dir` becomes a run-scoped directory below
  `remote_out_dir_root` or `~/tmp/remote-inference-launcher/slurm-vllm`.
- `local_port` is reserved locally immediately before tunnel startup.
- `remote_port` is selected inside the allocated Slurm job and written to
  `remote-inference-state.json` before vLLM starts.

Explicit values remain supported and are treated as intentional fixed choices.
Use them only for firewall rules, reproducible manual debugging, or fixed
integrations.

Already-running endpoints use `existing_endpoint` and are validated with the
same OpenAI-compatible readiness check:

```yaml
kind: existing_endpoint
name: shared-model
api_base: http://127.0.0.1:8000/v1
served_model_name: org/example-model
discover_model: true
```

If `served_model_name` and `model` are both omitted and `discover_model: true`,
the launcher uses the first model ID returned by `/v1/models`.

Local GPU and non-Slurm SSH configs use the same readiness, summary, capacity,
and EJEPA handoff fields:

```yaml
kind: local_vllm
name: local-gpu
model: org/example-model
python_bin: .venv-vllm/bin/python
max_model_len: 262144
max_num_seqs: 2
```

```yaml
kind: ssh_vllm
name: remote-ssh
ssh_target: gpu-host
model: org/example-model
python_bin: ~/.venv-vllm/bin/python
max_model_len: 262144
max_num_seqs: 2
```

Omit `remote_port` for generated remote port selection, or set it explicitly
when firewall rules or fixed integrations require a stable server port.

```python
from remote_inference_launcher.inference_config import load_inference_launcher

launcher = load_inference_launcher("slurm-vllm.yaml")

with launcher.running() as session:
    print(session.api_base)
```

Fleet configs use named endpoint mappings:

```yaml
kind: fleet
max_active_launches: 2
failure_policy: fail_fast
handoff_mode: all_ready
endpoints:
  actor:
    kind: slurm_vllm
    ssh_target: slurm-login
    model: org/actor-model
    served_model_name: actor
    python_bin: ~/.venv-vllm/bin/python
    partition: batch-8gpu
    walltime: "6:00:00"
    num_gpus: 2
    memory: 384GB
    cpus_per_task: 20
  critic:
    kind: local_vllm
    model: org/critic-model
    served_model_name: critic
    python_bin: .venv-vllm/bin/python
```

Independent sweep fleets can hand off endpoints incrementally and release each
endpoint after its assigned work:

```yaml
kind: fleet
name: sweep
max_active_launches: 4
launch_stagger_seconds: 15
failure_policy: keep_ready
handoff_mode: incremental_ready
endpoint_lifetime_policy: per_endpoint
resource_budget:
  max_concurrent_logical_launches: 4
  max_active_candidate_attempts: 4
  max_submitted_slurm_jobs: 4
  max_total_requested_gpus: 4
endpoints:
  variant_a:
    kind: slurm_vllm
    ssh_target: cluster-a
    model: org/example-model
    partition: gpu
    walltime: "6:00:00"
    num_gpus: 1
    memory: 180G
    cpus_per_task: 16
  variant_b:
    kind: slurm_vllm
    ssh_target: cluster-a
    model: org/example-model
    partition: gpu
    walltime: "6:00:00"
    num_gpus: 1
    memory: 180G
    cpus_per_task: 16
```

Use `endpoint_race` when several providers are viable and the first endpoint to
pass readiness should win:

```yaml
kind: endpoint_race
name: example-model
max_active_candidates: 2
winner_condition: endpoint_ready
cancel_losers: true
candidates:
  - name: local-gpu
    kind: local_vllm
    model: org/example-model
    python_bin: ~/.venv-vllm/bin/python
  - name: existing
    kind: existing_endpoint
    api_base: http://127.0.0.1:8000/v1
    served_model_name: org/example-model
  - name: remote-ssh
    kind: ssh_vllm
    ssh_target: gpu-host
    model: org/example-model
    python_bin: ~/.venv-vllm/bin/python
    remote_port: 18817
  - name: slurm
    kind: slurm_vllm
    ssh_target: cluster-a
    model: org/example-model
    partition: gpu
    walltime: "6:00:00"
    num_gpus: 1
    memory: 180G
    cpus_per_task: 16
```

`max_concurrent_logical_launches` limits logical endpoint launches in progress.
Candidate attempts, submitted Slurm jobs, and requested GPU budget stay held
until the corresponding owned endpoint is released or stopped.

Use Slurm `resource_preferences` when one logical endpoint should try resource
targets in order:

```yaml
kind: slurm_vllm
name: example-model
ssh_target: cluster-a
model: org/example-model
partition: gpu
walltime: "6:00:00"
num_gpus: 1
memory: 180G
cpus_per_task: 16
queue_policy:
  max_pending_seconds: 1800
  fallback_on_pending: true
  poll_interval_seconds: 60
  status_interval_seconds: 300
resource_preferences:
  - name: preferred
    ssh_target: cluster-a
    partition: gpu
  - name: fallback
    ssh_target: cluster-b
    partition: gpu-long
    python_bin: ~/.venv-vllm-alt/bin/python
    memory: 220G
    cpus_per_task: 24
```

Only candidates listed in `resource_preferences` are attempted by default. Set
`include_base_resource_candidate: true` to also try the parent Slurm config as a
final non-duplicate candidate.
Per-candidate entries override Slurm/vLLM launch fields; structured policies
such as `readiness`, `queue_policy`, and `candidate_race` stay on the parent
logical endpoint.

`queue_policy.max_pending_seconds` is the Slurm pending-state timeout. With
`resource_preferences`, a pending timeout advances to the next candidate only
when `fallback_on_pending: true`; otherwise it fails the logical endpoint.
`fallback_on_pending` requires both `resource_preferences` and
`max_pending_seconds`.
`queue_policy.poll_interval_seconds` controls scheduler polling frequency.
`queue_policy.status_interval_seconds` controls user-visible heartbeat output;
state or reason changes are still printed immediately.

Resource preferences can be raced instead of attempted serially. This is useful
when several partitions or clusters are valid and the first ready endpoint
should win:

```yaml
kind: slurm_vllm
name: example-model
ssh_target: cluster-a
model: org/example-model
partition: gpu
walltime: "6:00:00"
num_gpus: 1
memory: 180G
cpus_per_task: 16
candidate_race:
  enabled: true
  max_active_candidates: 2
  winner_condition: endpoint_ready
  cancel_losers: true
  launch_stagger_seconds: 30
resource_preferences:
  - name: short
    partition: gpu-short
  - name: long
    partition: gpu-long
```

For Slurm resource races, `winner_condition` must be `endpoint_ready` and
`cancel_losers` must stay true; detached loser jobs are intentionally not
supported.

Multi-node Slurm serving uses Ray explicitly:

```yaml
kind: slurm_vllm
name: ray-model
ssh_target: cluster-a
model: org/large-model
python_bin: ~/.venv-vllm-ray/bin/python
partition: gpu
walltime: "12:00:00"
nodes: 2
num_gpus: 4
memory: 384G
cpus_per_task: 32
distributed_backend: ray
tensor_parallel_size: 8
ray_port: 6379
```

`nodes > 1` requires `distributed_backend: ray`; `distributed_backend` must be
omitted for single-node jobs. `ray_port` is the Ray control port.
`head_node_port` may be set when the vLLM API port on the Ray head node must be
fixed; otherwise the launcher derives it from the selected remote server port.

Common serving fields accepted by `local_vllm`, `ssh_vllm`, and `slurm_vllm`
include `model`, `served_model_name`, `api_key`, `python_bin`, `out_dir`,
`runtime_tmp_root`, `hf_home`, `target_device`, `rocm_eager_fallback`,
`tensor_parallel_size`, `pipeline_parallel_size`, `data_parallel_size`,
`gpu_memory_utilization`, `max_model_len`, `max_num_seqs`,
`max_num_batched_tokens`, `extra_args`, `ready_timeout_seconds`,
`check_interval_seconds`, `readiness`, `diagnostics`, `launch_summary_path`, and
`overwrite_launch_summary`. Local serving also accepts `host`, `port`, and
`keep_server`. SSH and Slurm launchers also accept `setup_cmd`,
`local_bind_host`, `local_port`, `remote_port`, `remote_out_dir_root`, and
remote keep-alive flags (`keep_remote_process` for SSH, `keep_remote_job` for
Slurm). Slurm additionally accepts scheduler fields such as `partition`,
`walltime`, `num_gpus`, `memory`, `cpus_per_task`, `nodes`, `exclude`,
`nodelist`, `sbatch_cmd`, `job_name`, `job_name_prefix`, `queue_policy`,
`candidate_race`, `resource_preferences`, and
`include_base_resource_candidate`.

## CLI Usage

Generic config-driven serving:

```bash
remote-inference-launcher start \
  --config inference.yaml \
  --launch-summary results/inference-launch-summary.json \
  --env-file /tmp/inference.env
```

`start` loads the same strict YAML inference configs described above. It waits
until the endpoint is ready before printing env values or writing `--env-file`.
When the launcher owns a local process, SSH tunnel, Slurm job, or fleet member,
`start` keeps those resources alive until interrupted. Stopping the command
stops owned resources unless the config explicitly keeps them. For smoke tests,
use `--exit-after-ready` to print/write env values and then immediately clean
up owned resources. `existing_endpoint` configs exit after readiness because the
endpoint is owned by another process. Use `--format json` when callers need a
structured session payload instead of shell-style env output.

Each `start` creates a run registry before launch side effects:

```text
.remote-inference-launcher/runs/<run-id>/
```

Use the registry-backed commands to inspect or clean up a run:

```bash
remote-inference-launcher status <run-id-or-registry-path>
remote-inference-launcher status <run-id-or-registry-path> --format json
remote-inference-launcher env <run-id-or-registry-path>
remote-inference-launcher env <run-id-or-registry-path> --endpoint actor
remote-inference-launcher cleanup-command <run-id-or-registry-path>
remote-inference-launcher logs <run-id-or-registry-path>
remote-inference-launcher logs <run-id-or-registry-path> --tail 200
remote-inference-launcher logs <run-id-or-registry-path> --paths-only
remote-inference-launcher stop <run-id-or-registry-path> --force
```

The run argument can be a run ID, a registry directory, or a summary path.
`env` supports endpoint selection for multi-endpoint runs, and `logs` supports
path-only log discovery. `stop` executes recorded cleanup commands; use
`cleanup-command` when you only want to print them.

Detached controllers are supported for long-running launches:

```bash
remote-inference-launcher start --config inference.yaml --detach
remote-inference-launcher start --config inference.yaml --detach --wait-ready
```

The parent prints the run ID, registry path, status command, and logs command.
With `--wait-ready`, it waits for `READY`, `LEASED`, or a terminal failure and
prints env output from the registry. `--ready-timeout` controls that parent-side
wait and does not change endpoint readiness timeouts inside the launcher config.

For incremental fleet handoff, use JSONL output:

```bash
remote-inference-launcher start --config fleet.yaml --format jsonl
```

The command emits `endpoint_ready`, `endpoint_failed`, `fleet_complete`, and
`start_complete` records as one JSON object per line. Each ready event includes
the endpoint API base, served model, summary path, cleanup command, and session
metadata.

Every launcher writes a machine-readable launch summary. If `--launch-summary`
or `launch_summary_path` is omitted, the path is generated under
`.remote-inference-launcher/runs/<run-id>/...`. Existing summaries are never
overwritten unless `--overwrite-launch-summary` or
`overwrite_launch_summary: true` is set. Summaries contain endpoint name,
backend kind, lifecycle state, API base, served model, job/process ids, cleanup
commands, readiness results, vLLM capacity hints when logs are available,
resource attempts, and `benchmark_handoff`. API keys are redacted; summaries
record only `api_key_set: true`.

Readiness means the endpoint can serve benchmark-shaped OpenAI-compatible
traffic. The shared check requires `/v1/models` and, by default, a tiny
`/v1/chat/completions` smoke request:

```yaml
readiness:
  smoke_test: chat_completion
  prompt: Reply with OK.
  max_tokens: 4
  timeout_seconds: 2
```

Set `smoke_test: disabled` only for environments that intentionally expose the
models endpoint before chat completions can be served.

CLI-generated env files and registry env output contain non-secret generic
variables only:

```bash
OPENAI_BASE_URL=...
OPENAI_MODEL_NAME=...
INFERENCE_LAUNCH_SUMMARY_PATH=...
INFERENCE_DEFAULT_BASE_URL=...
INFERENCE_DEFAULT_MODEL=...
INFERENCE_DEFAULT_LAUNCH_SUMMARY_PATH=...
```

Named fleet endpoints use `INFERENCE_<NAME>_*`, for example
`INFERENCE_ACTOR_BASE_URL` and `INFERENCE_ACTOR_LAUNCH_SUMMARY_PATH`.
When an endpoint requires an API key, provide `OPENAI_API_KEY` separately.

Reusable endpoint leases keep a ready endpoint attachable across multiple
benchmark runs:

```bash
remote-inference-launcher lease start --config inference.yaml --ttl 12h --detach --wait-ready
remote-inference-launcher lease start --config inference.yaml --ttl 12h --ttl-start ready_at --detach --wait-ready --ready-timeout 1800
remote-inference-launcher lease status <lease-id>
remote-inference-launcher lease env <lease-id> > inference.env
remote-inference-launcher lease attach <lease-id> --format json > inference-handoff.json
remote-inference-launcher lease attach <lease-id> --expect-config inference.yaml --format json
remote-inference-launcher lease recover <lease-id> --expect-config inference.yaml --format json
remote-inference-launcher lease stop <lease-id> --dry-run
remote-inference-launcher lease stop <lease-id>
```

`lease attach` emits the launcher handoff schema with lease ID, run ID,
semantic config hash, stable identity key, served model names, summary paths,
cleanup ownership, and endpoint sessions. Benchmark integrations should record
that handoff metadata instead of deriving reproducibility identity from the
local endpoint URL or port.

TTL values accept compact suffixes such as `30m`, `12h`, and `2d`.
Use `--ttl-start ready_at` when queued Slurm time should not count against the
attachment lifetime; the lease expiry is then set when the endpoint activates.
`--cleanup-on-expiry` records that an expired lease should be cleaned up,
`--expect-config` rejects attachments whose semantic config hash differs from
the lease, `--allow-expired` allows an explicit recovery attachment to an
expired lease, and `lease stop --dry-run` prints the cleanup plan without
running it.

`lease start --detach --wait-ready --ready-timeout <seconds>` controls only the
parent-side wait for the detached controller to report readiness. If that wait
times out while the controller and Slurm job may still be progressing, the
lease is marked `ready_timeout` rather than final failed. After the endpoint is
healthy, run `lease recover <lease-id>`; recovery checks lease ownership,
semantic config hash, endpoint URL, served model identity, Slurm job identity,
tunnel identity, and endpoint health before activating the lease and emitting a
normal handoff.

Endpoint pools build on leases when multiple workers need exclusive claims on
long-lived endpoints:

```bash
remote-inference-launcher pool status --format json
remote-inference-launcher pool status --health-depth scheduler --format json
remote-inference-launcher pool acquire \
  --model org/model-name \
  --min-context 32768 \
  --min-remaining 2h \
  --owner worker-07 \
  --shard shard-a \
  --assignment-ttl 4h \
  --format json
remote-inference-launcher pool acquire-batch \
  --model org/model-name \
  --count 12 \
  --shard-file shards.json \
  --owner experiment-wave-1 \
  --assignment-ttl 4h \
  --health-depth models \
  --format json > batch-handoff.json
remote-inference-launcher pool health --assignment <assignment-id> --format json
remote-inference-launcher pool manifest --assignment <assignment-id> --format json
remote-inference-launcher pool release <assignment-id>
remote-inference-launcher pool release-batch <batch-id>
remote-inference-launcher pool reconnect --assignment <assignment-id>
remote-inference-launcher pool recommend-partitions --config inference.yaml --format json
```

`pool status` defaults to local lease, registry, assignment, and summary files.
Scheduler refreshes use one SSH batch per target and only cheap Slurm commands
such as `squeue` and `sinfo`. Capability data is derived from recorded plans and
summaries, so pool selection does not import model libraries or scan model/cache
trees on login nodes.
The status payload reports generic health separately from acquisition
eligibility: `summary.healthy_free` includes healthy registry-backed endpoints,
while `summary.acquirable_healthy_free` counts endpoints that `pool acquire`
can actually claim. Each endpoint includes `acquirable` and, when false,
`not_acquirable_reason` such as `registry_endpoint_without_lease`,
`assignment_active`, `lease_not_active`, or `model_mismatch`.

`pool acquire` creates an assignment record under
`.remote-inference-launcher/pool/assignments/` and returns a generic handoff
with endpoint URL, served model, capability facts, health evidence, assignment
expiry, and lease expiry. `pool release` frees the assignment without stopping
the underlying endpoint; use `lease stop` when the remote serving resource
should be torn down.

`pool acquire-batch` returns one assignment handoff per shard and writes a
shared batch ID into the assignment records. Batch acquire is all-or-nothing by
default: if any shard cannot acquire an endpoint, already-created assignments
are released before the command fails. Use `--partial` only when callers can
handle `missing_shards` in the JSON handoff. `pool release-batch` is
idempotent and can release by batch ID or by repeated `--assignment` IDs.

Pool health depth is intentionally split by workflow. `pool status` accepts
`cached`, `scheduler`, `light`, `models`, and `full`. `cached` reads local
records only. `scheduler` refreshes Slurm state in one SSH batch per target and
keeps endpoint health cached. `light` and `models` add local tunnel and
`/v1/models` probes, and `full` also runs the tiny chat-completions readiness
smoke test. `pool acquire` and `pool health` accept `cached`, `local`,
`models`, and `full`; they do not run fresh Slurm scheduler refreshes. Run one
coordinator-side `pool status --health-depth scheduler` before a worker wave
when fresh queue/job state matters, then let workers call `pool acquire`
without creating an SSH storm.

`pool manifest` emits either a selected assignment/endpoint manifest or the
available endpoint set with `--available`. `pool reconnect` rebuilds the local
SSH tunnel for a leased Slurm endpoint without cancelling or resubmitting the
Slurm job. `pool recommend-partitions` reads the requested Slurm resources from
one or more inference configs, batches cheap `sinfo` inventory by `ssh_target`,
and reports compatible partitions without importing model libraries or scanning
remote model/cache trees.

Benchmark integration is opt-in. The shared benchmark runner can inject these
generic variables from an inference YAML passed with `--inference-config`:

```bash
ejepa bench run <benchmark> --inference-config inference.yaml
ejepa bench run <benchmark> --inference-config fleet.yaml
```

`ejepa bench run --inference-config` owns the launcher lifecycle in-process:
it waits through queueing and readiness, keeps tunnels and owned processes alive
while tasks run, writes `inference-launch-summary.json` into the benchmark result
directory, applies generic OpenAI env values, and cleans up on normal exit,
failure, or handled termination. If `--max-parallel` is omitted and the launcher
summary has `recommended_benchmark_max_parallel`, EJEPA records that recommendation
and prints it without changing benchmark parallelism. Pass `--max-parallel`
explicitly to use the recommendation. If an explicit parallelism exceeds the
recommendation, EJEPA prints a warning.
EJEPA records launcher summary paths, endpoint metadata, stable inference
identity, task IDs, generation knobs, and a git fingerprint in the benchmark
request metadata. For detached runs, use `status`, `logs`, and `stop` with the
recorded run ID for inspection and cleanup.

For fleet configs with `handoff_mode: incremental_ready`, custom callers can use
the fleet launcher's `set_event_callback()` API to observe `endpoint_ready`
events as endpoints arrive. The general EJEPA CLI waits for the configured
launcher path and keeps sweep, shard, result-resume, and per-variant orchestration
outside the launcher contract.

Benchmark-owned executors must read the generic variables directly or map them
to their own variable names. For example:

```bash
export MYBENCH_LLM_BASE_URL="${OPENAI_BASE_URL}"
export MYBENCH_LLM_MODEL="${OPENAI_MODEL_NAME}"
export MYBENCH_LLM_API_KEY="${OPENAI_API_KEY:-}"
```

Validate an already-running endpoint:

```bash
remote-inference-launcher check --base-url http://host:8000/v1
remote-inference-launcher check \
  --base-url http://host:8000/v1 \
  --api-key "$OPENAI_API_KEY" \
  --format json
```

Validate launch configs before submitting work:

```bash
remote-inference-launcher validate inference-a.yaml inference-b.yaml
remote-inference-launcher validate --format json fleet.yaml
remote-inference-launcher validate --preflight inference.yaml
remote-inference-launcher validate --strict-preflight inference.yaml
remote-inference-launcher validate --strict-preflight --fail-on-unknown-preflight inference.yaml
remote-inference-launcher validate --runtime-preflight inference.yaml
```

Validation resolves generated-value strategies, checks explicit local bind
ports, reports duplicate explicit local ports across local/SSH/Slurm launchers,
reports duplicate explicit SSH/Slurm remote ports for the same `ssh_target`,
reports duplicate explicit Slurm `job_name` and `out_dir`, and validates
endpoint racing plus aggregate `resource_budget` caps. `--preflight` also runs
advisory target checks for SSH reachability, Slurm command availability, Python
startup, setup commands, vLLM importability for non-Slurm SSH serving, and
configured model/cache/temp path visibility when those checks apply.
`--strict-preflight` turns durable preflight failures into validation errors.
Transient or unknown strict preflight outcomes make the report inconclusive;
add `--fail-on-unknown-preflight` when unknown outcomes should be errors.
Advisory preflight failures remain warnings. Slurm preflight also checks
non-allocating scheduler diagnostics such as `sbatch --test-only`,
`squeue --start`, and partition walltime metadata when they are available on
the target. Slurm-backed vLLM import checks are not run on login nodes; runtime
validation for those targets must run inside a Slurm allocation.
`--runtime-preflight` submits a short Slurm job that runs the configured setup
command, checks the configured Python, imports vLLM, checks target-device
visibility when requested, and writes a bounded JSON manifest. It is opt-in
because it consumes cluster resources.
Fleet budget validation counts owned endpoint resources that remain held after
readiness, not only the currently active launch wave.

Direct lower-level subcommands are available for Slurm vLLM use. Their `--config`
files use the same strict generic inference schema as `start` and `validate`,
including `kind: slurm_vllm`:

```bash
remote-inference-launcher slurm-vllm \
  --config slurm-vllm.yaml \
  --local-port 8123 \
  --keep-remote-job \
  --hold
```

Configuration precedence is:

```text
dataclass defaults < YAML config < CLI flags
```

The CLI generates explicit kebab-case flags from `SlurmVllmConfig`; run
`remote-inference-launcher slurm-vllm --help` for the full scalar list.
Structured policy blocks such as `readiness`, `queue_policy`,
`candidate_race`, `resource_preferences`, and `resource_budget` are YAML-only.

Set `verbosity` to `quiet`, `progress`, or `verbose` in Python/YAML, or pass
`--verbosity`, `--quiet`, or `--verbose` on the CLI. Quiet mode suppresses
progress lines but still prints final key/value output and errors. Verbose mode
adds remote command diagnostics.

## Diagnostics and Capacity

vLLM-backed launchers parse common log lines such as `GPU KV cache size` and
`Maximum concurrency ...` into summary fields:

- `vllm_max_concurrency`
- `recommended_benchmark_max_parallel`
- `max_model_len`
- `max_num_seqs`
- `kv_cache_tokens`

Startup failures are classified with stable codes when evidence is available:
`local_process_exited`, `local_port_bind_failed`, `remote_process_exited`,
`slurm_pending_timeout`, `slurm_cancelled_or_failed`,
`slurm_prolog_failure`, `ssh_connection_reset`, `ssh_tunnel_bind_failed`,
`remote_port_not_written`, `vllm_oom`, `vllm_engine_dead`,
`vllm_context_too_large`, `vllm_rocm_device_error`,
`vllm_tool_parser_or_template_error`, `readiness_models_failed`, and
`readiness_smoke_failed`. Summaries keep sanitized excerpts and log paths so
classifiers do not hide the underlying evidence.

## Multi-Server Checklist

- Prefer generated Slurm job names, ports, output directories, and summary
  paths for normal benchmark work.
- Use explicit ports only when fixed integrations require them, and keep them
  unique.
- Run `remote-inference-launcher validate ...` before launching several
  endpoints.
- Use `remote-inference-launcher validate --preflight ...` when onboarding a new
  SSH target, Slurm partition, Python environment, model path, or cache path.
- Set an aggregate `resource_budget` when combining fleets with endpoint or
  resource racing, and use `max_concurrent_logical_launches` for launch
  concurrency rather than live endpoint count.
- Keep independent sweep, shard, and resume policy orchestration in
  benchmark-owned wrappers or `common.inference_runtime` helpers rather than
  expanding the general EJEPA CLI surface.
- Keep benchmark `--max-parallel` at or below the launcher recommendation
  unless intentionally testing over-capacity behavior.
- Use candidate racing sparingly and keep `max_active_candidates` low.
- Cancel losers only after a candidate passes endpoint readiness, not merely
  after Slurm allocation.
- Save launch summaries with benchmark artifacts and use their cleanup commands
  for manual recovery after controller death.

## Bootstrap

The generic `bootstrap` command requires one of these strict bootstrap kinds:

| Kind | What it prepares |
| --- | --- |
| `local_vllm_bootstrap` | Local uv environment with vLLM and optional Ray. |
| `ssh_vllm_bootstrap` | Remote SSH uv environment with vLLM and optional Ray. |
| `slurm_vllm_bootstrap` | Remote uv environment prepared and verified inside a Slurm allocation. |

### Local or SSH Bootstrap

Local and non-Slurm SSH bootstrap create or verify a uv virtual environment and
install an exact vLLM package. They are explicit preparation steps; serving
configs still point at the resulting `python_bin`.
Bootstrap installs `ray_package` as well, defaulting to `ray`; set
`ray_package: ""` to skip Ray when the environment will never serve multi-node
models.

Local example:

```yaml
kind: local_vllm_bootstrap
environment_name: project-vllm
venv_path: ~/.project-vllm
vllm_package: vllm==0.20.1
backend: auto
python: "3.12"
install_uv_if_missing: true
```

```bash
remote-inference-launcher bootstrap --config bootstrap-local.yaml
```

SSH example:

```yaml
kind: ssh_vllm_bootstrap
ssh_target: gpu-cluster
environment_name: project-vllm-rocm
venv_path: ~/.project-vllm-rocm
vllm_package: vllm==0.22.0+rocm722
backend: rocm
python: "3.12"
install_uv_if_missing: true
setup_cmd: |
  source ~/.bashrc
```

```bash
remote-inference-launcher bootstrap --config bootstrap-ssh.yaml
```

The command prints the environment path, Python path, logs, and manifest path.
Use the printed `python_bin` value in `local_vllm`, `ssh_vllm`, or
`slurm_vllm` serving configs.

The lower-level bootstrap-specific commands also remain available:

```bash
remote-inference-launcher local-vllm-bootstrap --config bootstrap-local.yaml
remote-inference-launcher ssh-vllm-bootstrap --config bootstrap-ssh.yaml
```

Those two lower-level commands accept configs with or without a `kind` field and
also expose scalar config fields as CLI flags.

### Slurm Bootstrap

Bootstrap submits a short Slurm job that creates or verifies a remote Python
environment with an exact vLLM package. It does not start a model server; run
it once per remote environment, then use the resulting Python path in serving
configs.

The generic bootstrap command accepts a strict `kind: slurm_vllm_bootstrap`
YAML:

```yaml
kind: slurm_vllm_bootstrap
ssh_target: slurm-login
environment_name: project-vllm-cu129
venv_path: ~/.project-vllm-cu129
partition: batch-2gpu
walltime: "1:00:00"
num_gpus: 1
memory: 32GB
cpus_per_task: 4
vllm_package: vllm==0.19.1
install_uv_if_missing: true
```

```bash
remote-inference-launcher bootstrap --config bootstrap-slurm.yaml
```

The lower-level `slurm-vllm-bootstrap --config` command loads the Slurm
bootstrap dataclass directly, so omit the `kind` field for configs passed to
that command. It also exposes scalar Slurm bootstrap fields as CLI flags.

For known shared clusters, the bootstrap command has tested defaults so callers
do not have to remember package and partition details:

```bash
remote-inference-launcher slurm-vllm-bootstrap --ssh-target gpu-cluster --force
remote-inference-launcher slurm-vllm-bootstrap --ssh-target slurm-login --force
```

The built-in `gpu-cluster` profile creates `~/.ejepa-vllm-rocm` under the current
SSH user's remote home directory. It installs `vllm==0.22.0+rocm722` from ROCm wheels,
uses `batch-1gpu-short`, and skips Ray for the single-node case. Use the printed
`REMOTE_PYTHON` in serving configs and set `target_device: rocm`. Replace
`org/model-name` with the model you intend to serve:

```yaml
kind: slurm_vllm
ssh_target: gpu-cluster
python_bin: ~/.ejepa-vllm-rocm/bin/python
partition: batch-1gpu-short
target_device: rocm
model: org/model-name
walltime: "00:30:00"
num_gpus: 1
memory: 32GB
cpus_per_task: 4
```

The built-in `slurm-login` profile creates `~/.ejepa-vllm-env` under the current SSH
user's remote home directory. It installs `vllm==0.19.1`, CUDA 12.9-compatible
PyTorch, uses `batch-1gpu-exclusive`, and skips Ray for the single-node case.
Do not set `target_device` on slurm-login:

```yaml
kind: slurm_vllm
ssh_target: slurm-login
python_bin: ~/.ejepa-vllm-env/bin/python
partition: batch-1gpu-exclusive
model: org/model-name
walltime: "00:30:00"
num_gpus: 1
memory: 32GB
cpus_per_task: 4
```

These aliases are login nodes for submission and tunneling only. Do not run
`ssh_vllm` serving directly against `gpu-cluster` or `slurm-login`; use Slurm serving so
vLLM starts on an allocated compute node. If you need multi-node Ray serving,
pass an explicit Ray package such as `--ray-package ray==2.55.1`.

For other clusters, `vllm_package` is required and must be an exact
`vllm==VERSION` specifier.
`ray_package` defaults to `ray` for multi-node serving support and can be set
to an empty string to skip that install.
The bootstrap manifest now verifies vLLM importability and performs a tiny
Torch CUDA/HIP initialization inside the Slurm allocation. This catches common
bad environments, including CUDA wheels that require a newer NVIDIA driver than
slurm-login currently provides. Avoid upgrading slurm-login to `vllm==0.21.x` or
`torch==2.11.x` unless the cluster driver has also been upgraded.

Provision a remote vLLM environment with an explicit environment name:

```yaml
ssh_target: slurm-login
environment_name: project-vllm-cu129
venv_path: ~/.project-vllm-cu129
partition: batch-2gpu
walltime: "1:00:00"
num_gpus: 1
memory: 32GB
cpus_per_task: 4
vllm_package: vllm==0.19.1
```

```bash
remote-inference-launcher slurm-vllm-bootstrap \
  --config bootstrap.yaml \
  --install-uv-if-missing
```

The bootstrap command prints the Slurm job id, remote logs, remote environment,
remote Python path, and manifest path after verification.

## SSH Setup

`ssh-setup` prints copy-pasteable `~/.ssh/config`, `ssh-agent`, `ssh-add`, and
verification commands. It does not edit local files or affect launcher behavior.
Pass `--interactive` to prompt for missing values instead of requiring every
field on the command line.

For a generic host:

```bash
remote-inference-launcher ssh-setup \
  --alias custom \
  --hostname example.internal \
  --port 10022 \
  --user alice \
  --identity-file ~/.ssh/team-key
```

For the internal team hosts:

```bash
remote-inference-launcher ssh-setup \
  --preset slurm-login \
  --preset gpu-cluster \
  --user alice \
  --identity-file ~/.ssh/team-key
```

## Testing

See [tests/README.md](tests/README.md) for local pytest usage and opt-in live
SSH/Slurm integration tests.
