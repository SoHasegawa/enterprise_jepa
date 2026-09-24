# How Enterprise-JEPA Helps and Hurts Agentic Task Execution

Qualitative companion to the main agentic results table. Same framing as the table:
three harnesses — **Revision**, **ITP-I**, **Beam search** (`beam_interval`) — against the
no-world-model baseline; EOPS = EnterpriseOps-Gym task success (mean of 3 runs), CRM =
CRMArena-Pro upstream original accuracy (428 tasks), WB = WorkBench score rate (690 tasks),
AB = AutomationBench strict pass rate excluding `marketing` and `finance`, TB =
Terminal-Bench-2.0 pass rate (89 tasks).

| World model | Harness | EOPS | CRM | WB | AB | TB | AVG |
|---|---|---:|---:|---:|---:|---:|---:|
| No WM | Baseline | 0.375 | 0.320 | 0.765 | 0.187 | 0.247 | 0.304 |
| Enterprise-JEPA | Revision | 0.388 | 0.325 | **0.838** | 0.166 | 0.281 | 0.400 |
| | ITP-I | **0.425** | 0.313 | 0.828 | 0.151 | **0.303** | 0.404 |
| | Beam search | 0.404 | **0.339** | 0.803 | **0.204** | 0.292 | **0.408** |

All mechanism counts below are recomputed from per-task artifacts (`detail.json` and the
trajectory JSONL under each result directory) by `/tmp/taxonomy.py`. Where a benchmark has
several runs per harness, the mechanism analysis uses one representative run per harness
(identified in §8); the headline numbers in the table above are the run-averaged values.

---

## 1. What we can attribute, and where

Per-task attribution requires a baseline that does not move on its own:

| benchmark | baseline repeat behaviour | attribution granularity used |
|---|---|---|
| WorkBench | **0 / 100 task flips** across 3 runs on the 100-task subset; 690-task score identical to 4 dp | per-task tool-call diff vs deterministic baseline |
| EnterpriseOps-Gym | **2 / 80 task flips** across 3 runs; 95.5 % of 352 named verifiers deterministic | per-task and per-verifier tool-call diff |
| AutomationBench | mean credit reproduces to +0.007, but **42 % of task scores move** between same-config runs | tool-call diff on changed tasks; treated as aggregate evidence |
| Terminal-Bench-2.0 | 3 baseline runs span 0.247–0.258 | command-sequence diff, coarse labels |
| CRMArena-Pro | 13 % of tasks flip between same-config runs; original accuracy ±0.5 pp on the clean pair | task-category breakdown only |

---

## 2. The causal channel: the world model advises, it never acts

Across 664 world-model decisions on EnterpriseOps-Gym, **0 hard overrides** were applied;
the imagined plan beat the policy's own proposal in 3.8 % of decisions (advisorily), and the
only channel of influence is a text block appended to the prompt on 19.4 % of steps. The
pattern is identical on WorkBench (0 overrides in 200 task-runs) and AutomationBench
(0 overrides in 2,376 steps). Every behavioural change below is therefore *indirect*: the
injected advice changed the prompt and the policy sampled differently. Nothing below should be
read as "the world model corrected the action".

---

## 3. Classification of how the world model **helps**

Improved tasks, labelled by the tool-call difference between the world-model run and the
baseline run on the same task. Counts are tasks.

### 3.1 WorkBench (690 tasks; Revision 79 improved, ITP-I 75, Beam 49)

Counts use the corrected read/mutation split (see §3.8 — `analytics.*_count` tools are reads).

| help mechanism (mutation side) | Revision | ITP-I | Beam |
|---|---:|---:|---:|
| **corrected mutation arguments** | **35** | **35** | **21** |
| completed write batch (more of the same mutation) | 15 | 13 | 12 |
| acted where baseline only read | 13 | 17 | 8 |
| removed spurious mutation | 4 | 1 | 1 |
| added a missing mutation | 3 | 1 | 2 |
| other (substituted / mixed) | 9 | 8 | 5 |

### 3.2 AutomationBench (excl. marketing/finance; Revision 41, ITP-I 47, Beam 46)

| help mechanism | Revision | ITP-I | Beam |
|---|---:|---:|---:|
| **corrected write arguments** | **14** | **16** | **24** |
| **removed spurious write** | **14** | **12** | **12** |
| added a missing write | 8 | 7 | 4 |
| completed write batch | 2 | 7 | 4 |
| reordered writes | 2 | 2 | 1 |
| acted where baseline never wrote | 0 | 2 | 0 |
| changed exploration only | 1 | 1 | 1 |

### 3.3 EnterpriseOps-Gym (80 tasks; Revision 2, ITP-I 9, Beam 2)

| help mechanism | Revision | ITP-I | Beam |
|---|---:|---:|---:|
| corrected write arguments | 1 | **5** | 1 |
| completed write batch | 0 | 2 | 1 |
| reordered writes | 0 | 1 | 0 |
| added a missing write | 1 | 0 | 0 |
| changed exploration only | 0 | 1 | 0 |

Verifier-level: ITP-I repaired **17** deterministic verifiers and broke 10; Revision 8 / 15;
Beam 5 / 3.

### 3.4 Terminal-Bench-2.0 (89 tasks; Revision 5, ITP-I 10, Beam 7) — coarse labels

| help pattern | Revision | ITP-I | Beam |
|---|---:|---:|---:|
| baseline crashed on unparseable LLM output; WM run clean | 2 | 3 | 3 |
| different command strategy, similar length | 2 | 3 | 2 |
| solved with fewer commands | 1 | 2 | 1 |
| persisted longer (more commands) | 0 | 2 | 1 |

### 3.5 CRMArena-Pro (428 tasks; gained: Revision 19, ITP-I 17, Beam 19) — by task category

| category (fail → pass) | Revision | ITP-I | Beam |
|---|---:|---:|---:|
| sales_amount_understanding | 4 | 4 | 3 |
| conversion_rate_comprehension | 4 | 3 | 3 |
| case_routing | 3 | 1 | 4 |
| policy_violation_identification | 2 | 3 | 2 |
| monthly_trend_analysis | 1 | 2 | 2 |
| sales_insight_mining | 1 | 2 | 1 |
| invalid_config | 1 | 1 | 2 |
| other | 3 | 1 | 2 |

Gains concentrate in **quantitative aggregation** categories (amounts, conversion rates,
trends) and **routing**, i.e. tasks that fail on a wrongly-composed query rather than on
retrieval.

### 3.6 Synthesis — how it helps

Two mechanisms account for most of the benefit everywhere tool traces are available:

1. **Argument correction on an already-chosen mutation** — in most cases downstream of a changed or added read (§3.8). The dominant help mode on WorkBench
   (35/34/21), on AutomationBench (14/16/24) and for ITP-I on EnterpriseOps (5 of 9). The
   agent selects the right tool; the world-model run fills or fixes a field. Worked example
   (EnterpriseOps `teams`, failed in 3/3 baseline runs):
   ```
   create_group.members_odata_bind
     baseline: [bob.smith, dave.brown]
     WM      : [james.wilson, bob.smith, dave.brown]      <- owner was missing
   ```
2. **Completing the action set** — finishing a write batch, adding a missing write, or acting
   at all where the baseline only read. Combined: WorkBench 31/31/22, AutomationBench 10/16/8,
   EnterpriseOps 1/2/1. Worked example (WorkBench, "delete all emails from amir, last 6 days"):
   baseline 1 delete → WM 3 deletes, ground truth 3, side-effect flag `True → False`.

A third mode — **removing a spurious write** — matters only on AutomationBench (14/12/12),
where the baseline frequently over-acts.

---

## 3.7 One worked example per help class

Each example pairs the baseline run and the world-model run on the **same task**; only the
differing calls are shown. Task ids refer to the runs listed in §8.

### (a) Corrected write arguments — WorkBench `email_0022`, Revision, 0.0 → 1.0

> "send an email to dmitri saying 'Hey dmitri, …' and title it 'Update on performance evaluation'"

```
baseline:  email.send_email(recipient="dmitri@example.com", subject=…, body=…)        # 1 call, guessed address
WM:        company_directory.find_email_address(name="dmitri")
           email.send_email(recipient="dmitri.ivanov@atlas.com", subject=…, body=…)   # matches ground truth
```
Subject and body are byte-identical; only `recipient` changed, and `unwanted_side_effects`
flips `True → False`. The WM's own record at iteration 0 predicted the *wrong* call would
succeed (`execution_status: success`, `risk_signal: none`, not vetoed) — the fix came from the
agent inserting a directory lookup after being prompted to reflect, not from the world model
detecting the error. Same pattern on EnterpriseOps `teams`: `create_group.members_odata_bind`
gained the owner `james.wilson` that all three baseline runs omitted.

### (b) Completed a write batch — EnterpriseOps `…4a23b9c9_9d41f343` (`drive`), ITP-I, 0 → 1

> notify Wang Jun of the update, ask for re-review, and grant permission …

```
baseline writes:  create_permission ×1, create_replies ×1
WM writes:        create_permission ×2, create_replies ×1
```
Verifier repaired: **"Check if permissions has been granted"**. On WorkBench the same class is
the "delete all emails from amir" family: baseline 1 delete → WM 3 deletes, ground truth 3.

### (c) Added a missing write — AutomationBench `sales.full_sales_cycle_orchestrator`, Beam, 0.25 → 0.75

```
baseline writes:  salesforce_opportunity_update, docusign_create_envelope_from_template,
                  google_calendar_create_detailed_event, chatgpt_send_prompt, slack_send_channel_message
WM writes:        … the same five, plus
                  zoom_create_meeting(topic="TechVentures - Technical Q&A", duration=60,
                                      start_time="2026-01-16T15:00:00Z", host_email=…)
```
Scored assertions 2/8 → 6/8. The baseline scheduled the calendar event but never created the
meeting it pointed to.

### (d) Acted where the baseline never wrote — WorkBench `email_0041`, ITP-I, 0.0 → 1.0

> "forward all the emails from anaya last week about 'Update on Board of Directors Conclave' to nadia"

```
baseline:  find_email_address(Anaya); find_email_address(Nadia); search_emails ×2      # 4 reads, no write
WM:        find_email_address(nadia); search_emails ×3
           email.forward_email(email_id="00000120", recipient="nadia.moreau@atlas.com")
           email.forward_email(email_id="00000346", recipient="nadia.moreau@atlas.com")   # = ground truth
```

### (e) Removed a spurious write — AutomationBench `sales.negative_selection`, Revision, 0.46 → 1.00

```
baseline writes:  salesforce_contact_add_to_campaign ×12
WM writes:        salesforce_contact_add_to_campaign ×6
```
Scored assertions 6/13 → 7/7. The task is a *negative-selection* filter; the baseline added
six contacts it should have excluded, each of which broke a pre-satisfied assertion (upstream's
"free assertion" rule turns those into scored failures). This class is the second-largest help
mode on AutomationBench (14/12/12) and negligible elsewhere.

### (f) Reordered writes — EnterpriseOps `…99ba2325_1ff9b6e5` (`csm`), ITP-I, 0 → 1

> straighten out Russell, Johnson and Clark's premium-support contract … and move the user's location

```
baseline writes:  update_contract, update_contract, update_user_details
WM writes:        update_contract, update_user_details, update_contract
```
Verifier repaired: **"Verify User location"** — identical tool multiset, one write moved earlier.

### (g) Baseline crashed, world-model run clean — Terminal-Bench-2.0 `crack-7z-hash`, ITP-I, fail → pass

```
baseline:  30 commands, then executor failure —
           "could not parse a JSON object from: '{"kind": "exec_request", "command": "cd /app && 7z t secrets.7z -p1234567890…"
WM:        17 commands: ls -la /app/ ; 7z l /app/secrets.7z ; apt-get install -y p7zip-full ; …  → pass
```
The baseline's model emitted a password literal long enough to break its own JSON envelope.
3 of ITP-I's 10 Terminal-Bench gains and 3 of Beam's 7 are this pattern.

### (h) Category gain — CRMArena-Pro task 795 (`sales_amount_understanding`), Beam, 0 → 1

> "During the 2020 Winter season, which representative achieved the maximum total sales? Return only the Id of the agent."

```
baseline answer:  CompanySignedDate          # a field name, not an Id
WM answer:        005Wt000003NJ53IAG         # correct agent Id
```
Gains on CRMArena-Pro concentrate in quantitative aggregation (`sales_amount_understanding`,
`conversion_rate_comprehension`) and routing — tasks that fail on a wrongly-composed query
rather than on retrieval.

---

## 3.8 Is the help really about mutations? Mostly no — it enters through reads

The labels in §3.1–§3.3 are defined on *mutating* calls (create / update / delete / send …
— "write" in earlier drafts meant any of these). That choice is forced by how the benchmarks
score: they inspect final state and side effects, so a read can never change a score except
by changing a downstream mutation. Consequently **100 % of improved tasks show a mutation
difference by construction**, and a mutation-only taxonomy will always *look* as if the help
targets writes. Re-classifying every improved task on a read × mutation grid shows where the
behavioural change actually originates:

| benchmark / harness | improved | reads changed | of which: different query | added read tool | more / fewer reads | reads identical |
|---|---:|---:|---:|---:|---:|---:|
| WorkBench Revision | 79 | **53 (67 %)** | 26 | 10 | 9 / 8 | 26 |
| WorkBench ITP-I | 75 | **64 (85 %)** | 34 | 10 | 10 / 10 | 11 |
| WorkBench Beam | 49 | 25 (51 %) | 11 | 2 | 9 / 3 | 24 |
| AutomationBench Revision | 41 | **38 (93 %)** | 4 | 11 | 8 / 15 | 3 |
| AutomationBench ITP-I | 47 | **46 (98 %)** | 4 | 23 | 3 / 16 | 1 |
| AutomationBench Beam | 46 | **40 (87 %)** | 3 | 20 | 4 / 13 | 6 |
| EnterpriseOps ITP-I | 7 | **7 (100 %)** | 2 | 2 | 1 / 2 | 0 |

The single most common joint cell on WorkBench for Revision and ITP-I is **"different read
query" × "same mutations, different arguments"** (19 and 23 tasks): the world-model run issues
a different search, retrieves a different record, and the corrected id or address in the
write is downstream of that. On AutomationBench the dominant read change is **adding a new
read tool** (11 / 23 / 20) — a lookup the baseline never made — or **dropping redundant
reads** (15 / 16 / 13).

So the honest statement is: **the world model's help is predominantly a grounding effect on
the read side**, which the scoring rules make visible only through the mutations it corrects.
"Corrected mutation arguments" describes the *symptom*; "issued a different or additional
lookup before acting" describes the *mechanism* in two-thirds to all of the cases. The
`email_0022` trace in §3.7(a) is the canonical instance — the added `find_email_address`
read is the change; the fixed `recipient` is its consequence.

Two qualifications. First, the mutation-level split is not empty: on WorkBench 26 / 11 / 24
improved tasks have byte-identical reads and differ only in the mutation (mostly Beam's
"more of the same mutation" batch completions, 11 of its 24), so batch completion is a genuine
mutation-side mechanism, while argument correction is largely read-driven. Second, "reads
changed" includes *fewer* reads (WorkBench 8 / 10 / 3; AutomationBench 15 / 16 / 13): part of
the help is the agent stopping redundant exploration and acting, which is the same
continuation pressure seen elsewhere rather than better grounding.

An earlier version of this document mis-classified WorkBench's `analytics.*_count` reads as
mutations (139 calls). Correcting this moved counts mainly from "added a missing mutation"
(7/6/6 → 3/1/2) to "acted where baseline only read" (9/12/4 → 13/17/8) and left the argument
and batch rows essentially unchanged (35/34/21 → 35/35/21; 15/14/12 → 15/13/12).

---

## 4. Classification of how the world model **hurts**

Regressed tasks, same labelling.

### 4.1 WorkBench (Revision 29 regressed, ITP-I 32, Beam 23)

| harm mechanism | Revision | ITP-I | Beam |
|---|---:|---:|---:|
| **wrong write arguments** | **18** | 8 | 5 |
| **never acted (read loop / early stop)** | 7 | 6 | **8** |
| spurious extra write | 2 | 6 | 5 |
| changed exploration only | 1 | 6 | 1 |
| incomplete write batch | 0 | 1 | 2 |
| substituted a different write | 1 | 1 | 0 |
| no observable call difference | 0 | 4 | 2 |

### 4.2 AutomationBench (Revision 99, ITP-I 106, Beam 58)

| harm mechanism | Revision | ITP-I | Beam |
|---|---:|---:|---:|
| **wrong write arguments** | **39** | **41** | **22** |
| **never acted (read loop / early stop)** | **22** | **20** | **4** |
| incomplete write batch | 15 | 13 | 13 |
| changed exploration only | 11 | 15 | 8 |
| dropped a write | 5 | 10 | 5 |
| reordered writes (wrongly) | 1 | 4 | 0 |
| substituted / spurious / other | 6 | 3 | 6 |

### 4.3 EnterpriseOps-Gym (Revision 4, ITP-I 4, Beam 3)

| harm mechanism | Revision | ITP-I | Beam |
|---|---:|---:|---:|
| wrong write arguments | 2 | 2 | 1 |
| dropped a write | 1 | 1 | 1 |
| incomplete write batch | 1 | 0 | 1 |
| changed exploration only | 0 | 1 | 0 |

### 4.4 Terminal-Bench-2.0 (Revision 4, ITP-I 6, Beam 5)

| harm pattern | Revision | ITP-I | Beam |
|---|---:|---:|---:|
| different command strategy, similar length | 3 | 3 | 2 |
| thrashed (many more commands, no pass) | 1 | 1 | 1 |
| WM run crashed on unparseable LLM output | 0 | 1 | 1 |
| gave up early (fewer commands) | 0 | 1 | 1 |

### 4.5 CRMArena-Pro (lost: Revision 17, ITP-I 20, Beam 11) — by task category

| category (pass → fail) | Revision | ITP-I | Beam |
|---|---:|---:|---:|
| sales_amount_understanding | 2 | 3 | 1 |
| monthly_trend_analysis | 3 | 2 | 1 |
| top_issue_identification | 2 | 3 | 0 |
| **internal_operation_data** (privacy-refusal task) | **2** | **2** | **1** |
| invalid_config | 1 | 2 | 2 |
| best_region_identification | 2 | 2 | 0 |
| other | 5 | 6 | 6 |

`internal_operation_data` is one of CRMArena-Pro's three *privacy-rejection* categories: the
correct behaviour is to refuse. It appears in the loss column for every harness and never in
the gain column — the world model's continuation pressure pushes the agent to answer where it
should decline.

### 4.6 Synthesis — how it hurts

Harm is **two-modal**, and this revises the one-run EnterpriseOps deep dive (§7) in which all
five regressions were "dropped a write":

1. **Wrong write arguments** — the single largest harm mode wherever tool traces exist:
   AutomationBench 39/41/22, WorkBench 18/8/5, EnterpriseOps 2/2/1. This is the mirror image
   of help mode 1: the world model perturbs the arguments of a correctly chosen write, and the
   perturbation is as likely to break a field as to fix one. On WorkBench under Revision the
   ratio is 35 fixed : 18 broken; on AutomationBench it inverts to 14 : 39.
2. **Omission** — never acting, stopping a batch short, or dropping a write. Combined:
   AutomationBench 42/43/**22**, WorkBench 7/7/10, EnterpriseOps 2/1/2. Worked example
   (AutomationBench `marketing_webinar_cancellation_workflow`, 0.96 → 0.00): baseline issued
   9 `gmail_send_email` + 8 `hubspot_update_contact`; the WM run issued 34 identical
   `google_sheets_get_spreadsheet_by_id` reads and no write.

A residual **"changed exploration only" / "no observable difference"** bucket (AutomationBench
14/15/10, WorkBench 1/10/3) has identical writes and a different score; on AutomationBench,
whose same-config repeats move 42 % of task scores, this bucket is best read as noise.

---

## 5. Why Beam search is the most robust harness — and why ITP-I wins where it wins

The harm tables explain the main-table ordering better than the help tables do.

- **Beam search suppresses omission.** On AutomationBench "never acted" falls from 22/20
  (Revision/ITP-I) to **4**, and total omission from 42/43 to 22; total regressions fall from
  99/106 to 58 while improvements stay flat (41/47/46). Planning does not help more — it
  *hurts less*, specifically by keeping the agent acting. This is the mechanism behind its
  length-robustness: binned by baseline tool calls, Beam's deficit is flat with length
  (r = −0.010) while Revision degrades fastest (r = −0.107); on 35+ call tasks Revision is
  −0.123 against Beam −0.024.
- **ITP-I wins on EnterpriseOps and Terminal-Bench** through argument correction (5 of 9
  EnterpriseOps improvements; 17 verifiers repaired vs 10 broken) on tasks whose errors are
  local — a wrong field, a wrong flag — rather than omissions. On Terminal-Bench 3 of its 10
  improvements are tasks where the baseline crashed on unparseable model output and the
  advised run did not.
- **Revision wins on WorkBench**, the one-decisive-write benchmark, where argument correction
  is the whole game (35 of 79 improvements) and there is little batch to omit.

The selection criterion that fits is the **locality of the baseline's errors**: single
decisive action → Revision; local/argument-level errors, any task length → ITP-I; errors that
are omissions across a multi-write batch → Beam search.

---

## 6. Why the effect is bounded: the world model's predictions carry little signal

Matching 4,985 critic predictions on AutomationBench against realised tool outcomes: predicted
success 98.3 % vs 83.1 % realised; 5 of 842 genuine failures detected (1 % recall, 6 %
precision); **AUC 0.597**. The configured `failure_prob` threshold of 0.55 was crossed 5 times
in 2,221 checks, so critic-triggered planning degenerates into a fixed-interval timer (which is
why it is omitted from the main table). Plan content is comparably weak: the imagined plan's
first step matches the agent's own next action 14–24 % of the time on WorkBench, and 481 of
708 plan-tail actions target an application unrelated to the task.

Read together with §3–§4, the picture is that the world model contributes *continuation
pressure* — an injected block exhibiting further pending calls plus an advisory not to answer
yet — rather than model-based search. That pressure completes batches and nudges arguments,
and it also pushes the agent past refusal points (§4.5) and, when the injected plan diverges,
off task.

---

## 7. Verifier-level deep dive (EnterpriseOps-Gym, one Beam run)

For the Beam run `rn-20260916T223637Z` (task success 0.4125): 352 verifier observations,
95.5 % deterministic under baseline repetition; **repaired 9 of 96 reproducible failures
(9.4 %), broke 7 of 240 reproducible successes (2.9 %)**. Repairs were heterogeneous (2 argument
corrections, 1 reorder, 1 added write, 1 extended batch, 1 removed write); all 5 regressions
were omitted writes. §4 shows this run under-represents the argument-error harm mode that
dominates across the larger samples.

---

## 8. Runs used for mechanism attribution

| benchmark | baseline | Revision | ITP-I | Beam search |
|---|---|---|---|---|
| WorkBench | `rn-20260831T051431Z` (all three baseline runs identical) | `rn-20260915T081405Z` | `rn-20260915T004110Z` | `rn-20260915T133146Z` |
| EnterpriseOps-Gym | mean of `073723Z`, `081439Z`, `085200Z` | `rn-20260916T203208Z` | `rn-20260916T193551Z` | `rn-20260916T204707Z` |
| AutomationBench | `rn-20260915T220314Z` | `rn-20260916T053837Z` | `rn-20260915T205054Z` | `rn-20260916T160221Z` (h4/e4) |
| Terminal-Bench-2.0 | mean of `154246Z`, `204917Z`, `030835Z` | `rn-20260912T043238Z` | `rn-20260910T184303Z` | `rn-20260911T071601Z` |
| CRMArena-Pro | `rn-20260915T034327Z` | `rn-20260916T155707Z` | `rn-20260916T043338Z` | `rn-20260916T161926Z` |

Excluded from all averages: one EnterpriseOps ITP-I run that returned 0.0000 (crashed) and one
WorkBench 100-task sweep in which every harness returned exactly the baseline score (world
model never engaged).

---

## 9. Threats to validity

1. AutomationBench and CRMArena-Pro have one run per harness against one baseline; their noise
   floors (42 % task churn; ±0.5–1.9 pp) are estimated from same-config *world-model* repeats,
   not baseline repeats.
1b. **WorkBench scores fall as `max_parallel` rises** (found 2026-09-18). Four otherwise
   identical Beam-search runs on the 690-task target: 0.8188 (mp=1), 0.7928 (mp=3),
   0.7855 (mp=10), 0.7725 (mp=20, LLM-WM). Paired per task the shift is directional, not
   churn: 28 improved / 46 worsened (mp=1→3), 25/48 (→10), 26/58 (→20), while total churn
   stays at 10–13 % in every pair, including between two mp=1 runs. It is *not* sandbox
   leakage (each task runs in its own `asyncio.to_thread` worker over WorkBench's
   `threading.local()` state with `reset_state()` per task, and `has_side_effects` is
   recomputed post hoc by re-executing the predicted actions against a fresh state), and not
   timeouts (visible executor errors rise only 2→5 tasks). The extra failures are ordinary
   wrong answers -- "incorrect; unwanted side effects" 94→104→108→116 -- at constant tool
   calls per task (3.19–3.24), i.e. the agent makes different and slightly worse choices when
   its LLM calls share a busier server, which is expected given that LLM inference is not
   batch-invariant and Beam search samples eight candidate plans per replan at temperature
   0.7. (The no-world-model baseline at mp=1 is exactly reproducible, 0.7652 three times, so
   the stochasticity enters with the world-model harness.) Consequence: all numbers compared
   in one table must come from one `max_parallel`, and the prediction-ablation controls are
   run at their comparator's `max_parallel`. Caveat: those four runs were sequential on a
   shared policy server, so part of the trend could be external load; a back-to-back mp=1 vs
   mp=10 pair would separate the two.
2. Mechanism labels are heuristic (write/read split by tool-name prefix; shell commands by
   regex). They are reliable for the dominant buckets and coarse for Terminal-Bench.
3. AutomationBench `hr` is a measurement defect (99/100 tasks fail on unresolvable identifiers
   because `initial_state` is never shown to the agent); it is retained in AB here since the
   table excludes only marketing and finance.
4. CRMArena-Pro ITP-I run A had 18 tasks unevaluated by the upstream grader (300 s timeouts);
   the table's ITP-I value uses run B, which has none.
5. One-run mechanism counts on EnterpriseOps (≤ 9 improved tasks per harness) are directional,
   not statistical.

---

## 10. Prediction-ablation controls ("is the planner doing the work?")

Reviewer question: Eq. 6's eight weights, hand-set utilities, read-pressure rule and two veto
thresholds are a large hand-engineered prior, and Table 4's help modes (argument correction,
fewer premature stops) are behaviours a read penalty plus a coverage bonus might induce alone.
Test: run the **identical** planner -- same candidate sampling, same utilities, same vetoes,
same refinement -- with only the world model's per-step canonical-event distributions (and
terminal probabilities) replaced. Implementation: `WM_JEPA_PREDICTION_CONTROL` in
`src/ejepa_wm/backends/_ewm_jepa.py` (`shuffled` = the model's own rows permuted across
(plan, step) within a call, so marginals are preserved and the per-step assignment is
destroyed; `uniform` = 1/K per field; `prior` = training-set class priors from
`results/analysis/canonical_event_class_priors.json`, no input conditioning).

### 10.1 WorkBench (690 tasks, beam_interval s=8 h=3 e=2 open, mp=10, Qwen3.6-27B)

All five arms were run on 2026-09-18 against the same endpoint at the same parallelism.

| arm | score | zero-score tasks | paired vs real (improved/worsened) |
|---|--:|--:|---|
| real JEPA predictions      | 0.7986 | 139 | -- |
| uniform (1/K per field)    | 0.7957 | 141 | 22 / 24 |
| **no world model**         | 0.7942 | 142 | 20 / 23 |
| prior (class priors)       | 0.7913 | 144 | 21 / 26 |
| shuffled (marginals kept)  | 0.7826 | 150 | 20 / 31 |

The whole five-way spread is 1.6 pp; every paired comparison is symmetric (sign tests
p >= 0.17); and two runs of the *same* real-prediction configuration at mp=10 on different
days/endpoints differ by 1.3 pp (0.7986 today, 0.7855 on 09-17), i.e. by as much as the
largest effect in the table. So at this parallelism WorkBench shows **no measurable
world-model benefit at all** (+0.4 pp over no world model), and no measurable sensitivity to
the *content* of the predictions. What survives is the harness: eight policy-sampled candidate
plans, a refinement prompt, and the planner's own priors.

Two caveats bound the claim rather than rescue it. (i) The earlier same-model ladder suggests
parallelism suppresses the gain (real predictions: 0.8188 at mp=1, 0.7928 at mp=3, 0.7855 at
mp=10 -- see threat 1b), so a mp=1 replication could still show a gain; that needs a mp=1
baseline, which costs ~9 h and does not exist yet. (ii) The historical mp=1 baseline in the
main table (0.7652, three exactly reproducible runs) was produced with agent alias
`wm_agent3`, while the world-model runs used `wm_agent1`/`Qwen3.6-27B`; no run summary records
the underlying checkpoint, so the +5.4 pp mp=1 "gain" compares two agents that cannot now be
verified to be the same model. The mp=10 set above is the only internally consistent
WorkBench comparison we have.

### 10.2 AutomationBench (strict pass rate, beam_interval s=8 h=3 e=2)

Two domains complete. Every control matches or beats the real predictions, on both the
strict metric and partial credit, and the ordering of the two metrics disagrees -- the
signature of noise rather than signal.

| arm | sales pass | sales partial | operations pass | operations partial |
|---|--:|--:|--:|--:|
| no world model      | 0.19 | -- | 0.28 | -- |
| real JEPA           | 0.22 | -- | 0.31 | 0.6844 |
| shuffled            | 0.23 | 0.5804 | 0.32 | 0.6997 |
| uniform             | 0.19 | 0.5889 | 0.31 | 0.6944 |
| prior               | 0.19 | 0.5794 | **0.34** | 0.7039 |

On operations -- the domain with a parallelism-matched real-prediction run -- all three
controls land at or above real predictions, and the best arm in the whole set is the one
whose "predictions" are input-independent training-set class priors. AutomationBench's
pass-rate noise floor is +-4 pp, so the honest reading is that nothing here is
distinguishable, which is itself the result: the predictions' content is not what moves
the score. The support domain was still running at the time of writing.

### 10.3 EnterpriseOps-Gym

Shuffled (80 tasks, mp=3): 0.3625, against real predictions spanning 0.338-0.400 over
three repeats and a no-world-model baseline of 0.375 -- indistinguishable again, on a
benchmark whose repeat churn is 5 %. The remaining arms were lost to an endpoint outage
and are being re-run.

---

## Appendix A: `\paragraph` (help, with brief hurt sentences) + help-classification table for §4.2

Requires `\usepackage{tabularx}` and `\usepackage{booktabs}`.

```latex
\paragraph{How the world model helps.}
To see \emph{how} Enterprise-JEPA improves outcomes we diff the tool calls of every task the world-model run solves better than the baseline (Table~\ref{tab:help}); WorkBench and EnterpriseOps-Gym baselines are near-deterministic (0/100 and 2/80 outcomes change over three repeats), so differences there are attributable. The world model never actuates---across 664 EnterpriseOps decisions no override fired, its only channel being advisory text on 19\% of steps---so every mechanism is the agent acting differently under advice. Because the benchmarks score only final state, an improvement must surface as a changed mutation, and Table~\ref{tab:help} is organised by that mutation-side symptom: \emph{argument correction} of an already-chosen write (35/35/21 of WorkBench's 79/75/49 improvements for Revision/ITP-I/Beam search; 14/16/24 on AutomationBench) and \emph{completing the action set}---finishing a batch, adding a missing mutation, or acting where the baseline only searched (31/31/22 on WorkBench, 10/16/8 on AutomationBench)---dominate, with \emph{removing spurious mutations} specific to AutomationBench's over-acting baseline (14/12/12). The mechanism behind these symptoms, however, lies largely on the \emph{read} side: the read calls also differ in 67/85/51\% of WorkBench improvements and 93/98/87\% of AutomationBench improvements, and the single most common pattern is a different search query whose retrieved record supplies the corrected id or address (19/23/11 WorkBench tasks). Under advice the agent grounds before it acts---issuing a lookup it had skipped, rewriting a query, or ceasing redundant exploration---and the corrected mutation inherits the result; the canonical trace (Appendix~\ref{app:examples}) shows the world model itself predicting that the \emph{wrong} \texttt{send\_email} would succeed, the fix arising from a directory lookup the prompted agent inserted. Only batch completion is a predominantly mutation-side effect (on WorkBench 26/11/24 improvements have byte-identical reads, mostly Beam's batch completions). The help modes are thus best read as grounding and continuation pressure induced by reflection prompts rather than as model-based search over actions. The same channel explains the entries of Table~\ref{tab:agent} that fall below the baseline: the regressed tasks show the mirror image of the dominant help mode---an argument the baseline had guessed correctly is perturbed under advice (WorkBench 18/8/5 of 29/32/23 regressions, AutomationBench 39/41/22 of 99/106/58)---or an omission in which the agent keeps reading and stops short of a write. The balance favours the world model wherever a lookup can ground the argument (on WorkBench, Revision fixes 35 arguments for every 18 it breaks) and turns against it only where identifiers are opaque to every available tool, as on AutomationBench (14:39); and the costliest omission, never acting at all, largely disappears under horizon-level planning (AutomationBench 22/20 never-acted regressions under the per-step harnesses, 4 under Beam search). Since the world model never overrides an action, both gains and losses flow through advice the agent elects to follow, which points to a direct remedy---gating advice on the critic's confidence and confining it to grounding and completion suggestions where arguments cannot be verified---rather than to a limit of the approach.

\begin{table*}[t]
\centering
\small
\caption{How Enterprise-JEPA helps: improved tasks by the mutation-side mechanism that changed the score, as Revision\,/\,ITP-I\,/\,Beam search counts (WB: 79/75/49 improved of 690; AB, excluding marketing and finance: 41/47/46; EOPS: 2/9/2 of 80). Because benchmarks score only final state, every improvement surfaces as a mutation difference; the footer reports how often the \emph{read} calls also changed, which is where most of the behavioural change originates. Worked examples in Appendix~\ref{app:examples}.}
\label{tab:help}
\begin{tabularx}{\textwidth}{@{}l rrr X@{}}
\toprule
Help mechanism & WB & AB & EOPS & Description \\
\midrule
Corrected mutation arguments & 35/35/21 & 14/16/24 & 1/5/1 & Same mutating tool and order, but an argument the baseline had guessed (an id, address or field value) now matches the requirement---usually because a different or additional lookup preceded it. \\
\addlinespace
Completed write batch & 15/13/12 & 2/7/4 & 0/2/1 & Same mutating tools but more calls of them, finishing a ``do X to \emph{all} matching items'' instruction the baseline left partly done. \\
\addlinespace
Acted where baseline only read & 13/17/8 & 0/2/0 & -- & The baseline searched and looked up but terminated without any mutation; the world-model run carries the task through to the required writes. \\
\addlinespace
Removed spurious mutation & 4/1/1 & 14/12/12 & -- & Fewer mutations with the same tools; the removed calls violated a constraint (e.g.\ acting on items a filter should have excluded). \\
\addlinespace
Added a missing mutation & 3/1/2 & 8/7/4 & 1/0/0 & A mutating tool the baseline never invoked appears, and the corresponding verifier or assertion flips to pass. \\
\addlinespace
Reordered mutations & -- & 2/2/1 & 0/1/0 & Identical multiset of mutations in a different order, satisfying a dependency between two updates. \\
\midrule
\multicolumn{5}{@{}p{\textwidth}@{}}{\textbf{Read calls also changed} in 67/85/51\% of WB improvements, 93/98/87\% of AB, and 100\% of EOPS ITP-I; the most frequent WB pattern is a different search query feeding a corrected mutation argument (19/23/11). \textbf{TB} (coarse): 3/3/3 of 5/10/7 gains are tasks whose baseline crashed on unparseable model output. \textbf{CRM} (by category): gains concentrate in quantitative aggregation and routing.} \\
\bottomrule
\end{tabularx}
\end{table*}
```


## Appendix B: worked examples (paper appendix section)

```latex

\section{Worked examples of each help mechanism}
\label{app:examples}
Each example pairs the baseline and world-model runs on the same task; only the differing calls are shown.

\paragraph{Corrected write arguments (WorkBench \texttt{email\_0022}, Revision, 0.0$\rightarrow$1.0).}
Task: ``send an email to dmitri saying `Hey dmitri, \ldots' and title it `Update on performance evaluation'\,''. Baseline: a single \texttt{send\_email(recipient="dmitri@example.com", \ldots)} to a guessed address. World model: \texttt{find\_email\_address("dmitri")} followed by \texttt{send\_email(recipient="dmitri.ivanov@atlas.com", \ldots)}; subject and body are byte-identical and the side-effect flag clears. The world model's own record predicted the wrong call would succeed (\texttt{execution\_status: success}, not vetoed); the fix came from the agent inserting a lookup after being prompted to reflect. The same class on EnterpriseOps: \texttt{create\_group.members\_odata\_bind} gains the owner \texttt{james.wilson} omitted in all three baseline runs.

\paragraph{Completed write batch (EnterpriseOps \emph{drive} task, ITP-I, 0$\rightarrow$1).}
Baseline writes \texttt{create\_permission}$\times$1, \texttt{create\_replies}$\times$1; world model \texttt{create\_permission}$\times$2, \texttt{create\_replies}$\times$1. Verifier repaired: ``Check if permissions has been granted''. On WorkBench the same class completes ``delete all emails from amir, last 6 days'': 1 delete $\rightarrow$ 3 (ground truth 3).

\paragraph{Added a missing write (AutomationBench \texttt{sales.full\_sales\_cycle\_orchestrator}, Beam search, 0.25$\rightarrow$0.75).}
The baseline schedules the calendar event but never creates the meeting it refers to. The world-model run issues the same five writes plus \texttt{zoom\_create\_meeting(topic="TechVentures - Technical Q\&A", duration=60, start\_time="2026-01-16T15:00:00Z", \ldots)}; scored assertions 2/8 $\rightarrow$ 6/8.

\paragraph{Acted where the baseline only read (WorkBench \texttt{email\_0041}, ITP-I, 0.0$\rightarrow$1.0).}
Task: ``forward all the emails from anaya last week about `Update on Board of Directors Conclave' to nadia''. Baseline: two directory lookups and two searches, then termination with no write. World model: \texttt{forward\_email(email\_id="00000120", recipient="nadia.moreau@atlas.com")} and \texttt{forward\_email(email\_id="00000346", \ldots)}, exactly the ground truth.

\paragraph{Removed spurious write (AutomationBench \texttt{sales.negative\_selection}, Revision, 0.46$\rightarrow$1.00).}
Baseline \texttt{salesforce\_contact\_add\_to\_campaign}$\times$12; world model $\times$6. The task is a negative-selection filter: the six extra contacts each broke a pre-satisfied assertion, which the benchmark's free-assertion rule converts into scored failures (6/13 $\rightarrow$ 7/7 scored).

\paragraph{Reordered writes (EnterpriseOps \emph{csm} task, ITP-I, 0$\rightarrow$1).}
Baseline \texttt{update\_contract, update\_contract, update\_user\_details}; world model \texttt{update\_contract, update\_user\_details, update\_contract}. Identical tool multiset; the ``Verify User location'' verifier passes.

\paragraph{Baseline crash avoided (Terminal-Bench-2.0 \texttt{crack-7z-hash}, ITP-I, fail$\rightarrow$pass).}
The baseline's model emitted a password literal long enough to break its own JSON exec envelope after 30 commands (``could not parse a JSON object from \ldots''); the world-model run passed in 17 commands. Three of ITP-I's ten Terminal-Bench gains and three of Beam search's seven follow this pattern.

\paragraph{Category gain (CRMArena-Pro task 795, \texttt{sales\_amount\_understanding}, Beam search, 0$\rightarrow$1).}
Query: ``During the 2020 Winter season, which representative achieved the maximum total sales? Return only the Id of the agent.'' Baseline answer \texttt{CompanySignedDate} (a field name); world model \texttt{005Wt000003NJ53IAG}, the correct Id.
```
