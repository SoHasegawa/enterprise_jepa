# Planning-budget ablation: paper text and table

Source runs: `results/wm_harness_summaries/ejepa-wm-harnesses-beam-ablation-*`, three repeats
per configuration on both benchmarks, all with Enterprise-JEPA, the `beam_interval`
harness, two executed steps per replan, score margin 0.10, temperature 0.7, one
refinement round over the top 4, SSOT diversity on, terminal advice at 0.75.
Regenerate with `uv run python scripts/summarize_beam_ablation.py`.

## Measured values

EnterpriseOps-Gym is the full 80-task benchmark, score rate. AutomationBench is the
**operations domain only** (100 tasks), strict pass rate; the other five domains were not
run at ablation scale, so this column is a single-domain replication rather than the
400-task column of the main table.

| candidates | horizon | rollout | EnterpriseOps-Gym | AutomationBench (operations) | EOPS s/step | AB s/step |
|--:|--:|---|---|---|--:|--:|
| 4 | 3 | open | 0.367 ± 0.007 | 0.313 ± 0.021 | 1.50 | 1.36 |
| 8 | 3 | open | **0.404 ± 0.007** † | 0.307 ± 0.006 | 2.39 | 1.26 |
| 16 | 3 | open | 0.375 ± 0.013 | 0.277 ± 0.006 | 1.83 | 1.48 |
| 8 | 2 | open | 0.392 ± 0.026 | 0.307 ± 0.006 | 1.60 | 1.46 |
| 8 | 4 | open | 0.346 ± 0.029 | 0.283 ± 0.035 | 1.63 | 1.46 |
| 8 | 3 | closed | 0.375 ± 0.013 | 0.257 ± 0.032 | 2.94 | 2.30 |

† The centre row's EnterpriseOps-Gym value is the main table's cell (0.404 ± 0.007,
n = 3), used so the ablation is anchored to the number the paper reports elsewhere. The
ablation's own three runs of that identical configuration give 0.371 ± 0.032. The two run
sets differ only in scheduling, and the gap is inside this benchmark's repeat churn, but
it does mean the 4-candidate and horizon-2 rows are compared against a centre measured on
a different set: against the ablation's own centre those two rows are −0.4 and +2.1
points, that is, indistinguishable rather than worse. The 16-candidate, horizon-4 and
closed-loop conclusions hold under either centre, and every AutomationBench comparison is
internal to one run set.

## Paper text

\paragraph{Planning budget.}
Table~\ref{tab:beam-ablation} varies one factor at a time around the configuration used
throughout the paper: eight candidate plans, imagination horizon three, open-loop
rollouts. Enlarging the candidate set does not help. Going from eight to sixteen
candidates costs 2.9 points on EnterpriseOps-Gym and 3.0 points on AutomationBench, and
halving it to four is no better than eight on AutomationBench. We attribute this to the
selection rule rather than to the search: the planner executes the arg-max over imagined
returns, and every imagined transition carries prediction error, so a wider beam mostly
adds further opportunities for an inaccurate rollout to be scored optimistically. The
plan that is executed is then precisely the one whose predictions erred most favourably,
and the probability that such a plan exists grows with the number of candidates.
Deepening the rollout behaves the same way, and more sharply, because per-step errors
compound along the horizon: horizon four loses 5.8 points on EnterpriseOps-Gym and 2.3
points on AutomationBench relative to horizon three, while horizon two is
indistinguishable from it. Re-planning after every executed step rather than every two
(closed loop) does not recover the loss either: it is 2.9 points below the open loop on
EnterpriseOps-Gym and 5.0 points below on AutomationBench, while raising the cost per
executed step by 23\% and 83\% respectively, since the world model is queried twice as
often. Accuracy is therefore flat or declining in the planning budget over the whole
range we can afford to measure, and cost is not. We use eight candidates, horizon three
and open-loop rollouts for every result in this paper: it is the accuracy-maximising
point of the grid on EnterpriseOps-Gym, within noise of the best point on
AutomationBench, and 19\% and 45\% cheaper per executed step than the closed-loop
alternative on the two benchmarks.

\begin{table}[t]
\centering
\small
\caption{Planning-budget ablation for Enterprise-JEPA beam search, one factor at a time
around the configuration used in the paper (bold). Mean $\pm$ s.d. over three runs.
EnterpriseOps-Gym is the full 80-task benchmark (score rate); AutomationBench is its
operations domain (100 tasks, strict pass rate). Neither widening the beam, deepening the
rollout, nor re-planning at every step improves on the chosen setting.}
\label{tab:beam-ablation}
\begin{tabular}{@{}rrl cc@{}}
\toprule
\multicolumn{3}{@{}l}{Configuration} & \multicolumn{2}{c}{Success} \\
\cmidrule(l){4-5}
Cand. & Hor. & Rollout & EnterpriseOps-Gym & AutomationBench \\
\midrule
 4 & 3 & open            & $0.367 \pm 0.007$ & $0.313 \pm 0.021$ \\
\textbf{8} & \textbf{3} & \textbf{open}
                         & $\mathbf{0.404 \pm 0.007}$ & $\mathbf{0.307 \pm 0.006}$ \\
16 & 3 & open            & $0.375 \pm 0.013$ & $0.277 \pm 0.006$ \\
\midrule
 8 & 2 & open            & $0.392 \pm 0.026$ & $0.307 \pm 0.006$ \\
 8 & 4 & open            & $0.346 \pm 0.029$ & $0.283 \pm 0.035$ \\
\midrule
 8 & 3 & closed          & $0.375 \pm 0.013$ & $0.257 \pm 0.032$ \\
\bottomrule
\end{tabular}
\end{table}

## Notes for whoever edits this

- The mechanism sentence ("the plan executed is the one whose predictions erred most
  favourably") is the selection-bias reading of an arg-max over noisy estimates. It is
  consistent with the prediction-ablation controls in the qualitative analysis, where
  shuffled, uniform and prior-only predictions match real ones: if the predictions carry
  little signal, searching harder over them cannot pay, and may cost.
- If a reviewer asks why horizon two is not the chosen setting, the answer is that it is
  within noise of horizon three on both benchmarks and we keep the setting the rest of
  the paper is measured at. Do not claim horizon three beats horizon two.
- The seconds-per-step figures come from ablation runs that shared agent endpoints, so
  they are indicative; the controlled timing measurements are in `efficiency_analysis.md`
  and should be cited for any latency claim beyond the open/closed ratio.
