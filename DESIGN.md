# Design notes: three modes, three rewards

The passive mode is the familiar ARC setup. The other two exist because of two findings from the first
round of results and two things the ARC-AGI-3 technical report gets right.

## What the first round showed

Every model solved 100% of single-rule tasks. Only composed rules separated models (Opus 77% vs Haiku
27% on 3-rule chains). Reading the failed transcripts, models fail by misidentifying the rule, not by
executing it badly. So the thing worth measuring is *rule identification under uncertainty*, and the
passive mode only measures it indirectly.

## Active mode (probe the rule)

ARC-AGI-3 moved from static pairs to interactive environments where the agent is never told the
objective and is scored on action efficiency relative to a human: per level, `min(1, h/a)^2` where
`a` is the agent's action count and `h` a human baseline, so using 2x the actions costs 75% of the
score. The point is to measure *how efficiently you discover the rule*, not whether you eventually do.

Our active mode borrows that directly. The agent sees one example pair instead of three, and can design
probe inputs; the hidden rule is applied and the result returned. Reward:

    reward = solved x min(1, ref / (probes + submissions))^2,   ref = 4 (what passive mode gets for free)

A well-designed probe (one pixel, two colors, a single object) isolates one hypothesis. A model that
probes with random clutter, or probes six times when two would do, is penalised quadratically. The
summary reports mean probes used on solved tasks and the share solved with <=2 probes. This is an
experiment-design test dressed as a puzzle, which is why it sits next to the autoresearch env.

## Program mode (write the rule)

The agent writes `transform(grid)` in Python. It is run on the visible examples (results shown) and
on five hidden inputs sampled from the same generator with the same parameters (only the pass count
shown). Reward is the hidden pass fraction of the *final* submission, decayed per attempt:

    reward = hidden_pass_fraction x (1 - 0.15 x (attempt - 1))

Why this is the strictest mode: a grid answer can be pattern-completed from the examples without
understanding the rule. A program that passes five fresh inputs has to encode the rule. Procedural
generators are what make this possible; the public ARC sets have one test input per task, so you
can't measure generalisation this way on them. The split between "passes the examples" and "passes
hidden tests" is reported separately: that gap is overfitting to the examples, made visible.

Sandboxing: the code runs in a subprocess under `python -I -S` with a 3s alarm per input, a 2GB
address-space limit, an empty environment and no network. It is adequate for frontier-API models and
not adequate for an RL policy under optimisation pressure; for that you would want a container.

## Shaped vs sparse

`reward_mode=shaped` adds `0.3 x best_cell_accuracy` for unsolved passive episodes. It is off by
default. Per-cell accuracy is a results-based checkpoint (what the shaping literature recommends over
action-based ones), and the 0.3 cap keeps it from ever competing with a solve. ARC's official metric
is exact match, so this is a training-time signal, not an eval number.

## Random-agent calibration

ARC-AGI-3 requires that a random policy solve a non-tutorial level less than 1 in 10,000 times. Ours
is a weaker check but the same idea: `python -m arc_env.baselines` scores four trivial policies
(identity, copy the first example's output, fill the correct shape with the most common colour, random
fill) on 500 tasks per tier. Worst case is 6% on medium, entirely from the count-objects family whose
output is a single-colour bar; that policy is also given the correct output shape, which the agent is
not. Everything else is under 1%.

## What I'd change next

- The count-objects family should output something with more structure than a monochrome bar.
- Active mode's `ref=4` is a stand-in for a human baseline. The right thing is to measure how many
  probes a human needs per family and use that, as ARC-AGI-3 does.
- Program mode could accept a DSL instead of Python and score program length (the ARC-AGI-3 novelty
  criterion is about shortest-program overlap). Shorter correct programs are better rule statements.
- For training rather than eval: the env is already reset/step shaped and all three modes expose a
  dense enough signal (cell accuracy, probe count, hidden pass fraction) for GRPO against the local
  Gemma server.

## Sources

- ARC-AGI-3 technical report (arXiv 2603.24621): RHAE scoring, interactive format, random-agent
  calibration, multiple-mechanics requirement.
- MLGym (arXiv 2502.14499): best-attempt vs best-submission, performance profiles.
- Snorkel, "RL environments for LLM agents: design, rewards, validation" (2026): results-based
  partial credit.
- Rubric-RL reward-hacking analyses (arXiv 2606.04923, 2609.16816): why there is no LLM judge here.
