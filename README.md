# arc-env

A procedurally generated ARC-AGI style environment for evaluating and training LLMs on
abstract visual reasoning. Unlike the public ARC datasets (fixed, small, and in every
pretraining corpus by now), tasks here are sampled fresh from a library of transformation
families with known solutions, so you get unlimited verifiable episodes and a difficulty knob.

It is an *environment*, not a benchmark: the model gets per-cell feedback and multiple attempts,
and the reward decays with attempts. That makes it directly usable for RL, and it also lets you
measure something a single-shot eval can't: how well a model uses feedback.

Three action modes share the same task generators (details in [DESIGN.md](DESIGN.md)):

- **passive**: see train pairs, submit the test output grid, get per-cell feedback. The ARC format.
- **active**: see fewer train pairs, but you may send your own input grids to the hidden rule and
  observe the outputs before answering. Reward penalises wasted probes (ARC-AGI-3 style efficiency).
- **program**: submit a Python `transform(grid)` instead of a grid. It runs in a sandbox and is
  scored on five hidden test pairs, so memorising the visible pairs scores nothing.

## What a task looks like

<p float="left">
<img src="docs/sample_easy.svg" width="30%"/> <img src="docs/sample_medium.svg" width="30%"/> <img src="docs/sample_hard.svg" width="34%"/>
</p>

The rule is only shown here for illustration; the model never sees it.

```
Example 1 input:          Example 1 output:
0 0 3 0 4 0 0             3 4 0 0 0 0 0
0 0 0 0 1 0 0             1 0 0 0 0 0 0
2 3 0 3 0 0 0             2 3 3 0 0 0 0
...
```

Hidden rule (never shown to the model, logged for analysis): `gravity:left`.

### Transformation families (15)

| family | tier | rule |
|---|---|---|
| flip_rotate | easy | flip H/V, rotate 90/180/270, transpose |
| gravity | easy | all cells fall in one direction and stack |
| recolor_map | easy | fixed color permutation |
| fill_enclosed | easy | fill the inside of hollow shapes with a color |
| ray | easy | every colored cell shoots a line to the edge |
| mirror | easy | output = input glued to its mirror image |
| translate | medium | shift everything by a fixed (dr, dc) |
| swap_most_common | medium | swap the two most frequent colors |
| recolor_by_size | medium | singletons get one color, bigger objects another |
| hollow_out | medium | keep only the border of filled rectangles |
| bounding_box | medium | draw a box around each object |
| denoise | medium | delete isolated pixels |
| crop_largest | medium | output is the bounding box of the largest object |
| tile | medium | tile the input, optionally alternating flips |
| count_objects | medium | count objects of a color, output a 1xN bar |

**Difficulty tiers**: `easy` picks from the six easy families on 5-8 grids. `medium` uses all
15 on 7-12 grids. `hard` uses 9-15 grids and, 70% of the time, *composes two* shape-preserving
primitives (e.g. `translate then gravity`), which gives ~100 distinct rule types.

Every task is checked to be non-trivial (output differs from input, outputs aren't all identical)
and is deterministic from `(seed, difficulty)`.

## Environment API

```python
from arc_env import ArcEnv

env = ArcEnv(difficulty="medium", max_attempts=3)            # mode="passive" (default)
obs = env.reset(seed=42)        # obs["train"]: list of (input, output); obs["test_input"]
obs, reward, done, info = env.step(my_grid)
# obs["feedback"]: shape_ok, cell_acc, wrong_mask (grid of 1s where you were wrong)
# reward: 1.0 on first-try solve, 0.85 on second, 0.70 on third, else 0

env = ArcEnv(difficulty="hard", mode="active", max_probes=6, active_train_pairs=1)
obs = env.probe(my_input_grid)  # returns the hidden rule applied to your grid
# reward on solve: min(1, 4 / (probes + submissions)) ** 2

env = ArcEnv(difficulty="hard", mode="program", n_hidden_tests=5)
obs, reward, done, info = env.step("def transform(grid):\n    ...")
# reward: fraction of hidden tests passed, times the attempt decay
# reward_mode="shaped" adds up to 0.3 * cell_acc partial credit on failed attempts (any mode)
```

## Running the eval

```bash
uv venv && source .venv/bin/activate && uv pip install -e .
source ~/.secrets/anthropic.env           # for Claude models
python -m arc_env.run --models gemma-4-31b,haiku-5.5,sonnet-5.5 --episodes 30
python -m arc_env.run --mode program --difficulties hard,expert --episodes 15 --max-cost 3.5
python -m arc_env.run --mode active  --difficulties hard,expert --episodes 15 --max-cost 3
python -m arc_env.run --summary-only      # rebuild results/summary.md from raw jsonl
python -m arc_env.baselines               # random-agent calibration table
```

Models: `gemma-4-31b` (any OpenAI-compatible endpoint, default is a local vLLM on :5151),
`haiku-5.5`, `sonnet-5.5`, `opus-5.5`, `sonnet-5.5-high`, `fable-5.1` (Anthropic API). Add more
in `arc_env/clients.py`. `--max-cost` is a per-model USD cap at list price; the run stops
dispatching new episodes once it is hit. Non-passive modes write to `results/<mode>/`.

Each episode's full transcript (prompts, model responses, feedback) is saved to
`results/raw/<model>.jsonl` so failures can be read, not just counted.

Claude calls pin `effort="medium"` with a 16k token budget and retry once at low effort if the
reply is all thinking and no answer. The passive run below predates that and used the API
defaults (Sonnet high, Opus/Haiku medium), so Sonnet's passive numbers are slightly flattered
relative to the other two modes.

## Results: passive mode

30 episodes per tier per model, 3 attempts each, single run (so treat differences under ~10
points as noise). Full tables including per-family breakdown: [results/summary.md](results/summary.md).

| model | easy | medium | hard | expert | first-try (all) | after feedback (all) | cost |
|---|---|---|---|---|---|---|---|
| Claude Opus 5.5 | 100% | 100% | 97% | 77% | 84% | 93% | $5.43 |
| Claude Fable 5.1* | 100% | 100% | 90% | 69% | 84% | 91% | $15.17 |
| Claude Sonnet 5.5 | 100% | 100% | 87% | 60% | 75% | 87% | $3.85 |
| Claude Haiku 5.5 | 100% | 100% | 63% | 27% | 62% | 72% | $0.29 |
| Gemma 4 31B (local vLLM) | 100% | 100% | 50% | 37% | 57% | 72% | $0 |

\* Fable hit the $15 per-model spend cap after 116 episodes, so its expert tier is n=26.

Things I took from this:

- **Easy and medium are saturated** for every model, including a 31B open model. Single-rule
  ARC-style tasks are no longer a test of anything. The signal is entirely in composed rules
  (hard = 2 rules, expert = 3), which is also the direction a curriculum would need to go.
- **Feedback helps the weaker models most.** Per-cell feedback plus two more attempts lifts
  Gemma by 15 points and Haiku by 10, but Fable by only 7. The env's multi-turn design makes
  this measurable; a single-shot benchmark can't see it.
- **The failure modes are legible.** Every single-rule task (easy/medium) was solved by every
  model, so the per-family numbers above 0 failures all come from composed tasks. Reading the
  failed transcripts: the models do not mis-execute a rule they stated correctly, they fail to
  identify the composition at all. Gemma confidently describes a wrong rule; Haiku says outright
  that it "couldn't pin down the rule" and submits a placeholder. Geometric primitives
  (gravity, translate, rotate) are the ones most often lost inside a composition, presumably
  because they move every cell and hide the other rule's signature.
- Cell accuracy on failed expert tasks is 0.6-0.8, so "almost right" is common; a reward with
  partial credit would be dense enough for RL.

## Results: program mode (write the rule as code)

Hard and expert tiers only, 15 episodes each, 3 attempts, Claude at `effort=medium`. The model
submits `transform(grid)`; the env runs it in a sandbox on the visible pairs *and five hidden
test pairs* sampled from the same generator. Reward is the hidden pass fraction, so a program that
hard-codes the visible outputs scores 0. Full tables: [results/program/summary.md](results/program/summary.md).

| model | hard solve | expert solve | hidden-test pass (mean) | all 5 hidden pass | first-try (all) | after feedback (all) | cost |
|---|---|---|---|---|---|---|---|
| Claude Opus 5.5 | 93% | 67% | 0.90 | 80% | 70% | 80% | $4.02 |
| Claude Sonnet 5.5 | 73% | 40% | 0.67 | 57% | 30% | 57% | $1.80 |
| Claude Haiku 5.5 | 53% | 20% | 0.44 | 37% | 30% | 37% | $0.17 |
| Gemma 4 31B | 47% | 20% | 0.50 | 38% | 0% | 33% | $0 |

Gemma lost 4 of 30 episodes to vLLM read timeouts (counted as unsolved). The first program-mode
run was thrown away: the Claude client had `max_tokens=4096` and Sonnet's reply was frequently
all thinking and no code (`results/program_v0_maxtokens_bug/` keeps it). The fix is in
`arc_env/clients.py`.

- **Writing the rule is harder than drawing the answer.** Every model drops 10-20 points from
  passive to program mode at the same tier, although program mode also gets 3 train pairs. Opus
  expert goes 77% to 67%, Sonnet 60% to 40%. The gap is the difference between pattern-completing
  one grid and stating a rule that holds for five more.
- **Hidden tests catch overfit programs.** 4-6 episodes per model ended with a program that passes
  the visible pairs but only 1-4 of 5 hidden ones (hidden pass 0.2-0.8). In passive mode those
  would all count as solved. Partial hidden credit is also a natural dense reward for RL.
- **Feedback still helps, but less than in passive mode**: a wrong program gets per-cell feedback
  on the visible test only, and most fixes are to the rule, not the code.

## Results: active mode (probe the rule)

Same tiers and episode counts, but the model sees **one** train pair instead of three and may send up
to 6 input grids of its own design to the hidden rule before answering. Reward on solve is
`min(1, 4 / (probes + submissions))^2`, the ARC-AGI-3 efficiency shape, so a two-probe solve scores 1
and a six-probe solve scores 0.25.

Full tables: [results/active/summary.md](results/active/summary.md).

| model | hard solve | expert solve | mean probes (solved) | solved with <=2 probes | first-try (all) | cost |
|---|---|---|---|---|---|---|
| Claude Opus 5.5* | 100% | 100% | 1.1 | 85% | 88% | $3.12 |
| Claude Sonnet 5.5 | 80% | 33% | 0.6 | 94% | 33% | $1.40 |
| Gemma 4 31B | 60% | 20% | 1.2 | 92% | 27% | $0 |
| Claude Haiku 5.5 | 60% | 13% | 2.0 | 45% | 27% | $0.24 |

\* Opus hit the $3 per-model cap at 26 episodes (expert n=11).

- **Opus can use probes; nobody else can.** With one train pair and about two probes of its own
  design, Opus solves every expert task, above its 77% in passive mode with three pairs. Its probes
  are textbook experiments: a single pixel, then one small object, then two colours. Sonnet mostly
  does not probe at all (0.6 per solve) and drops to 33% on expert; Haiku probes the most and gets
  the least from it. Probe design, not probe count, is the skill.
- **Fewer examples hurt the weaker models.** Gemma falls from 37% (passive expert, 3 pairs) to 20%
  with one pair plus probes, Haiku from 27% to 13%. They cannot replace a shown example with an
  experiment of their own.
- After the fix, one Opus and one Sonnet episode each tried the test-input probe once, were refused,
  and solved the task legitimately.

**The first active run was an exploit, and it is kept.** Opus discovered in 14 of 30 episodes
(9 of 15 expert) that the cheapest probe is the test input itself: send it, read the output, submit
it, two actions, full reward. Sonnet did it 4 times, Haiku 3, Gemma never. Opus scored 100% on
expert that way (`results/active_v1_testinput_exploit/`). Non-background overlap between probes and
the test input is bimodal, near 0 for genuine probes and near 1 for the copies, so the env now
rejects any probe with overlap >= 0.5, still charges the action, says why, and counts the attempts.
The table above is from the hardened env. The lesson is the one every RL env team relearns: the
strongest model is also the best auditor of your reward, so run it first and read its transcripts.

## Random-agent calibration

ARC-AGI-3 requires that random play solve a level less than 1 in 10,000 times. `python -m
arc_env.baselines` scores four trivial policies on 500 tasks per tier:

| tier | identity | copy first output | most common colour (given shape) | random fill (given shape) |
|---|---|---|---|---|
| easy | 0.0% | 0.0% | 0.4% | 0.0% |
| medium | 0.0% | 0.8% | 6.2% | 0.0% |
| hard | 0.0% | 0.2% | 1.0% | 0.0% |
| expert | 0.0% | 1.0% | 4.2% | 0.2% |

The 6% is the count-to-bar family, whose output is a single-colour bar; the policy is also handed
the correct output shape, which the agent is not. Everything else is under 1%.

## Notes / things I'd do next

- Count-to-bar should output something with more structure than a monochrome bar.
- Active mode's reference of 4 actions stands in for a human baseline; measure probes-per-family
  on people and use that, as ARC-AGI-3 does.
- A DSL instead of Python in program mode, with program length in the reward (shortest-program
  overlap is the ARC-AGI-3 novelty criterion).
- More families (symmetry repair, path drawing, counting with comparisons).
- RL training: all three modes expose a dense signal (cell accuracy, probe count, hidden pass
  fraction); wire the env into a GRPO loop against the local Gemma server and test whether
  feedback-driven gains transfer across families.
- Run every new mode on the strongest model first and read its transcripts before trusting the
  numbers. See above.
