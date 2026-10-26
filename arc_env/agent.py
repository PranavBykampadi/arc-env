"""LLM agent that plays one ArcEnv episode as a multi-turn conversation."""
import re
import numpy as np
from .grid import to_text, parse_grid
from .env import ArcEnv

SYSTEM = """You are solving ARC-style grid puzzles. Each puzzle has a hidden transformation rule.
You are shown a few input->output examples, then a new input. Infer the rule and produce the output grid.
Grids are rows of integers 0-9 separated by spaces; 0 is background. Output size may differ from input size.

Think briefly about the rule, then give your final answer as a single fenced block tagged `grid`, e.g.

```grid
0 1 0
2 2 2
```

You may get feedback and another try if your answer is wrong. Always end with exactly one ```grid block."""


def format_task(obs):
    parts = []
    for i, (x, y) in enumerate(obs["train"], 1):
        parts.append(f"Example {i} input:\n{to_text(x)}\nExample {i} output:\n{to_text(y)}\n")
    parts.append(f"Test input:\n{to_text(obs['test_input'])}\n\nWhat is the test output?")
    return "\n".join(parts)


def extract_grid(text):
    blocks = re.findall(r"```(?:grid)?\s*\n(.*?)```", text, flags=re.S)
    for b in reversed(blocks):
        try:
            return parse_grid(b)
        except ValueError:
            continue
    # fall back: last run of digit-only lines
    lines = text.strip().splitlines()
    tail = []
    for ln in reversed(lines):
        if re.fullmatch(r"[\d\s,\[\]]+", ln.strip()) and ln.strip():
            tail.append(ln)
        elif tail:
            break
    if tail:
        try:
            return parse_grid("\n".join(reversed(tail)))
        except ValueError:
            pass
    return None


def run_episode(client, env: ArcEnv, seed, verbose=False):
    obs = env.reset(seed)
    messages = [{"role": "user", "content": format_task(obs)}]
    usage = {"in": 0, "out": 0}
    transcript = []
    done = False
    reward = 0.0
    info = {}
    while not done:
        text, u = client.chat(messages, system=SYSTEM)
        usage["in"] += u["in"]; usage["out"] += u["out"]
        messages.append({"role": "assistant", "content": text})
        grid = extract_grid(text)
        if grid is None:
            grid = np.zeros((1, 1), dtype=int)
            parse_fail = True
        else:
            parse_fail = False
        obs, reward, done, info = env.step(grid)
        fb = env.feedback_text(obs["feedback"], env.reveal_shape)
        if parse_fail:
            fb = "Could not find a ```grid block in your answer. " + fb
        transcript.append({"response": text, "parse_fail": parse_fail, "feedback": fb,
                           "cell_acc": info["cell_acc"], "solved": info["solved"]})
        if verbose:
            print(f"  attempt {info['attempt']}: acc={info['cell_acc']:.2f} solved={info['solved']}")
        if not done:
            messages.append({"role": "user", "content": fb + "\n\nTry again. End with one ```grid block."})
    return {"seed": seed, "difficulty": env.difficulty, "rule": info["rule"], "families": info["families"],
            "solved": info["solved"], "attempts": info["attempt"], "reward": reward,
            "first_try": transcript[0]["solved"], "best_cell_acc": max(t["cell_acc"] for t in transcript),
            "usage": usage, "transcript": transcript}
