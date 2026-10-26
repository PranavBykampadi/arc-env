"""LLM agent that plays one ArcEnv episode as a multi-turn conversation, in any of the three modes."""
import re
import numpy as np
from .grid import to_text, parse_grid
from .env import ArcEnv

SYSTEM_PASSIVE = """You are solving ARC-style grid puzzles. Each puzzle has a hidden transformation rule.
You are shown a few input->output examples, then a new input. Infer the rule and produce the output grid.
Grids are rows of integers 0-9 separated by spaces; 0 is background. Output size may differ from input size.

Think briefly about the rule, then give your final answer as a single fenced block tagged `grid`, e.g.

```grid
0 1 0
2 2 2
```

You may get feedback and another try if your answer is wrong. Always end with exactly one ```grid block."""

SYSTEM_ACTIVE = """You are solving ARC-style grid puzzles with a hidden transformation rule, but you are shown very
few examples. Instead you can run experiments: design an input grid of your own, and the rule will be applied
to it and the result shown to you. Good probes isolate one hypothesis at a time (a single pixel, one small
object, two colors, etc.). Grids are rows of integers 0-9 separated by spaces; 0 is background.

Each message must end with exactly one fenced block, either a probe:

```probe
0 0 0
0 3 0
0 0 0
```

or your answer for the test input:

```grid
0 1 0
2 2 2
```

Every probe and every answer counts as an action and your score drops with the square of the action count,
so probe only when it buys information, and answer as soon as you are confident. You have a limited number of
probes and answer attempts; wrong answers get per-cell feedback. Probes that copy the test input (or most of
it) are rejected and still cost an action: the point is to work out the rule, not to look up the answer."""

SYSTEM_PROGRAM = """You are solving ARC-style grid puzzles by writing code. Each puzzle has a hidden transformation
rule shown through a few input->output examples. Write a Python function

    def transform(grid: list[list[int]]) -> list[list[int]]

that implements the rule in general (not just for the examples). It will be run on the examples and on
several hidden test inputs drawn from the same rule; you are told which examples passed and how many hidden
tests passed. Only the standard library and numpy are available; no file or network access.

Think about the rule, then give the full function in a single ```python block. You may get feedback and
another try. Always end with exactly one ```python block."""

SYSTEMS = {"passive": SYSTEM_PASSIVE, "active": SYSTEM_ACTIVE, "program": SYSTEM_PROGRAM}


def format_task(obs, mode):
    parts = []
    for i, (x, y) in enumerate(obs["train"], 1):
        parts.append(f"Example {i} input:\n{to_text(x)}\nExample {i} output:\n{to_text(y)}\n")
    parts.append(f"Test input:\n{to_text(obs['test_input'])}\n")
    if mode == "active":
        parts.append(f"You have {obs['probes_left']} probes and {obs['attempts_left']} answer attempts. "
                     f"Probe or answer.")
    elif mode == "program":
        parts.append("Write transform().")
    else:
        parts.append("What is the test output?")
    return "\n".join(parts)


def extract_block(text, tags):
    """Last fenced block whose tag is in `tags`. Returns (tag, body) or (None, None).
    An unclosed final fence (response cut off by max_tokens) is accepted up to the end of the text."""
    found = re.findall(r"```(\w*)\s*\n(.*?)```", text, flags=re.S)
    for tag, body in reversed(found):
        if tag in tags:
            return tag, body
    m = list(re.finditer(r"```(\w*)\s*\n", text))
    if m and text.count("```") % 2 == 1 and m[-1].group(1) in tags:
        return m[-1].group(1), text[m[-1].end():]
    return None, None


def extract_grid(text, tag="grid"):
    t, body = extract_block(text, {tag, ""} if tag == "grid" else {tag})
    if body is not None:
        try:
            return parse_grid(body)
        except ValueError:
            pass
    if tag != "grid":
        return None
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


def run_episode(client, env: ArcEnv, seed, verbose=False, max_turns=14):
    obs = env.reset(seed)
    mode = env.mode
    messages = [{"role": "user", "content": format_task(obs, mode)}]
    usage = {"in": 0, "out": 0}
    transcript = []
    done, reward, info = False, 0.0, {}
    for turn in range(max_turns):
        text, u = client.chat(messages, system=SYSTEMS[mode])
        usage["in"] += u["in"]; usage["out"] += u["out"]
        messages.append({"role": "assistant", "content": text})
        parse_fail = False
        if mode == "program":
            _, code = extract_block(text, {"python", "py"})
            if code is None:
                parse_fail, code = True, "def transform(grid):\n    raise ValueError('no code block')"
            obs, reward, done, info = env.step(code)
            kind = "submit"
        else:
            tag, _ = extract_block(text, {"probe", "grid", ""})
            if mode == "active" and tag == "probe":
                grid = extract_grid(text, "probe")
                if grid is None:
                    grid = np.zeros((1, 1), dtype=int); parse_fail = True
                obs, reward, done, info = env.probe(grid)
                kind = "probe"
            else:
                grid = extract_grid(text, "grid")
                if grid is None:
                    grid = np.zeros((1, 1), dtype=int); parse_fail = True
                obs, reward, done, info = env.step(grid)
                kind = "submit"
        fb = env.feedback_text(obs["feedback"], env.reveal_shape)
        if parse_fail:
            fb = "Could not find a valid fenced block in your answer. " + fb
        transcript.append({"kind": kind, "response": text, "parse_fail": parse_fail, "feedback": fb,
                           "cell_acc": info.get("cell_acc", 0.0), "solved": info.get("solved", False),
                           "hidden_pass": info.get("hidden_pass")})
        if verbose:
            print(f"  {kind} acc={info.get('cell_acc', 0):.2f} solved={info.get('solved')} hidden={info.get('hidden_pass')}")
        if done:
            break
        nxt = {"passive": "Try again. End with one ```grid block.",
               "active": "Probe or answer. End with one ```probe or ```grid block.",
               "program": "Revise transform(). End with one ```python block."}[mode]
        messages.append({"role": "user", "content": fb + "\n\n" + nxt})
    subs = [t for t in transcript if t["kind"] == "submit"]
    if not done:  # ran out of turns (active mode probing forever): score as a miss
        info = {**info, "solved": False, "attempt": env.attempt, "probes": env.probes, "rule": env.task.rule,
                "families": env.task.families}
    return {"seed": seed, "difficulty": env.difficulty, "mode": mode, "rule": info["rule"], "families": info["families"],
            "solved": bool(info.get("solved")), "attempts": env.attempt, "probes": env.probes, "reward": reward,
            "test_probe_rejections": getattr(env, "test_probe_rejections", 0),
            "first_try": bool(subs and subs[0]["solved"]),
            "best_cell_acc": max([t["cell_acc"] for t in subs], default=0.0),
            "final_hidden_pass": (subs[-1]["hidden_pass"] if subs and mode == "program" else None),
            "best_hidden_pass": (max(t["hidden_pass"] or 0 for t in subs) if subs and mode == "program" else None),
            "usage": usage, "transcript": transcript}
