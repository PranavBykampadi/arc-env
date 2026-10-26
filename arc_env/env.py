"""Multi-turn ARC environment with three interaction modes.

passive : classic. See train pairs + test input, submit a grid, get per-cell feedback, up to K attempts.
active  : ARC-AGI-3 flavoured. Fewer train pairs are shown, but the agent can *probe* the hidden rule with
          inputs of its own design. Reward is action-efficiency: solved * min(1, ref/actions)^2, where
          actions = probes + submissions and ref is what the passive mode hands over for free.
program : the agent submits a Python transform(grid) function. It is run on the train pairs (results shown)
          and on hidden test pairs sampled from the same rule (only the pass count is shown). Reward is the
          hidden pass fraction of the final submission. This is the hardest to pattern-match.

reward_mode: "sparse" gives solve-or-nothing (with attempt decay); "shaped" adds partial credit for the best
per-cell accuracy of an unsolved episode (capped at 0.3 so it never competes with solving).
"""
import json, os, subprocess, sys
import numpy as np
from .generators import make_task, Task
from .grid import to_text

MODES = ("passive", "active", "program")


class ArcEnv:
    def __init__(self, difficulty="easy", max_attempts=3, reveal_shape=True, n_train=3, mode="passive",
                 reward_mode="sparse", max_probes=6, active_train_pairs=1, n_hidden_tests=5):
        assert mode in MODES
        self.difficulty, self.max_attempts, self.reveal_shape = difficulty, max_attempts, reveal_shape
        self.n_train, self.mode, self.reward_mode = n_train, mode, reward_mode
        self.max_probes, self.active_train_pairs, self.n_hidden_tests = max_probes, active_train_pairs, n_hidden_tests
        self.task: Task = None
        self.attempt = 0
        self.probes = 0
        self.history = []

    # ----- lifecycle -----
    def reset(self, seed):
        n_test = self.n_hidden_tests if self.mode == "program" else 1  # program: shown test input is one of the hidden-output tests
        self.task = make_task(seed, self.difficulty, self.n_train, n_test=n_test)
        self.attempt, self.probes, self.history = 0, 0, []
        shown = self.task.train[:self.active_train_pairs] if self.mode == "active" else self.task.train
        return {"train": shown, "test_input": self.task.test_input, "attempts_left": self.max_attempts,
                "probes_left": self.max_probes if self.mode == "active" else 0, "feedback": None}

    def _obs(self, fb):
        shown = self.task.train[:self.active_train_pairs] if self.mode == "active" else self.task.train
        return {"train": shown, "test_input": self.task.test_input, "attempts_left": self.max_attempts - self.attempt,
                "probes_left": self.max_probes - self.probes, "feedback": fb}

    # ----- active mode: probe the hidden rule -----
    def probe(self, grid):
        """Apply the hidden rule to a grid the agent designed. Costs one action. Returns (obs, reward, done, info)."""
        assert self.mode == "active", "probe() only in active mode"
        if self.probes >= self.max_probes:
            fb = {"probe": True, "error": "no probes left", "solved": False}
            return self._obs(fb), 0.0, False, {"probe": True}
        self.probes += 1
        grid = np.asarray(grid, dtype=int)
        fb = {"probe": True, "solved": False, "input": grid}
        if grid.ndim != 2 or grid.size == 0 or grid.shape[0] > 30 or grid.shape[1] > 30 or (grid < 0).any() or (grid > 9).any():
            fb["error"] = "probe grid must be 2D, at most 30x30, values 0-9"
        else:
            try:
                out = self.task.apply(grid)
                fb["output"] = None if out is None or out.size == 0 else out
                if fb["output"] is None:
                    fb["error"] = "the rule produced an empty grid for this input"
            except Exception as e:
                fb["error"] = f"the rule is undefined for this input ({type(e).__name__})"
        self.history.append(fb)
        return self._obs(fb), 0.0, False, {"probe": True, "probes": self.probes}

    # ----- submissions -----
    def step(self, action):
        """action: a grid (passive/active) or Python source defining transform(grid) (program)."""
        self.attempt += 1
        if self.mode == "program":
            fb, solved = self._eval_program(action)
        else:
            fb, solved = self._eval_grid(action)
        fb["attempt"], fb["solved"] = self.attempt, solved
        self.history.append(fb)
        done = solved or self.attempt >= self.max_attempts
        reward = self._reward(solved, fb) if done else 0.0
        info = {"solved": solved, "attempt": self.attempt, "probes": self.probes, "rule": self.task.rule,
                "families": self.task.families, "cell_acc": fb.get("cell_acc", 0.0),
                "hidden_pass": fb.get("hidden_pass"), "actions": self.probes + self.attempt}
        return self._obs(fb), reward, done, info

    def _eval_grid(self, grid):
        target = self.task.test_output
        grid = np.asarray(grid, dtype=int)
        fb = {"shape_ok": grid.shape == target.shape}
        if fb["shape_ok"]:
            wrong = grid != target
            fb["cell_acc"] = float(1 - wrong.mean())
            fb["wrong_mask"] = wrong.astype(int)
            return fb, not wrong.any()
        fb.update(cell_acc=0.0, wrong_mask=None, got_shape=grid.shape,
                  expected_shape=target.shape if self.reveal_shape else None)
        return fb, False

    def _eval_program(self, code):
        pairs = list(self.task.train) + list(self.task.tests)
        req = {"code": code, "inputs": [a.tolist() for a, _ in pairs]}
        runner = os.path.join(os.path.dirname(__file__), "sandbox_runner.py")
        try:
            p = subprocess.run([sys.executable, "-I", "-S", runner], input=json.dumps(req).encode(),
                               capture_output=True, timeout=30, env={"PATH": "/usr/bin:/bin"})
            res = json.loads(p.stdout.decode() or "{}")
        except Exception as e:
            res = {}
        outs = res.get("outputs", [None] * len(pairs)); errs = res.get("errors", ["runner failed"] * len(pairs))
        n_tr = len(self.task.train)
        results = []
        for (a, b), o, e in zip(pairs, outs, errs):
            if o is None:
                results.append({"ok": False, "error": e})
            else:
                o = np.asarray(o, dtype=int)
                ok = o.shape == b.shape and np.array_equal(o, b)
                acc = float((o == b).mean()) if o.shape == b.shape else 0.0
                results.append({"ok": ok, "error": None, "cell_acc": acc, "got": o})
        train_res, test_res = results[:n_tr], results[n_tr:]
        hidden_pass = sum(r["ok"] for r in test_res) / len(test_res)
        fb = {"train_results": train_res, "hidden_pass": hidden_pass, "n_hidden": len(test_res),
              "cell_acc": float(np.mean([r.get("cell_acc", 0.0) for r in test_res])), "shape_ok": True}
        return fb, hidden_pass == 1.0

    def _reward(self, solved, fb):
        decay = 1.0 - 0.15 * (self.attempt - 1)
        if self.mode == "active":
            ref = self.n_train + 1  # passive mode gets n_train pairs for free plus one submission
            r = min(1.0, ref / (self.probes + self.attempt)) ** 2 if solved else 0.0
            if not solved and self.reward_mode == "shaped":
                r = 0.3 * max(h.get("cell_acc", 0.0) for h in self.history if not h.get("probe"))
            return r
        if self.mode == "program":
            r = fb["hidden_pass"] * decay  # graded by construction
            return r
        if solved:
            return decay
        if self.reward_mode == "shaped":
            return 0.3 * max(h.get("cell_acc", 0.0) for h in self.history)
        return 0.0

    # ----- text rendering of feedback for LLM agents -----
    def feedback_text(self, fb, reveal_shape=True):
        if fb is None:
            return ""
        if fb.get("probe"):
            if fb.get("error"):
                return f"Probe result: {fb['error']}. Probes left: {self.max_probes - self.probes}."
            return (f"Probe result, your input:\n{to_text(fb['input'])}\nrule output:\n{to_text(fb['output'])}\n"
                    f"Probes left: {self.max_probes - self.probes}.")
        if fb["solved"]:
            return "Correct."
        if self.mode == "program":
            lines = []
            for i, r in enumerate(fb["train_results"], 1):
                if r["ok"]:
                    lines.append(f"example {i}: pass")
                elif r["error"]:
                    lines.append(f"example {i}: ERROR {r['error']}")
                else:
                    lines.append(f"example {i}: wrong ({r['cell_acc']:.0%} cells match). Your output:\n{to_text(r['got'])}")
            k = round(fb["hidden_pass"] * fb["n_hidden"])
            lines.append(f"hidden tests: {k}/{fb['n_hidden']} pass")
            return "Not yet.\n" + "\n".join(lines)
        if not fb["shape_ok"]:
            s = f"Wrong. Your grid is {fb['got_shape'][0]}x{fb['got_shape'][1]}"
            if reveal_shape and fb.get("expected_shape"):
                s += f" but the expected output is {fb['expected_shape'][0]}x{fb['expected_shape'][1]}"
            return s + "."
        pct = round(fb["cell_acc"] * 100)
        return (f"Wrong. Shape is right, {pct}% of cells are correct. "
                f"Cells marked 1 below are wrong (0 = correct):\n{to_text(fb['wrong_mask'])}")
