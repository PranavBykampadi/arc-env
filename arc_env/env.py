"""Multi-turn ARC environment.

reset() -> observation with train pairs and the test input
step(grid) -> (observation, reward, done, info)
  observation carries feedback about the last attempt: shape match, fraction of
  cells correct, and a mask of the wrong cells. The model gets max_attempts tries.
"""
import numpy as np
from .generators import make_task, Task
from .grid import to_text


class ArcEnv:
    def __init__(self, difficulty="easy", max_attempts=3, reveal_shape=True, n_train=3):
        self.difficulty = difficulty
        self.max_attempts = max_attempts
        self.reveal_shape = reveal_shape
        self.n_train = n_train
        self.task: Task = None
        self.attempt = 0
        self.history = []

    def reset(self, seed):
        self.task = make_task(seed, self.difficulty, self.n_train)
        self.attempt = 0
        self.history = []
        return {"train": self.task.train, "test_input": self.task.test_input,
                "attempts_left": self.max_attempts, "feedback": None}

    def step(self, grid):
        self.attempt += 1
        target = self.task.test_output
        grid = np.asarray(grid, dtype=int)
        fb = {"attempt": self.attempt, "shape_ok": grid.shape == target.shape}
        if fb["shape_ok"]:
            wrong = (grid != target)
            fb["cell_acc"] = float(1 - wrong.mean())
            fb["wrong_mask"] = wrong.astype(int)
            solved = not wrong.any()
        else:
            fb["cell_acc"] = 0.0
            fb["wrong_mask"] = None
            fb["got_shape"] = grid.shape
            fb["expected_shape"] = target.shape if self.reveal_shape else None
            solved = False
        fb["solved"] = solved
        self.history.append(fb)
        done = solved or self.attempt >= self.max_attempts
        # full credit on first try, small penalty per extra attempt
        reward = (1.0 - 0.15 * (self.attempt - 1)) if solved else 0.0
        obs = {"train": self.task.train, "test_input": self.task.test_input,
               "attempts_left": self.max_attempts - self.attempt, "feedback": fb}
        info = {"solved": solved, "attempt": self.attempt, "rule": self.task.rule,
                "families": self.task.families, "cell_acc": fb["cell_acc"]}
        return obs, reward, done, info

    @staticmethod
    def feedback_text(fb, reveal_shape=True):
        if fb is None:
            return ""
        if fb["solved"]:
            return "Correct."
        if not fb["shape_ok"]:
            s = f"Wrong. Your grid is {fb['got_shape'][0]}x{fb['got_shape'][1]}"
            if reveal_shape and fb.get("expected_shape"):
                s += f" but the expected output is {fb['expected_shape'][0]}x{fb['expected_shape'][1]}"
            return s + "."
        pct = round(fb["cell_acc"] * 100)
        return (f"Wrong. Shape is right, {pct}% of cells are correct. "
                f"Cells marked 1 below are wrong (0 = correct):\n{to_text(fb['wrong_mask'])}")
