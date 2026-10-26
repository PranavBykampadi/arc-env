"""Random-agent calibration (the ARC-AGI-3 practice): how often do trivial policies solve a task by accident?
If a tier is solvable by copying the input or by guessing, it is not measuring reasoning."""
import numpy as np
from .generators import make_task


def identity(task):
    return task.test_input


def copy_first_output(task):
    return task.train[0][1]


def most_common_output_color(task):
    y = task.train[0][1]
    vals, cnt = np.unique(y, return_counts=True)
    return np.full(task.test_output.shape, vals[np.argmax(cnt)], dtype=int)


def random_fill(task, rng):
    return rng.integers(0, 10, size=task.test_output.shape)


def calibrate(n=500, difficulties=("easy", "medium", "hard", "expert"), seed0=50_000):
    rng = np.random.default_rng(0)
    rows = []
    for d in difficulties:
        hits = {"identity": 0, "copy_first_output": 0, "most_common_color": 0, "random_fill(correct shape)": 0}
        for i in range(n):
            t = make_task(seed0 + i, d)
            tgt = t.test_output
            for name, g in [("identity", identity(t)), ("copy_first_output", copy_first_output(t)),
                            ("most_common_color", most_common_output_color(t)), ("random_fill(correct shape)", random_fill(t, rng))]:
                if g.shape == tgt.shape and np.array_equal(g, tgt):
                    hits[name] += 1
        rows.append((d, {k: v / n for k, v in hits.items()}))
    return rows


if __name__ == "__main__":
    rows = calibrate()
    names = list(rows[0][1])
    print("| tier | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for d, h in rows:
        print(f"| {d} | " + " | ".join(f"{h[k]:.2%}" for k in names) + " |")
