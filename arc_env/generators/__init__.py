"""Task generation: pick primitives by difficulty, optionally compose two, build train/test pairs."""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
from .primitives import ALL, Primitive

GENERATORS = {p.name: p for p in ALL}

TIER_ORDER = {"easy": 0, "medium": 1, "hard": 2, "expert": 3}

SIZES = {"easy": (5, 8), "medium": (7, 12), "hard": (9, 15), "expert": (10, 16)}


@dataclass
class Task:
    train: List[Tuple[np.ndarray, np.ndarray]]
    tests: List[Tuple[np.ndarray, np.ndarray]]  # tests[0] is the one shown; the rest stay hidden
    rule: str  # hidden description, for analysis only
    difficulty: str
    seed: int
    families: List[str] = field(default_factory=list)
    _gen: object = field(default=None, repr=False)
    _params: dict = field(default_factory=dict, repr=False)

    @property
    def test_input(self):
        return self.tests[0][0]

    @property
    def test_output(self):
        return self.tests[0][1]

    def apply(self, grid):
        """Run the hidden rule on an arbitrary grid (used by the active/probe mode)."""
        return self._gen.apply(np.asarray(grid, dtype=int), self._params)

    def to_json(self):
        return {
            "train": [{"input": a.tolist(), "output": b.tolist()} for a, b in self.train],
            "tests": [{"input": a.tolist(), "output": b.tolist()} for a, b in self.tests],
            "rule": self.rule, "difficulty": self.difficulty, "seed": self.seed, "families": self.families,
        }


class Composed:
    """Apply a chain of primitives in order. All but the last must be shape-preserving,
    since each one's output is the next one's input and only the first samples the input."""

    def __init__(self, *prims: Primitive):
        self.prims = prims
        self.name = "+".join(p.name for p in prims)
        self.shape_preserving = all(p.shape_preserving for p in prims)

    def sample_params(self, rng):
        return {"chain": [p.sample_params(rng) for p in self.prims]}

    def sample_input(self, rng, size, params):
        return self.prims[0].sample_input(rng, size, params["chain"][0])

    def apply(self, g, params):
        for p, pp in zip(self.prims, params["chain"]):
            g = p.apply(g, pp)
        return g

    def describe(self, params):
        return " then ".join(p.describe(pp) for p, pp in zip(self.prims, params["chain"]))


def _candidates(difficulty):
    lvl = TIER_ORDER[difficulty]
    return [p for p in ALL if TIER_ORDER[p.tier] <= lvl]


def _pick_generator(rng, difficulty):
    cands = _candidates(difficulty)
    sp = [p for p in cands if p.shape_preserving]
    if difficulty == "hard" and rng.random() < 0.7:
        a, b = rng.choice(len(sp), size=2, replace=False)
        return Composed(sp[a], sp[b])
    if difficulty == "expert":
        # three rules in a row; 30% of the time the last one changes the grid shape
        idx = rng.choice(len(sp), size=3, replace=False)
        chain = [sp[i] for i in idx]
        if rng.random() < 0.3:
            sc = [p for p in cands if not p.shape_preserving]
            chain[-1] = sc[int(rng.integers(len(sc)))]
        return Composed(*chain)
    return cands[int(rng.integers(len(cands)))]


def _nontrivial(pairs):
    # every pair must change something, and outputs must not all be identical
    if any(np.array_equal(a, b) for a, b in pairs):
        return False
    outs = [b for _, b in pairs]
    if all(o.shape == outs[0].shape and np.array_equal(o, outs[0]) for o in outs):
        return False
    if any(b.size == 0 or b.shape[0] > 30 or b.shape[1] > 30 for _, b in pairs):
        return False
    return True


def make_task(seed, difficulty="easy", n_train=3, n_test=1):
    rng = np.random.default_rng(seed)
    lo, hi = SIZES[difficulty]
    for attempt in range(50):
        gen = _pick_generator(rng, difficulty)
        params = gen.sample_params(rng)
        pairs = []
        ok = True
        for _ in range(n_train + n_test):
            size = (int(rng.integers(lo, hi + 1)), int(rng.integers(lo, hi + 1)))
            try:
                x = gen.sample_input(rng, size, params)
                y = gen.apply(x, params)
            except Exception:
                ok = False
                break
            pairs.append((x, y))
        if ok and _nontrivial(pairs):
            fams = gen.name.split("+")
            return Task(train=pairs[:n_train], tests=pairs[n_train:],
                        rule=gen.describe(params), difficulty=difficulty, seed=seed, families=fams,
                        _gen=gen, _params=params)
    raise RuntimeError(f"could not build a task for seed={seed} difficulty={difficulty}")
