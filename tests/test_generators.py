import numpy as np
import pytest
from arc_env.generators import make_task, GENERATORS
from arc_env.generators.primitives import ALL
from arc_env.grid import parse_grid, to_text
from arc_env.env import ArcEnv


@pytest.mark.parametrize("prim", ALL, ids=[p.name for p in ALL])
def test_primitive_changes_input(prim):
    rng = np.random.default_rng(0)
    for i in range(10):
        params = prim.sample_params(rng)
        x = prim.sample_input(rng, (8, 8), params)
        y = prim.apply(x, params)
        assert y.ndim == 2 and y.size > 0
        if prim.shape_preserving:
            assert y.shape == x.shape


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard", "expert"])
def test_make_task_is_deterministic_and_nontrivial(difficulty):
    for seed in range(30):
        t1 = make_task(seed, difficulty)
        t2 = make_task(seed, difficulty)
        assert t1.rule == t2.rule
        assert np.array_equal(t1.test_output, t2.test_output)
        assert len(t1.train) == 3
        for x, y in t1.train:
            assert not np.array_equal(x, y)


def test_hard_tier_composes_sometimes():
    fams = [make_task(s, "hard").families for s in range(40)]
    assert any(len(f) == 2 for f in fams)


def test_parse_roundtrip():
    g = np.array([[0, 1, 2], [3, 4, 5]])
    assert np.array_equal(parse_grid(to_text(g)), g)
    with pytest.raises(ValueError):
        parse_grid("1 2\n3")


def test_env_feedback_and_reward():
    env = ArcEnv("easy", max_attempts=3)
    obs = env.reset(seed=1)
    target = env.task.test_output
    wrong = np.zeros((1, 1), dtype=int)
    obs, r, done, info = env.step(wrong)
    assert r == 0 and not done and not obs["feedback"]["shape_ok"]
    almost = target.copy()
    almost[0, 0] = (almost[0, 0] + 1) % 10
    obs, r, done, info = env.step(almost)
    assert r == 0 and not done and obs["feedback"]["shape_ok"]
    assert obs["feedback"]["wrong_mask"].sum() == 1
    obs, r, done, info = env.step(target)
    assert done and info["solved"] and abs(r - 0.7) < 1e-9


def test_expert_tier_is_three_deep():
    fams = [make_task(s, "expert").families for s in range(20)]
    assert all(len(f) == 3 for f in fams)
