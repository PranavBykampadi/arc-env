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


def test_hidden_tests_and_apply():
    t = make_task(11, "medium", n_test=4)
    assert len(t.tests) == 4
    for x, y in t.tests:
        assert np.array_equal(t.apply(x), y)


def test_active_mode_probe_and_efficiency_reward():
    env = ArcEnv("easy", mode="active", max_probes=3)
    obs = env.reset(seed=2)
    assert len(obs["train"]) == 1 and obs["probes_left"] == 3
    obs, r, done, info = env.probe(env.task.train[1][0])
    assert np.array_equal(obs["feedback"]["output"], env.task.train[1][1]) and not done
    obs, r, done, info = env.step(env.task.test_output)
    assert done and info["solved"] and abs(r - 1.0) < 1e-9  # 2 actions <= ref 4
    env.reset(seed=2)
    for _ in range(3):
        env.probe(np.zeros((3, 3), dtype=int))
    obs, r, done, info = env.step(np.zeros((1, 1), dtype=int))
    obs, r, done, info = env.step(env.task.test_output)
    assert done and abs(r - (4 / 5) ** 2) < 1e-9


def test_program_mode_hidden_pass_fraction():
    env = ArcEnv("easy", mode="program", n_hidden_tests=3)
    env.reset(seed=4)
    obs, r, done, info = env.step("def transform(g):\n    return g\n")
    assert not done and info["hidden_pass"] == 0.0 and obs["feedback"]["n_hidden"] == 3
    obs, r, done, info = env.step("def transform(g):\n    raise RuntimeError('x')\n")
    assert obs["feedback"]["train_results"][0]["error"]
    # a correct program: use the hidden rule itself via a lookup is impossible, so use the real generator for flip tasks
    t = make_task(0, "easy")
    env2 = ArcEnv("easy", mode="program", n_hidden_tests=3)
    env2.reset(seed=0)
    code = {"flip_rotate:flip_h": "def transform(g):\n    return [row[::-1] for row in g]\n",
            "flip_rotate:flip_v": "def transform(g):\n    return g[::-1]\n"}.get(env2.task.rule)
    if code:
        obs, r, done, info = env2.step(code)
        assert done and info["solved"] and abs(r - 1.0) < 1e-9


def test_program_sandbox_timeout():
    env = ArcEnv("easy", mode="program", n_hidden_tests=1)
    env.reset(seed=4)
    obs, r, done, info = env.step("def transform(g):\n    while True: pass\n")
    assert all(tr["error"] for tr in obs["feedback"]["train_results"])


def test_shaped_reward_gives_partial_credit():
    env = ArcEnv("easy", max_attempts=1, reward_mode="shaped")
    env.reset(seed=1)
    almost = env.task.test_output.copy(); almost[0, 0] = (almost[0, 0] + 1) % 10
    obs, r, done, info = env.step(almost)
    assert done and 0 < r <= 0.3
