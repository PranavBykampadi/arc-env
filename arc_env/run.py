"""Run N episodes per (model, difficulty), write JSONL + a markdown summary.

  python -m arc_env.run --models gemma-4-31b,haiku-5.5 --difficulties easy,medium,hard --episodes 30
"""
import argparse, json, os, time, collections
from concurrent.futures import ThreadPoolExecutor, as_completed
from .clients import get_client, MODELS
from .env import ArcEnv
from .agent import run_episode


def run_model(name, difficulties, episodes, attempts, workers, out_dir, seed0, max_cost=15.0):
    client = get_client(name)
    pin, pout = MODELS[name][1]
    cost = 0.0
    rows = []
    jobs = [(d, seed0 + i) for d in difficulties for i in range(episodes)]

    def one(d, seed):
        env = ArcEnv(d, max_attempts=attempts)
        try:
            r = run_episode(client, env, seed)
        except Exception as e:
            r = {"seed": seed, "difficulty": d, "error": repr(e), "solved": False, "attempts": 0,
                 "reward": 0.0, "first_try": False, "best_cell_acc": 0.0, "usage": {"in": 0, "out": 0},
                 "rule": env.task.rule if env.task else "?", "families": env.task.families if env.task else []}
        r["model"] = name
        return r

    path = os.path.join(out_dir, "raw", f"{name}.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex, open(path, "a") as f:
        futs = [ex.submit(one, d, s) for d, s in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            rows.append(r)
            f.write(json.dumps(r) + "\n"); f.flush()
            cost += r["usage"]["in"] / 1e6 * pin + r["usage"]["out"] / 1e6 * pout
            if cost > max_cost:
                print(f"[{name}] spend cap ${max_cost} hit after {i} episodes, cancelling the rest", flush=True)
                for fu in futs:
                    fu.cancel()
                break
            if i % 5 == 0 or i == len(jobs):
                solved = sum(x["solved"] for x in rows)
                print(f"[{name}] {i}/{len(jobs)} solved={solved} ({time.time()-t0:.0f}s)", flush=True)
    return rows


def summarize(out_dir):
    rows = []
    raw = os.path.join(out_dir, "raw")
    for fn in sorted(os.listdir(raw)):
        if fn.endswith(".jsonl"):
            with open(os.path.join(raw, fn)) as f:
                rows += [json.loads(l) for l in f if l.strip()]
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["model"], r["difficulty"])].append(r)
    models = sorted({m for m, _ in by})
    diffs = [d for d in ["easy", "medium", "hard"] if any((m, d) in by for m in models)]

    lines = ["# Results", "", "Solve rate = solved within 3 attempts. First-try = solved on attempt 1. "
             "Cell acc = best per-cell accuracy over attempts (0 when shape wrong). Cost uses list prices.", ""]
    lines.append("| model | " + " | ".join(f"{d} solve / first-try / cell acc" for d in diffs) + " | n | $ |")
    lines.append("|---|" + "---|" * (len(diffs) + 2))
    for m in models:
        cells = []
        n = 0; cost = 0.0
        pin, pout = MODELS.get(m, (None, (0, 0)))[1]
        for d in diffs:
            rs = by.get((m, d), [])
            if not rs:
                cells.append("-"); continue
            n += len(rs)
            cost += sum(r["usage"]["in"] for r in rs) / 1e6 * pin + sum(r["usage"]["out"] for r in rs) / 1e6 * pout
            sr = sum(r["solved"] for r in rs) / len(rs)
            ft = sum(r["first_try"] for r in rs) / len(rs)
            ca = sum(r["best_cell_acc"] for r in rs) / len(rs)
            cells.append(f"{sr:.0%} / {ft:.0%} / {ca:.2f}")
        lines.append(f"| {m} | " + " | ".join(cells) + f" | {n} | {cost:.2f} |")

    # feedback helps? solved on attempt 2 or 3
    lines += ["", "## Does feedback help?", "", "| model | first-try | by attempt 2 | by attempt 3 |", "|---|---|---|---|"]
    for m in models:
        rs = [r for (mm, _), v in by.items() if mm == m for r in v]
        n = len(rs)
        a1 = sum(r["first_try"] for r in rs) / n
        a2 = sum(r["solved"] and r["attempts"] <= 2 for r in rs) / n
        a3 = sum(r["solved"] for r in rs) / n
        lines.append(f"| {m} | {a1:.0%} | {a2:.0%} | {a3:.0%} |")

    # per family
    fam = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        for f_ in r.get("families", []):
            fam[f_][r["model"]].append(r["solved"])
    lines += ["", "## Solve rate by transformation family (all difficulties)", "",
              "| family | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    for f_ in sorted(fam):
        cells = []
        for m in models:
            v = fam[f_].get(m, [])
            cells.append(f"{sum(v)/len(v):.0%} (n={len(v)})" if v else "-")
        lines.append(f"| {f_} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gemma-4-31b")
    ap.add_argument("--difficulties", default="easy,medium,hard")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--out", default="results")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--max-cost", type=float, default=15.0, help="USD cap per model run")
    a = ap.parse_args()
    if not a.summary_only:
        for m in a.models.split(","):
            run_model(m, a.difficulties.split(","), a.episodes, a.attempts, a.workers, a.out, a.seed0, a.max_cost)
    print(summarize(a.out))


if __name__ == "__main__":
    main()
