"""Run N episodes per (model, difficulty), write JSONL + a markdown summary.

  python -m arc_env.run --models gemma-4-31b,haiku-5.5 --difficulties easy,medium,hard --episodes 30
"""
import argparse, json, os, time, collections
from concurrent.futures import ThreadPoolExecutor, as_completed
from .clients import get_client, MODELS
from .env import ArcEnv
from .agent import run_episode


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return (max(0.0, c - h), min(1.0, c + h))


def run_model(name, difficulties, episodes, attempts, workers, out_dir, seed0, max_cost=15.0, mode="passive",
              reward_mode="sparse"):
    client = get_client(name)
    pin, pout = MODELS[name][1]
    cost = 0.0
    rows = []
    jobs = [(d, seed0 + i) for d in difficulties for i in range(episodes)]

    def one(d, seed):
        env = ArcEnv(d, max_attempts=attempts, mode=mode, reward_mode=reward_mode)
        try:
            r = run_episode(client, env, seed)
        except Exception as e:
            r = {"seed": seed, "difficulty": d, "mode": mode, "error": repr(e), "solved": False, "attempts": 0,
                 "probes": 0, "reward": 0.0, "first_try": False, "best_cell_acc": 0.0, "usage": {"in": 0, "out": 0},
                 "final_hidden_pass": None, "best_hidden_pass": None,
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
    diffs = [d for d in ["easy", "medium", "hard", "expert"] if any((m, d) in by for m in models)]

    mode = rows[0].get("mode", "passive") if rows else "passive"
    lines = [f"# Results ({mode} mode)", "", "Solve rate = solved within the attempt limit, with a 95% Wilson interval. "
             "First-try = solved on the first submission. Reward = mean episode reward "
             "(passive: attempt-decayed solve; active: solved x min(1, 4/actions)^2; program: hidden-test pass fraction of the final submission). "
             "Cost uses list prices.", ""]
    lines.append("| model | " + " | ".join(f"{d} solve [95% CI] / first-try / reward" for d in diffs) + " | n | $ |")
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
            k = sum(r["solved"] for r in rs); lo, hi = wilson(k, len(rs))
            ft = sum(r["first_try"] for r in rs) / len(rs)
            rw = sum(r["reward"] for r in rs) / len(rs)
            cells.append(f"{k/len(rs):.0%} [{lo:.0%}-{hi:.0%}] / {ft:.0%} / {rw:.2f}")
        lines.append(f"| {m} | " + " | ".join(cells) + f" | {n} | {cost:.2f} |")
    if mode == "active":
        lines += ["", "## Actions used (probes + submissions), solved episodes only", "", "| model | mean probes | mean actions | solved with <=2 probes | episodes that tried to probe with the test input |", "|---|---|---|---|---|"]
        for m in models:
            allr = [r for (mm, _), v in by.items() if mm == m for r in v]
            rs = [r for r in allr if r["solved"]]
            if rs:
                lines.append(f"| {m} | {sum(r['probes'] for r in rs)/len(rs):.1f} | {sum(r['probes']+r['attempts'] for r in rs)/len(rs):.1f} | "
                             f"{sum(r['probes']<=2 for r in rs)/len(rs):.0%} | {sum(1 for r in allr if r.get('test_probe_rejections'))}/{len(allr)} |")
    if mode == "program":
        lines += ["", "## Generalization: hidden-test pass fraction", "", "| model | final submission (mean) | best attempt (mean) | all-hidden-pass rate |", "|---|---|---|---|"]
        for m in models:
            rs = [r for (mm, _), v in by.items() if mm == m for r in v if r.get("final_hidden_pass") is not None]
            if rs:
                lines.append(f"| {m} | {sum(r['final_hidden_pass'] for r in rs)/len(rs):.2f} | {sum(r['best_hidden_pass'] for r in rs)/len(rs):.2f} | "
                             f"{sum(r['solved'] for r in rs)/len(rs):.0%} |")

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
    ap.add_argument("--mode", default="passive", choices=["passive", "active", "program"])
    ap.add_argument("--reward-mode", default="sparse", choices=["sparse", "shaped"])
    a = ap.parse_args()
    out = a.out if a.mode == "passive" else os.path.join(a.out, a.mode)
    if not a.summary_only:
        for m in a.models.split(","):
            run_model(m, a.difficulties.split(","), a.episodes, a.attempts, a.workers, out, a.seed0, a.max_cost,
                      mode=a.mode, reward_mode=a.reward_mode)
    print(summarize(out))


if __name__ == "__main__":
    main()
