"""Render a task to SVG (no dependencies) for docs and transcript inspection."""
from .generators import make_task

PALETTE = ["#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00", "#AAAAAA",
           "#F012BE", "#FF851B", "#7FDBFF", "#870C25"]


def grid_svg(g, x0, y0, cell=14):
    parts = []
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            parts.append(f'<rect x="{x0 + c*cell}" y="{y0 + r*cell}" width="{cell}" height="{cell}" '
                         f'fill="{PALETTE[int(g[r, c])]}" stroke="#555" stroke-width="0.5"/>')
    return "\n".join(parts), w * cell, h * cell


def task_svg(task, cell=14, gap=18):
    rows = list(task.train) + [(task.test_input, task.test_output)]
    x, y = 10, 28
    out = []
    maxw = 0
    for i, (a, b) in enumerate(rows):
        label = f"example {i+1}" if i < len(task.train) else "test"
        out.append(f'<text x="{x}" y="{y-8}" font-family="monospace" font-size="11" fill="#ccc">{label}</text>')
        s1, w1, h1 = grid_svg(a, x, y, cell)
        s2, w2, h2 = grid_svg(b, x + w1 + 30, y, cell)
        out += [s1, f'<text x="{x + w1 + 9}" y="{y + max(h1,h2)/2 + 4}" font-size="13" fill="#ccc">→</text>', s2]
        y += max(h1, h2) + gap + 12
        maxw = max(maxw, w1 + w2 + 30)
    W, H = maxw + 20, y
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">'
            f'<rect width="100%" height="100%" fill="#1e1e1e"/>'
            f'<text x="10" y="14" font-family="monospace" font-size="11" fill="#999">rule: {task.rule} ({task.difficulty})</text>'
            + "\n".join(out) + "</svg>")


if __name__ == "__main__":
    import sys
    for d, seed in [("easy", 5), ("medium", 7), ("hard", 3)]:
        t = make_task(seed, d)
        with open(f"docs/sample_{d}.svg", "w") as f:
            f.write(task_svg(t))
        print(d, t.rule)
