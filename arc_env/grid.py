"""Small helpers for working with ARC grids (2D numpy int arrays, 0 = background)."""
import numpy as np
from collections import deque

COLORS = list(range(1, 10))


def random_sparse(rng, h, w, colors, density):
    g = np.zeros((h, w), dtype=int)
    mask = rng.random((h, w)) < density
    g[mask] = rng.choice(colors, size=mask.sum())
    return g


def components(g, diagonal=False):
    """Connected components of non-zero cells, same color. Returns list of (color, [(r,c),...])."""
    h, w = g.shape
    seen = np.zeros_like(g, dtype=bool)
    out = []
    if diagonal:
        nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for r in range(h):
        for c in range(w):
            if g[r, c] == 0 or seen[r, c]:
                continue
            color = g[r, c]
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in nbrs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and g[nr, nc] == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append((int(color), cells))
    return out


def bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def place_rect(g, r, c, h, w, color, hollow=False):
    if hollow:
        g[r, c:c + w] = color
        g[r + h - 1, c:c + w] = color
        g[r:r + h, c] = color
        g[r:r + h, c + w - 1] = color
    else:
        g[r:r + h, c:c + w] = color


def random_rects(rng, h, w, n, colors, min_size=2, max_size=4, hollow=False, distinct_colors=True, max_tries=200):
    """Place n non-overlapping (with 1-cell gap) rectangles. Returns grid and list of (r,c,rh,rw,color)."""
    g = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    rects = []
    pool = list(colors)
    for _ in range(n):
        for _ in range(max_tries):
            rh = rng.integers(min_size, max_size + 1)
            rw = rng.integers(min_size, max_size + 1)
            if rh > h or rw > w:
                continue
            r = rng.integers(0, h - rh + 1)
            c = rng.integers(0, w - rw + 1)
            r0, c0, r1, c1 = max(0, r - 1), max(0, c - 1), min(h, r + rh + 1), min(w, c + rw + 1)
            if occupied[r0:r1, c0:c1].any():
                continue
            if distinct_colors and pool:
                color = pool.pop(rng.integers(len(pool)))
            else:
                color = rng.choice(list(colors))
            place_rect(g, r, c, rh, rw, color, hollow=hollow)
            occupied[r:r + rh, c:c + rw] = True
            rects.append((int(r), int(c), int(rh), int(rw), int(color)))
            break
    return g, rects


def to_text(g):
    return "\n".join(" ".join(str(int(v)) for v in row) for row in g)


def parse_grid(text):
    """Parse rows of space/comma separated ints. Raises ValueError on ragged or empty input."""
    rows = []
    for line in text.strip().splitlines():
        line = line.strip().replace(",", " ").replace("[", " ").replace("]", " ")
        if not line:
            continue
        toks = line.split()
        if not all(t.lstrip("-").isdigit() for t in toks):
            raise ValueError(f"non-integer token in row: {line!r}")
        rows.append([int(t) for t in toks])
    if not rows:
        raise ValueError("empty grid")
    w = len(rows[0])
    if any(len(r) != w for r in rows):
        raise ValueError("ragged rows")
    return np.array(rows, dtype=int)
