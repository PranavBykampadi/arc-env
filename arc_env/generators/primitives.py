"""Transformation families. Each primitive has:
  sample_input(rng, size, params) -> grid     (an input where the rule is non-trivial)
  apply(grid, params) -> grid                   (the hidden rule)
  sample_params(rng) -> dict                    (fixed per task, shared across all pairs)
  shape_preserving: bool                        (can be composed with other primitives)
"""
import numpy as np
from ..grid import random_sparse, random_rects, components, bbox, COLORS


class Primitive:
    name = "base"
    shape_preserving = True
    tier = "easy"  # minimum difficulty where this shows up

    def sample_params(self, rng):
        return {}

    def sample_input(self, rng, size, params):
        raise NotImplementedError

    def apply(self, g, params):
        raise NotImplementedError

    def describe(self, params):
        return self.name


# ---------------- geometric ----------------

class FlipRotate(Primitive):
    name = "flip_rotate"
    OPS = ["flip_h", "flip_v", "rot90", "rot180", "rot270", "transpose"]

    def sample_params(self, rng):
        return {"op": str(rng.choice(self.OPS))}

    def sample_input(self, rng, size, params):
        h, w = size
        # square grids only for ops that would otherwise change shape
        if params["op"] in ("rot90", "rot270", "transpose"):
            w = h
        g = random_sparse(rng, h, w, COLORS[:4], 0.3)
        # make sure the transform actually changes something
        for _ in range(20):
            if not np.array_equal(self.apply(g, params), g):
                return g
            g = random_sparse(rng, h, w, COLORS[:4], 0.3)
        return g

    def apply(self, g, params):
        op = params["op"]
        if op == "flip_h":
            return g[:, ::-1].copy()
        if op == "flip_v":
            return g[::-1, :].copy()
        if op == "rot90":
            return np.rot90(g, -1).copy()
        if op == "rot180":
            return np.rot90(g, 2).copy()
        if op == "rot270":
            return np.rot90(g, 1).copy()
        return g.T.copy()

    def describe(self, params):
        return f"{self.name}:{params['op']}"


class Gravity(Primitive):
    name = "gravity"
    DIRS = ["down", "up", "left", "right"]

    def sample_params(self, rng):
        return {"dir": str(rng.choice(self.DIRS))}

    def sample_input(self, rng, size, params):
        h, w = size
        return random_sparse(rng, h, w, COLORS[:5], 0.25)

    def apply(self, g, params):
        d = params["dir"]
        work = g
        if d == "up":
            work = g[::-1, :]
        elif d == "left":
            work = g.T[::-1, :]
        elif d == "right":
            work = g.T
        out = np.zeros_like(work)
        h, w = work.shape
        for c in range(w):
            col = [v for v in work[:, c] if v != 0]
            if col:
                out[h - len(col):, c] = col
        if d == "up":
            out = out[::-1, :]
        elif d == "left":
            out = out[::-1, :].T
        elif d == "right":
            out = out.T
        return out.copy()

    def describe(self, params):
        return f"{self.name}:{params['dir']}"


class Translate(Primitive):
    name = "translate"
    tier = "medium"

    def sample_params(self, rng):
        dr, dc = 0, 0
        while dr == 0 and dc == 0:
            dr, dc = int(rng.integers(-2, 3)), int(rng.integers(-2, 3))
        return {"dr": dr, "dc": dc}

    def sample_input(self, rng, size, params):
        h, w = size
        g = np.zeros((h, w), dtype=int)
        # keep content away from the edge it would fall off
        dr, dc = params["dr"], params["dc"]
        r0 = max(0, -dr); r1 = h - max(0, dr)
        c0 = max(0, -dc); c1 = w - max(0, dc)
        inner = random_sparse(rng, r1 - r0, c1 - c0, COLORS[:4], 0.3)
        g[r0:r1, c0:c1] = inner
        return g

    def apply(self, g, params):
        dr, dc = params["dr"], params["dc"]
        out = np.zeros_like(g)
        h, w = g.shape
        for r in range(h):
            for c in range(w):
                if g[r, c] and 0 <= r + dr < h and 0 <= c + dc < w:
                    out[r + dr, c + dc] = g[r, c]
        return out

    def describe(self, params):
        return f"{self.name}:({params['dr']},{params['dc']})"


# ---------------- color ----------------

class RecolorMap(Primitive):
    name = "recolor_map"

    def sample_params(self, rng):
        src = list(rng.choice(COLORS, size=3, replace=False))
        dst = list(rng.permutation(src))
        while any(a == b for a, b in zip(src, dst)):
            dst = list(rng.permutation(src))
        return {"map": {int(a): int(b) for a, b in zip(src, dst)}}

    def sample_input(self, rng, size, params):
        h, w = size
        return random_sparse(rng, h, w, list(params["map"].keys()), 0.35)

    def apply(self, g, params):
        out = g.copy()
        for a, b in params["map"].items():
            out[g == a] = b
        return out

    def describe(self, params):
        return f"{self.name}:{params['map']}"


class SwapTwoMostCommon(Primitive):
    name = "swap_most_common"
    tier = "medium"

    def sample_input(self, rng, size, params):
        h, w = size
        for _ in range(30):
            cols = list(rng.choice(COLORS, size=3, replace=False))
            g = random_sparse(rng, h, w, cols, 0.4)
            counts = sorted(((g == c).sum(), c) for c in cols)
            # need an unambiguous top two
            if counts[-1][0] > counts[-2][0] > counts[-3][0] > 0:
                return g
        return g

    def apply(self, g, params):
        vals, cnts = np.unique(g[g != 0], return_counts=True)
        order = np.argsort(-cnts)
        a, b = int(vals[order[0]]), int(vals[order[1]])
        out = g.copy()
        out[g == a] = b
        out[g == b] = a
        return out


class RecolorBySize(Primitive):
    """Objects of size 1 become one color, bigger objects another."""
    name = "recolor_by_size"
    tier = "medium"

    def sample_params(self, rng):
        a, b, src = [int(x) for x in rng.choice(COLORS, size=3, replace=False)]
        return {"single": a, "multi": b, "src": src}

    def sample_input(self, rng, size, params):
        h, w = size
        g, _ = random_rects(rng, h, w, int(rng.integers(2, 4)), [params["src"]], 2, 3, distinct_colors=False)
        # sprinkle singletons that don't touch anything
        for _ in range(int(rng.integers(2, 5))):
            for _ in range(30):
                r, c = int(rng.integers(h)), int(rng.integers(w))
                r0, c0, r1, c1 = max(0, r - 1), max(0, c - 1), min(h, r + 2), min(w, c + 2)
                if not g[r0:r1, c0:c1].any():
                    g[r, c] = params["src"]
                    break
        return g

    def apply(self, g, params):
        out = np.zeros_like(g)
        for color, cells in components(g):
            new = params["single"] if len(cells) == 1 else params["multi"]
            for r, c in cells:
                out[r, c] = new
        return out


# ---------------- object / region ----------------

class FillEnclosed(Primitive):
    """Fill the interior of hollow rectangles with a fixed color."""
    name = "fill_enclosed"

    def sample_params(self, rng):
        return {"fill": int(rng.choice(COLORS))}

    def sample_input(self, rng, size, params):
        h, w = size
        cols = [c for c in COLORS if c != params["fill"]]
        g, _ = random_rects(rng, h, w, int(rng.integers(1, 4)), cols, 3, 5, hollow=True)
        return g

    def apply(self, g, params):
        # flood fill background from the border; anything background left is enclosed
        h, w = g.shape
        reach = np.zeros_like(g, dtype=bool)
        stack = [(r, c) for r in range(h) for c in (0, w - 1)] + [(r, c) for c in range(w) for r in (0, h - 1)]
        while stack:
            r, c = stack.pop()
            if not (0 <= r < h and 0 <= c < w) or reach[r, c] or g[r, c] != 0:
                continue
            reach[r, c] = True
            stack.extend([(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)])
        out = g.copy()
        out[(g == 0) & ~reach] = params["fill"]
        return out

    def describe(self, params):
        return f"{self.name}:{params['fill']}"


class Outline(Primitive):
    """Keep only the border of each filled rectangle."""
    name = "hollow_out"
    tier = "medium"

    def sample_input(self, rng, size, params):
        h, w = size
        g, _ = random_rects(rng, h, w, int(rng.integers(1, 4)), COLORS, 3, 5)
        return g

    def apply(self, g, params):
        out = g.copy()
        h, w = g.shape
        for r in range(h):
            for c in range(w):
                if g[r, c] == 0:
                    continue
                nb = [g[r + dr, c + dc] if 0 <= r + dr < h and 0 <= c + dc < w else 0
                      for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))]
                if all(v == g[r, c] for v in nb):
                    out[r, c] = 0
        return out


class BoundingBox(Primitive):
    """Draw a box of a fixed color around every object."""
    name = "bounding_box"
    tier = "medium"

    def sample_params(self, rng):
        return {"box": int(rng.choice(COLORS))}

    def sample_input(self, rng, size, params):
        h, w = size
        cols = [c for c in COLORS if c != params["box"]]
        g = np.zeros((h, w), dtype=int)
        n = int(rng.integers(1, 3))
        for _ in range(n):
            for _ in range(50):
                oh, ow = int(rng.integers(1, 3)), int(rng.integers(1, 3))
                r, c = int(rng.integers(1, h - oh)), int(rng.integers(1, w - ow))
                if not g[max(0, r - 2):r + oh + 2, max(0, c - 2):c + ow + 2].any():
                    g[r:r + oh, c:c + ow] = int(rng.choice(cols))
                    break
        return g

    def apply(self, g, params):
        out = g.copy()
        h, w = g.shape
        for _, cells in components(g):
            r0, c0, r1, c1 = bbox(cells)
            r0, c0, r1, c1 = r0 - 1, c0 - 1, r1 + 1, c1 + 1
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    if (r in (r0, r1) or c in (c0, c1)) and 0 <= r < h and 0 <= c < w and out[r, c] == 0:
                        out[r, c] = params["box"]
        return out


class Denoise(Primitive):
    """Remove isolated single pixels, keep the big objects."""
    name = "denoise"
    tier = "medium"

    def sample_input(self, rng, size, params):
        h, w = size
        g, _ = random_rects(rng, h, w, int(rng.integers(1, 4)), COLORS[:5], 2, 4)
        noise_color = int(rng.choice(COLORS[5:]))
        for _ in range(int(rng.integers(3, 8))):
            r, c = int(rng.integers(h)), int(rng.integers(w))
            r0, c0, r1, c1 = max(0, r - 1), max(0, c - 1), min(h, r + 2), min(w, c + 2)
            if not g[r0:r1, c0:c1].any():
                g[r, c] = noise_color
        return g

    def apply(self, g, params):
        out = g.copy()
        for _, cells in components(g):
            if len(cells) == 1:
                r, c = cells[0]
                out[r, c] = 0
        return out


class RayCast(Primitive):
    """Every colored cell shoots a line to the edge in a fixed direction."""
    name = "ray"
    DIRS = {"down": (1, 0), "up": (-1, 0), "left": (0, -1), "right": (0, 1)}

    def sample_params(self, rng):
        return {"dir": str(rng.choice(list(self.DIRS)))}

    def sample_input(self, rng, size, params):
        h, w = size
        return random_sparse(rng, h, w, COLORS[:5], 0.08)

    def apply(self, g, params):
        dr, dc = self.DIRS[params["dir"]]
        out = g.copy()
        h, w = g.shape
        for r in range(h):
            for c in range(w):
                if g[r, c]:
                    nr, nc = r + dr, c + dc
                    while 0 <= nr < h and 0 <= nc < w and g[nr, nc] == 0:
                        out[nr, nc] = g[r, c]
                        nr, nc = nr + dr, nc + dc
        return out

    def describe(self, params):
        return f"{self.name}:{params['dir']}"


# ---------------- shape changing ----------------

class CropLargest(Primitive):
    name = "crop_largest"
    shape_preserving = False
    tier = "medium"

    def sample_input(self, rng, size, params):
        h, w = size
        for _ in range(30):
            g, rects = random_rects(rng, h, w, int(rng.integers(2, 4)), COLORS, 1, 4)
            areas = sorted(rh * rw for _, _, rh, rw, _ in rects)
            if len(areas) >= 2 and areas[-1] > areas[-2]:
                return g
        return g

    def apply(self, g, params):
        comps = components(g)
        _, cells = max(comps, key=lambda x: len(x[1]))
        r0, c0, r1, c1 = bbox(cells)
        return g[r0:r1 + 1, c0:c1 + 1].copy()


class MirrorComplete(Primitive):
    """Output is the input glued to its own mirror image."""
    name = "mirror"
    shape_preserving = False

    def sample_params(self, rng):
        return {"axis": str(rng.choice(["h", "v"]))}

    def sample_input(self, rng, size, params):
        h, w = size
        return random_sparse(rng, max(2, h // 2), max(2, w // 2) if params["axis"] == "h" else w, COLORS[:5], 0.35)

    def apply(self, g, params):
        if params["axis"] == "h":
            return np.concatenate([g, g[:, ::-1]], axis=1)
        return np.concatenate([g, g[::-1, :]], axis=0)

    def describe(self, params):
        return f"{self.name}:{params['axis']}"


class Tile(Primitive):
    name = "tile"
    shape_preserving = False
    tier = "medium"

    def sample_params(self, rng):
        return {"reps": (int(rng.integers(1, 3)), int(rng.integers(2, 4))), "alt_flip": bool(rng.random() < 0.5)}

    def sample_input(self, rng, size, params):
        h, w = size
        return random_sparse(rng, max(2, h // 3), max(2, w // 3), COLORS[:5], 0.4)

    def apply(self, g, params):
        rr, rc = params["reps"]
        row = []
        for j in range(rc):
            t = g[:, ::-1] if (params["alt_flip"] and j % 2 == 1) else g
            row.append(t)
        row = np.concatenate(row, axis=1)
        return np.concatenate([row] * rr, axis=0)

    def describe(self, params):
        return f"{self.name}:{params['reps']}:{'altflip' if params['alt_flip'] else 'plain'}"


class CountToBar(Primitive):
    """Count objects of a target color and output a 1xN bar of that color."""
    name = "count_objects"
    shape_preserving = False
    tier = "medium"

    def sample_params(self, rng):
        return {"target": int(rng.choice(COLORS))}

    def sample_input(self, rng, size, params):
        h, w = size
        n_target = int(rng.integers(1, 6))
        others = [c for c in COLORS if c != params["target"]]
        g, _ = random_rects(rng, h, w, n_target, [params["target"]], 1, 2, distinct_colors=False)
        g2, _ = random_rects(rng, h, w, int(rng.integers(1, 4)), others, 1, 2)
        # merge where free (and not adjacent to target objects)
        for r in range(h):
            for c in range(w):
                if g2[r, c] and not g[max(0, r - 1):r + 2, max(0, c - 1):c + 2].any():
                    g[r, c] = g2[r, c]
        return g

    def apply(self, g, params):
        n = sum(1 for color, _ in components(g) if color == params["target"])
        return np.full((1, max(n, 1)), params["target"], dtype=int)

    def describe(self, params):
        return f"{self.name}:{params['target']}"


ALL = [FlipRotate(), Gravity(), Translate(), RecolorMap(), SwapTwoMostCommon(), RecolorBySize(),
       FillEnclosed(), Outline(), BoundingBox(), Denoise(), RayCast(),
       CropLargest(), MirrorComplete(), Tile(), CountToBar()]
