# Results

Solve rate = solved within 3 attempts. First-try = solved on attempt 1. Cell acc = best per-cell accuracy over attempts (0 when shape wrong). Cost uses list prices.

| model | easy solve / first-try / cell acc | medium solve / first-try / cell acc | hard solve / first-try / cell acc | n | $ |
|---|---|---|---|---|---|
| gemma-4-31b | 100% / 87% / 1.00 | 100% / 93% / 1.00 | 50% / 33% / 0.86 | 90 | 0.00 |

## Does feedback help?

| model | first-try | by attempt 2 | by attempt 3 |
|---|---|---|---|
| gemma-4-31b | 71% | 79% | 83% |

## Solve rate by transformation family (all difficulties)

| family | gemma-4-31b |
|---|---|
| bounding_box | 54% (n=13) |
| count_objects | 100% (n=1) |
| crop_largest | 100% (n=2) |
| denoise | 100% (n=7) |
| fill_enclosed | 86% (n=22) |
| flip_rotate | 50% (n=8) |
| gravity | 60% (n=10) |
| hollow_out | 100% (n=4) |
| mirror | 100% (n=6) |
| ray | 83% (n=12) |
| recolor_by_size | 60% (n=5) |
| recolor_map | 70% (n=10) |
| swap_most_common | 43% (n=7) |
| tile | 100% (n=4) |
| translate | 50% (n=4) |
