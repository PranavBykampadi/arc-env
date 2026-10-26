# Results

Solve rate = solved within 3 attempts. First-try = solved on attempt 1. Cell acc = best per-cell accuracy over attempts (0 when shape wrong). Cost uses list prices.

| model | easy solve / first-try / cell acc | medium solve / first-try / cell acc | hard solve / first-try / cell acc | expert solve / first-try / cell acc | n | $ |
|---|---|---|---|---|---|---|
| fable-5.1 | 100% / 100% / 1.00 | 100% / 100% / 1.00 | 90% / 77% / 0.93 | 69% / 54% / 0.69 | 116 | 15.17 |
| gemma-4-31b | 100% / 87% / 1.00 | 100% / 93% / 1.00 | 50% / 33% / 0.86 | 37% / 13% / 0.80 | 120 | 0.00 |
| haiku-5.5 | 100% / 97% / 1.00 | 100% / 97% / 1.00 | 63% / 33% / 0.85 | 27% / 20% / 0.70 | 120 | 0.29 |
| opus-5.5 | 100% / 97% / 1.00 | 100% / 100% / 1.00 | 97% / 80% / 0.97 | 77% / 60% / 0.78 | 120 | 5.43 |
| sonnet-5.5 | 100% / 93% / 1.00 | 100% / 93% / 1.00 | 87% / 67% / 0.89 | 60% / 47% / 0.63 | 120 | 3.85 |

## Does feedback help?

| model | first-try | by attempt 2 | by attempt 3 |
|---|---|---|---|
| fable-5.1 | 84% | 87% | 91% |
| gemma-4-31b | 57% | 68% | 72% |
| haiku-5.5 | 62% | 69% | 72% |
| opus-5.5 | 84% | 92% | 93% |
| sonnet-5.5 | 75% | 86% | 87% |

## Solve rate by transformation family (all difficulties)

| family | fable-5.1 | gemma-4-31b | haiku-5.5 | opus-5.5 | sonnet-5.5 |
|---|---|---|---|---|---|
| bounding_box | 80% (n=20) | 45% (n=22) | 55% (n=22) | 95% (n=22) | 82% (n=22) |
| count_objects | 100% (n=1) | 100% (n=1) | 100% (n=1) | 100% (n=1) | 100% (n=1) |
| crop_largest | 67% (n=6) | 86% (n=7) | 57% (n=7) | 71% (n=7) | 71% (n=7) |
| denoise | 100% (n=10) | 75% (n=12) | 75% (n=12) | 100% (n=12) | 92% (n=12) |
| fill_enclosed | 100% (n=29) | 76% (n=29) | 79% (n=29) | 100% (n=29) | 97% (n=29) |
| flip_rotate | 69% (n=16) | 38% (n=16) | 38% (n=16) | 69% (n=16) | 62% (n=16) |
| gravity | 78% (n=18) | 30% (n=20) | 40% (n=20) | 80% (n=20) | 60% (n=20) |
| hollow_out | 100% (n=11) | 73% (n=11) | 73% (n=11) | 100% (n=11) | 100% (n=11) |
| mirror | 100% (n=7) | 86% (n=7) | 86% (n=7) | 100% (n=7) | 86% (n=7) |
| ray | 80% (n=15) | 69% (n=16) | 62% (n=16) | 81% (n=16) | 69% (n=16) |
| recolor_by_size | 87% (n=15) | 53% (n=17) | 53% (n=17) | 88% (n=17) | 88% (n=17) |
| recolor_map | 75% (n=20) | 55% (n=20) | 55% (n=20) | 85% (n=20) | 70% (n=20) |
| swap_most_common | 89% (n=9) | 40% (n=10) | 20% (n=10) | 90% (n=10) | 70% (n=10) |
| tile | 100% (n=4) | 80% (n=5) | 80% (n=5) | 100% (n=5) | 100% (n=5) |
| translate | 67% (n=12) | 42% (n=12) | 33% (n=12) | 83% (n=12) | 58% (n=12) |
