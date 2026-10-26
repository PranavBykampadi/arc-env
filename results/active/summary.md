# Results (active mode)

Solve rate = solved within the attempt limit, with a 95% Wilson interval. First-try = solved on the first submission. Reward = mean episode reward (passive: attempt-decayed solve; active: solved x min(1, 4/actions)^2; program: hidden-test pass fraction of the final submission). Cost uses list prices.

| model | hard solve [95% CI] / first-try / reward | expert solve [95% CI] / first-try / reward | n | $ |
|---|---|---|---|---|
| gemma-4-31b | 60% [36%-80%] / 47% / 0.60 | 20% [7%-45%] / 7% / 0.20 | 30 | 0.00 |
| haiku-5.5 | 60% [36%-80%] / 47% / 0.52 | 13% [4%-38%] / 7% / 0.10 | 30 | 0.24 |
| opus-5.5 | 100% [80%-100%] / 93% / 1.00 | 100% [74%-100%] / 82% / 0.87 | 26 | 3.12 |
| sonnet-5.5 | 80% [55%-93%] / 53% / 0.76 | 33% [15%-58%] / 13% / 0.33 | 30 | 1.40 |

## Actions used (probes + submissions), solved episodes only

| model | mean probes | mean actions | solved with <=2 probes | episodes that tried to probe with the test input |
|---|---|---|---|---|
| gemma-4-31b | 1.2 | 2.6 | 92% | 0/30 |
| haiku-5.5 | 2.0 | 3.4 | 45% | 0/30 |
| opus-5.5 | 1.1 | 2.2 | 85% | 1/26 |
| sonnet-5.5 | 0.6 | 2.2 | 94% | 1/30 |

## Does feedback help?

| model | first-try | by attempt 2 | by attempt 3 |
|---|---|---|---|
| gemma-4-31b | 27% | 37% | 40% |
| haiku-5.5 | 27% | 33% | 37% |
| opus-5.5 | 88% | 100% | 100% |
| sonnet-5.5 | 33% | 47% | 57% |

## Solve rate by transformation family (all difficulties)

| family | gemma-4-31b | haiku-5.5 | opus-5.5 | sonnet-5.5 |
|---|---|---|---|---|
| bounding_box | 50% (n=8) | 50% (n=8) | 100% (n=8) | 62% (n=8) |
| crop_largest | 100% (n=1) | 100% (n=1) | 100% (n=1) | 100% (n=1) |
| denoise | 25% (n=4) | 25% (n=4) | 100% (n=4) | 25% (n=4) |
| fill_enclosed | 50% (n=2) | 50% (n=2) | 100% (n=2) | 100% (n=2) |
| flip_rotate | 15% (n=13) | 0% (n=13) | 100% (n=11) | 46% (n=13) |
| gravity | 10% (n=10) | 10% (n=10) | 100% (n=7) | 40% (n=10) |
| hollow_out | 0% (n=2) | 0% (n=2) | 100% (n=1) | 50% (n=2) |
| mirror | 0% (n=1) | 0% (n=1) | 100% (n=1) | 100% (n=1) |
| ray | 50% (n=4) | 50% (n=4) | 100% (n=4) | 100% (n=4) |
| recolor_by_size | 33% (n=6) | 33% (n=6) | 100% (n=4) | 50% (n=6) |
| recolor_map | 75% (n=4) | 75% (n=4) | 100% (n=4) | 50% (n=4) |
| swap_most_common | 38% (n=8) | 25% (n=8) | 100% (n=6) | 12% (n=8) |
| translate | 29% (n=7) | 43% (n=7) | 100% (n=5) | 57% (n=7) |
