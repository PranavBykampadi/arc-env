# Results (active mode)

Solve rate = solved within the attempt limit, with a 95% Wilson interval. First-try = solved on the first submission. Reward = mean episode reward (passive: attempt-decayed solve; active: solved x min(1, 4/actions)^2; program: hidden-test pass fraction of the final submission). Cost uses list prices.

| model | hard solve [95% CI] / first-try / reward | expert solve [95% CI] / first-try / reward | n | $ |
|---|---|---|---|---|
| gemma-4-31b | 50% [30%-70%] / 35% / 0.50 | 10% [3%-30%] / 5% / 0.10 | 40 | 0.00 |
| haiku-5.5 | 65% [43%-82%] / 40% / 0.55 | 35% [18%-57%] / 5% / 0.21 | 40 | 0.27 |
| opus-5.5 | 100% [84%-100%] / 90% / 1.00 | 90% [70%-97%] / 50% / 0.80 | 40 | 4.20 |
| sonnet-5.5 | 70% [48%-85%] / 50% / 0.70 | 40% [22%-61%] / 20% / 0.37 | 40 | 2.57 |

## Actions used (probes + submissions), solved episodes only

| model | mean probes | mean actions | solved with <=2 probes |
|---|---|---|---|
| gemma-4-31b | 0.7 | 2.1 | 92% |
| haiku-5.5 | 1.8 | 3.6 | 60% |
| opus-5.5 | 0.9 | 2.3 | 89% |
| sonnet-5.5 | 0.5 | 1.9 | 95% |

## Does feedback help?

| model | first-try | by attempt 2 | by attempt 3 |
|---|---|---|---|
| gemma-4-31b | 20% | 28% | 30% |
| haiku-5.5 | 22% | 38% | 50% |
| opus-5.5 | 70% | 88% | 95% |
| sonnet-5.5 | 35% | 50% | 55% |

## Solve rate by transformation family (all difficulties)

| family | gemma-4-31b | haiku-5.5 | opus-5.5 | sonnet-5.5 |
|---|---|---|---|---|
| bounding_box | 30% (n=10) | 60% (n=10) | 100% (n=10) | 50% (n=10) |
| crop_largest | 100% (n=1) | 100% (n=1) | 100% (n=1) | 100% (n=1) |
| denoise | 17% (n=6) | 33% (n=6) | 83% (n=6) | 50% (n=6) |
| fill_enclosed | 50% (n=4) | 50% (n=4) | 100% (n=4) | 100% (n=4) |
| flip_rotate | 7% (n=15) | 33% (n=15) | 87% (n=15) | 20% (n=15) |
| gravity | 9% (n=11) | 27% (n=11) | 91% (n=11) | 64% (n=11) |
| hollow_out | 25% (n=4) | 25% (n=4) | 100% (n=4) | 50% (n=4) |
| mirror | 0% (n=2) | 50% (n=2) | 100% (n=2) | 100% (n=2) |
| ray | 0% (n=6) | 83% (n=6) | 100% (n=6) | 33% (n=6) |
| recolor_by_size | 38% (n=8) | 50% (n=8) | 88% (n=8) | 50% (n=8) |
| recolor_map | 50% (n=4) | 50% (n=4) | 100% (n=4) | 50% (n=4) |
| swap_most_common | 33% (n=12) | 50% (n=12) | 92% (n=12) | 42% (n=12) |
| translate | 18% (n=11) | 27% (n=11) | 100% (n=11) | 55% (n=11) |
