# Results (program mode)

Solve rate = solved within the attempt limit, with a 95% Wilson interval. First-try = solved on the first submission. Reward = mean episode reward (passive: attempt-decayed solve; active: solved x min(1, 4/actions)^2; program: hidden-test pass fraction of the final submission). Cost uses list prices.

| model | hard solve [95% CI] / first-try / reward | expert solve [95% CI] / first-try / reward | n | $ |
|---|---|---|---|---|
| gemma-4-31b | 47% [25%-70%] / 0% / 0.44 | 20% [7%-45%] / 0% / 0.22 | 30 | 0.00 |
| haiku-5.5 | 53% [30%-75%] / 47% / 0.55 | 20% [7%-45%] / 13% / 0.26 | 30 | 0.17 |
| opus-5.5 | 93% [70%-99%] / 80% / 0.94 | 67% [42%-85%] / 60% / 0.77 | 30 | 4.02 |
| sonnet-5.5 | 73% [48%-89%] / 47% / 0.77 | 40% [20%-64%] / 13% / 0.43 | 30 | 1.80 |

## Generalization: hidden-test pass fraction

| model | final submission (mean) | best attempt (mean) | all-hidden-pass rate |
|---|---|---|---|
| gemma-4-31b | 0.50 | 0.50 | 38% |
| haiku-5.5 | 0.44 | 0.45 | 37% |
| opus-5.5 | 0.90 | 0.90 | 80% |
| sonnet-5.5 | 0.67 | 0.69 | 57% |

## Does feedback help?

| model | first-try | by attempt 2 | by attempt 3 |
|---|---|---|---|
| gemma-4-31b | 0% | 20% | 33% |
| haiku-5.5 | 30% | 37% | 37% |
| opus-5.5 | 70% | 80% | 80% |
| sonnet-5.5 | 30% | 53% | 57% |

## Solve rate by transformation family (all difficulties)

| family | gemma-4-31b | haiku-5.5 | opus-5.5 | sonnet-5.5 |
|---|---|---|---|---|
| bounding_box | 30% (n=10) | 30% (n=10) | 50% (n=10) | 40% (n=10) |
| crop_largest | 100% (n=1) | 100% (n=1) | 100% (n=1) | 100% (n=1) |
| denoise | 25% (n=4) | 25% (n=4) | 50% (n=4) | 25% (n=4) |
| fill_enclosed | 0% (n=2) | 50% (n=2) | 100% (n=2) | 50% (n=2) |
| flip_rotate | 9% (n=11) | 9% (n=11) | 91% (n=11) | 45% (n=11) |
| gravity | 17% (n=6) | 17% (n=6) | 83% (n=6) | 50% (n=6) |
| hollow_out | 67% (n=3) | 67% (n=3) | 100% (n=3) | 100% (n=3) |
| ray | 40% (n=5) | 0% (n=5) | 60% (n=5) | 20% (n=5) |
| recolor_by_size | 38% (n=8) | 38% (n=8) | 88% (n=8) | 62% (n=8) |
| recolor_map | 33% (n=6) | 67% (n=6) | 67% (n=6) | 67% (n=6) |
| swap_most_common | 12% (n=8) | 12% (n=8) | 75% (n=8) | 50% (n=8) |
| translate | 33% (n=6) | 33% (n=6) | 83% (n=6) | 50% (n=6) |
