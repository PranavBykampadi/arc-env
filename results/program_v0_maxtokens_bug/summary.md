# Results (program mode)

Solve rate = solved within the attempt limit, with a 95% Wilson interval. First-try = solved on the first submission. Reward = mean episode reward (passive: attempt-decayed solve; active: solved x min(1, 4/actions)^2; program: hidden-test pass fraction of the final submission). Cost uses list prices.

| model | hard solve [95% CI] / first-try / reward | expert solve [95% CI] / first-try / reward | n | $ |
|---|---|---|---|---|
| fable-5.1 | - | 43% [16%-75%] / 14% / 0.39 | 7 | 3.41 |
| gemma-4-31b | 60% [39%-78%] / 5% / 0.52 | 10% [3%-30%] / 0% / 0.15 | 40 | 0.00 |
| haiku-5.5 | 50% [30%-70%] / 50% / 0.52 | 20% [8%-42%] / 15% / 0.24 | 40 | 0.15 |
| opus-5.5 | 95% [76%-99%] / 75% / 0.93 | 45% [26%-66%] / 20% / 0.45 | 40 | 4.46 |
| sonnet-5.5 | 80% [58%-92%] / 55% / 0.75 | 30% [15%-52%] / 10% / 0.30 | 40 | 3.03 |

## Generalization: hidden-test pass fraction

| model | final submission (mean) | best attempt (mean) | all-hidden-pass rate |
|---|---|---|---|
| fable-5.1 | 0.43 | 0.43 | 43% |
| gemma-4-31b | 0.42 | 0.42 | 35% |
| haiku-5.5 | 0.40 | 0.40 | 35% |
| opus-5.5 | 0.76 | 0.78 | 70% |
| sonnet-5.5 | 0.58 | 0.64 | 55% |

## Does feedback help?

| model | first-try | by attempt 2 | by attempt 3 |
|---|---|---|---|
| fable-5.1 | 14% | 43% | 43% |
| gemma-4-31b | 2% | 25% | 35% |
| haiku-5.5 | 32% | 35% | 35% |
| opus-5.5 | 48% | 60% | 70% |
| sonnet-5.5 | 32% | 48% | 55% |

## Solve rate by transformation family (all difficulties)

| family | fable-5.1 | gemma-4-31b | haiku-5.5 | opus-5.5 | sonnet-5.5 |
|---|---|---|---|---|---|
| bounding_box | 50% (n=4) | 27% (n=11) | 36% (n=11) | 45% (n=11) | 45% (n=11) |
| crop_largest | - | 50% (n=2) | 50% (n=2) | 50% (n=2) | 50% (n=2) |
| denoise | - | 33% (n=6) | 33% (n=6) | 67% (n=6) | 50% (n=6) |
| fill_enclosed | - | 25% (n=4) | 50% (n=4) | 100% (n=4) | 75% (n=4) |
| flip_rotate | 67% (n=3) | 15% (n=13) | 0% (n=13) | 54% (n=13) | 31% (n=13) |
| gravity | 67% (n=3) | 14% (n=7) | 14% (n=7) | 57% (n=7) | 29% (n=7) |
| hollow_out | - | 50% (n=6) | 50% (n=6) | 100% (n=6) | 67% (n=6) |
| mirror | - | 100% (n=1) | 100% (n=1) | 100% (n=1) | 100% (n=1) |
| ray | 50% (n=2) | 14% (n=7) | 0% (n=7) | 14% (n=7) | 14% (n=7) |
| recolor_by_size | 0% (n=1) | 30% (n=10) | 40% (n=10) | 70% (n=10) | 60% (n=10) |
| recolor_map | 0% (n=2) | 29% (n=7) | 57% (n=7) | 71% (n=7) | 57% (n=7) |
| swap_most_common | 25% (n=4) | 9% (n=11) | 18% (n=11) | 55% (n=11) | 36% (n=11) |
| translate | 50% (n=2) | 33% (n=9) | 33% (n=9) | 89% (n=9) | 67% (n=9) |
