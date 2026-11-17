| Cấu hình | R@50 | R@100 | mR@50 | mR@100 |
|---|---:|---:|---:|---:|
| loss_ce_off | 64.42 ± 0.10 | 66.85 ± 0.09 | 18.11 ± 0.16 | 20.56 ± 0.19 |
| loss_ce_on | 63.63 ± 0.14 | 65.93 ± 0.13 | 19.27 ± 0.75 | 21.51 ± 0.91 |
| loss_adjusted_off | 62.18 ± 0.16 | 64.81 ± 0.13 | 24.59 ± 0.93 | 27.89 ± 1.02 |
| loss_adjusted_on | 62.01 ± 0.31 | 64.71 ± 0.46 | 24.67 ± 1.07 | 28.00 ± 1.19 |
| fusion_baseline | 62.06 ± 0.39 | 64.83 ± 0.21 | 24.32 ± 0.91 | 27.72 ± 1.08 |
| fusion_pair_real | 62.37 ± 0.19 | 65.01 ± 0.20 | 25.24 ± 0.39 | 28.65 ± 0.27 |
| fusion_pair_shuffled | 62.34 ± 0.18 | 65.06 ± 0.24 | 25.00 ± 0.71 | 28.33 ± 0.74 |
| fusion_visual_real | 62.34 ± 0.21 | 65.05 ± 0.22 | 24.52 ± 0.64 | 27.98 ± 0.73 |
| fusion_visual_shuffled | 62.20 ± 0.23 | 64.96 ± 0.22 | 24.82 ± 1.25 | 27.94 ± 1.10 |

| Treatment − control | ΔR@50 | ΔmR@50 | ΔmR@50 seed 42 / 123 / 3407 |
|---|---:|---:|---|
| loss_adjusted_off − loss_ce_off | -2.235 | +6.479 | +6.081 / +7.710 / +5.645 |
| loss_adjusted_on − loss_ce_on | -1.612 | +5.399 | +6.384 / +3.771 / +6.042 |
| loss_ce_on − loss_ce_off | -0.792 | +1.166 | +0.254 / +1.848 / +1.395 |
| loss_adjusted_on − loss_adjusted_off | -0.169 | +0.086 | +0.557 / -2.090 / +1.793 |
| fusion_pair_real − fusion_baseline | +0.306 | +0.926 | +1.783 / -0.560 / +1.556 |
| fusion_pair_real − fusion_pair_shuffled | +0.025 | +0.238 | +0.400 / -0.823 / +1.135 |
| fusion_visual_real − fusion_baseline | +0.278 | +0.206 | +0.297 / -1.120 / +1.439 |
| fusion_visual_real − fusion_visual_shuffled | +0.142 | -0.300 | -0.917 / +0.752 / -0.735 |
