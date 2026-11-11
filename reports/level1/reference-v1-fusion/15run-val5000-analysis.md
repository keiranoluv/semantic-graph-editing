# fusion: reference-v1 full validation

5.000 ảnh; 3 seed; đơn vị %; mean ± sample std. Checkpoint chọn theo metric legacy, giữ nguyên để chấm lại.

| Điều kiện | R@50 | mR@50 | R@100 | mR@100 |
|---|---:|---:|---:|---:|
| baseline | 61.957 ± 0.668 | 25.779 ± 0.994 | 64.469 ± 0.502 | 28.418 ± 0.861 |
| pair_real | 62.645 ± 0.411 | 25.718 ± 0.302 | 64.888 ± 0.451 | 27.921 ± 0.289 |
| pair_shuffled | 62.398 ± 0.133 | 25.847 ± 0.093 | 64.774 ± 0.301 | 28.295 ± 0.357 |
| visual_real | 62.813 ± 0.093 | 25.199 ± 0.304 | 65.162 ± 0.059 | 27.683 ± 0.502 |
| visual_shuffled | 62.823 ± 0.495 | 25.972 ± 0.913 | 65.299 ± 0.103 | 28.710 ± 0.048 |

## Chênh lệch ghép cặp theo seed (điểm phần trăm)

| So sánh | Δ mR@50 seed 42 / 123 / 3407 | Mean Δ mR@50 | Mean Δ R@50 |
|---|---|---:|---:|
| pair_real - baseline | +0.415 / -0.936 / +0.337 | -0.061 | +0.688 |
| pair_real - pair_shuffled | +0.055 / +0.077 / -0.520 | -0.129 | +0.247 |
| visual_real - baseline | -0.230 / -1.386 / -0.124 | -0.580 | +0.856 |
| visual_real - visual_shuffled | -1.119 / +0.576 / -1.776 | -0.773 | -0.010 |

Không đưa 6 run relation prior vào bảng này do bất nhất L1/L2. Không suy ra kết luận cho nhánh bị loại.

Chỉ so sánh trực tiếp trong cùng family/protocol. Không dùng validation này như test độc lập hoặc chọn một seed có lợi.
