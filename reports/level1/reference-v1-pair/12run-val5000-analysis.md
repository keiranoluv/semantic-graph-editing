# pair: reference-v1 full validation

5.000 ảnh; 3 seed; đơn vị %; mean ± sample std. Checkpoint chọn theo metric legacy, giữ nguyên để chấm lại.

| Điều kiện | R@50 | mR@50 | R@100 | mR@100 |
|---|---:|---:|---:|---:|
| ce_off | 65.156 ± 0.104 | 19.177 ± 0.573 | 67.330 ± 0.107 | 20.902 ± 0.423 |
| ce_on | 64.054 ± 0.149 | 19.749 ± 0.995 | 66.170 ± 0.263 | 21.676 ± 1.154 |
| adjusted_off | 61.970 ± 0.123 | 25.800 ± 1.102 | 64.312 ± 0.328 | 27.881 ± 0.943 |
| adjusted_on | 61.966 ± 0.423 | 25.968 ± 0.614 | 64.394 ± 0.665 | 28.480 ± 0.738 |

## Chênh lệch ghép cặp theo seed (điểm phần trăm)

| So sánh | Δ mR@50 seed 42 / 123 / 3407 | Mean Δ mR@50 | Mean Δ R@50 |
|---|---|---:|---:|
| ce_on - ce_off | -0.453 / +2.044 / +0.125 | +0.572 | -1.102 |
| adjusted_off - ce_off | +6.171 / +7.952 / +5.746 | +6.623 | -3.186 |
| adjusted_on - adjusted_off | +0.426 / -1.511 / +1.588 | +0.168 | -0.003 |

Chỉ so sánh trực tiếp trong cùng family/protocol. Không dùng validation này như test độc lập hoặc chọn một seed có lợi.
