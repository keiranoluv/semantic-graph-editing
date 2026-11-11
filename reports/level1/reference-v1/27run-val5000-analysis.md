# main: reference-v1 full validation

5.000 ảnh; 3 seed; đơn vị %; mean ± sample std. Checkpoint chọn theo metric legacy, giữ nguyên để chấm lại.

| Điều kiện | R@50 | mR@50 | R@100 | mR@100 |
|---|---:|---:|---:|---:|
| mlp_semantic | 64.096 ± 0.817 | 17.208 ± 1.165 | 66.608 ± 0.346 | 19.528 ± 1.245 |
| gat | 64.459 ± 0.934 | 19.552 ± 0.137 | 66.522 ± 0.948 | 21.396 ± 0.070 |
| kg_off | 64.358 ± 0.482 | 19.008 ± 0.206 | 66.377 ± 0.529 | 20.959 ± 0.166 |
| wordnet_kg_on | 64.784 ± 0.475 | 18.464 ± 0.652 | 66.979 ± 0.404 | 20.489 ± 0.318 |
| wordnet_edges_off | 65.117 ± 1.130 | 19.474 ± 1.246 | 67.150 ± 1.148 | 21.340 ± 1.161 |
| conceptnet_transe_kg_on | 64.824 ± 0.291 | 19.310 ± 0.593 | 66.866 ± 0.303 | 21.330 ± 0.810 |
| conceptnet_transe_edges_off | 65.511 ± 0.343 | 19.487 ± 0.936 | 67.591 ± 0.344 | 21.320 ± 0.844 |
| conceptnet_rotate_kg_on | 64.707 ± 0.676 | 18.753 ± 0.532 | 66.792 ± 0.717 | 20.414 ± 0.451 |
| conceptnet_complex_kg_on | 64.934 ± 0.351 | 19.049 ± 0.226 | 66.993 ± 0.437 | 21.023 ± 0.281 |

## Chênh lệch ghép cặp theo seed (điểm phần trăm)

| So sánh | Δ mR@50 seed 42 / 123 / 3407 | Mean Δ mR@50 | Mean Δ R@50 |
|---|---|---:|---:|
| gat - mlp_semantic | +1.270 / +3.439 / +2.322 | +2.344 | +0.363 |

Chỉ so sánh trực tiếp trong cùng family/protocol. Không dùng validation này như test độc lập hoặc chọn một seed có lợi.
