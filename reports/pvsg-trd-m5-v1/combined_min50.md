# M5 · Tổng hợp có tính hội tụ muộn

GT tracks, 62 video test, K=50, tIoU=0,5. Đơn vị %, trung bình ± độ lệch chuẩn mẫu trên seeds 17, 42, 2026. Cột kết hợp thay đúng bảy cặp model/seed được chọn theo validation bằng kết quả min50.

| Model | Seed thay | R gốc | R kết hợp | mR gốc | mR kết hợp |
|---|---|---:|---:|---:|---:|
| mlp | — | 3.80 ± 0.43 | 3.80 ± 0.43 | 1.62 ± 0.12 | 1.62 ± 0.12 |
| static_gat | — | 3.80 ± 0.65 | 3.80 ± 0.65 | 1.68 ± 0.17 | 1.68 ± 0.17 |
| st_gat | 17, 2026 | 1.65 ± 2.85 | 4.98 ± 0.27 | 0.98 ± 1.70 | 2.38 ± 0.54 |
| pvsg_conv_k5_l3 | — | 8.63 ± 0.61 | 8.63 ± 0.61 | 1.25 ± 0.25 | 1.25 ± 0.25 |
| st_no_temporal | 17, 2026 | 1.25 ± 2.17 | 3.58 ± 1.35 | 0.55 ± 0.95 | 1.28 ± 0.75 |
| st_no_spatial | — | 3.47 ± 1.20 | 3.47 ± 1.20 | 1.07 ± 0.70 | 1.07 ± 0.70 |
| st_no_edges | 17, 42, 2026 | 0.00 ± 0.00 | 3.62 ± 0.16 | 0.00 ± 0.00 | 1.51 ± 0.04 |
| mlp_conv_k3_l2 | — | 6.77 ± 0.65 | 6.77 ± 0.65 | 1.29 ± 0.26 | 1.29 ± 0.26 |

17 lượt gốc và 7 lượt bổ sung tạo thành 24 kết quả trong cột kết hợp. [Chi tiết đủ 7 cặp trước/sau](report.md).
