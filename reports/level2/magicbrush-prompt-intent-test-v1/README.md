# Task B, nhiệm vụ 2 · Kết quả MagicBrush theo prompt

Bộ dữ liệu gồm 240 cặp train, 46 validation và 59 test. Người duyệt xác nhận 20/20 mẫu lấy ngẫu nhiên từ bộ 345 cặp đạt tiêu chí nhãn. Hai nhánh dùng cùng GNN và CLIP text đóng băng. Mỗi checkpoint được chọn trên validation và chấm trên cùng 59 cặp test với 102 thao tác đích theo prompt. Một thao tác được tính đúng khi loại thao tác, đối tượng và toàn bộ nội dung chỉnh sửa khớp nhãn sau khi chuẩn hóa ID.

## Precision, recall và F1 trên test

Các giá trị là phần trăm, trung bình của ba seed 42, 123 và 3407. Cột số thao tác đếm nhãn chuẩn trên 59 cặp test một lần.

| Loại thao tác | Số thao tác | P · graph diff | R · graph diff | F1 · graph diff | P · prompt | R · prompt | F1 · prompt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| micro | 102 | 5.30 | 8.50 | 6.53 | 5.35 | 3.92 | 4.45 |
| add_node | 24 | 0.00 | 0.00 | 0.00 | 5.95 | 5.56 | 5.73 |
| delete_node | 12 | 26.51 | 16.67 | 20.29 | 16.67 | 2.78 | 4.76 |
| replace_node | 35 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| add_edge | 17 | 2.39 | 5.88 | 3.39 | 2.56 | 1.96 | 2.22 |
| delete_edge | 14 | 25.50 | 40.48 | 30.68 | 35.68 | 14.29 | 18.84 |
| replace_edge | 0 | — | — | — | — | — | — |

Tập test có 0 nhãn replace_edge và cả sáu lượt đều dự đoán 0 thao tác thuộc loại này. Với replace_node, cả hai nhánh đạt F1 bằng 0 trên 35 thao tác chuẩn.

## Kết quả từng lượt

| Nhãn train | Seed | Epoch đã chạy | Epoch được chọn | Micro-F1 validation (%) | Micro-F1 test (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| graph_diff | 42 | 40 | 10 | 7.24 | 4.53 |
| prompt | 42 | 40 | 10 | 6.13 | 3.88 |
| graph_diff | 123 | 40 | 10 | 9.28 | 7.75 |
| prompt | 123 | 40 | 10 | 5.52 | 3.23 |
| graph_diff | 3407 | 42 | 12 | 7.96 | 7.30 |
| prompt | 3407 | 58 | 28 | 7.09 | 6.25 |

Nhánh theo prompt có micro-F1 test thấp hơn graph diff ở cả ba seed. Precision trung bình của hai nhánh xấp xỉ nhau, nhưng recall giảm từ 8,50% xuống 3,92%. Cả hai nhánh có 100% giao dịch hợp lệ về cấu trúc. Tỷ lệ khớp hoàn toàn đồ thị đích đạt 0,00% với graph diff và 0,56% với nhãn theo prompt, tính trung bình ba seed.

Chi tiết kết quả và số đếm TP/FP/FN nằm trong [summary.json](summary.json). Danh sách checkpoint cùng SHA-256 được ghi trong [selection.json](selection.json). Có thể tái lập từ thư mục gốc trên CPU bằng lệnh sau.

```bash
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m experiments.magicbrush_prompt_intent_v1.evaluate
```
