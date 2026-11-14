# MagicBrush manual experiment v2

V1 hoàn tất chương trình nhưng cả sáu lượt có F1 validation bằng 0, mọi epoch đều có 67 lỗi giải mã, và chọn checkpoint epoch 1. V2 giữ nguyên dữ liệu, kiến trúc và codec; sửa ngân sách huấn luyện, chọn checkpoint và điều kiện cho phép hàng đợi chạy tiếp. Chưa có kết quả GPU v2 tại thời điểm bàn giao.

## Chạy

```bash
bash /workspace/workspace/semantic-graph-editing/scripts/run_magicbrush_manual_full602_v2.sh
```

Launcher dùng `myenv/bin/python`, W&B offline, một model trên GPU mỗi thời điểm. Có thể dùng `--wandb-mode disabled` nếu không cần lưu W&B offline. Không cài thêm thư viện. Không đọc checkpoint v1 để khởi tạo: mọi lượt v2 học từ đầu theo seed đã định.

| Tham số | Giá trị |
| --- | --- |
| Mô hình | MLP và GNN; cùng attentive GRU byte decoder của v1 |
| Đặc trưng | CLIP ViT-B/32 text đóng băng, chỉ graph nguồn và câu yêu cầu |
| Dữ liệu | 301 train, 67 validation, 69 test; giữ nguyên bộ 437 cặp được chấp nhận |
| Train batch / validation-test batch / CLIP batch | 16 / 32 / 128 |
| Seed | 42, 123, 3407 |
| Epoch | Tối đa 300; không dừng sớm trước 80 |
| AdamW | LR 0,0003; weight decay 0,01; gradient clip 5 |
| Validation | Epoch 1, mỗi 5 epoch, mốc tối thiểu và epoch cuối |
| Patience | 40 epoch không cải thiện thứ tự chọn checkpoint |
| Chọn checkpoint | F1 thao tác → tỷ lệ transaction hợp lệ → loss token validation thấp hơn; hòa cả ba giữ bản sớm hơn |
| Ngưỡng chạy tiếp | Tỷ lệ transaction hợp lệ ≥ 0,50 **và** F1 ≥ 0,01 trên validation |

Với 301 mẫu và batch 16, mỗi epoch có 19 bước cập nhật: tối thiểu 1.520, tối đa 5.700 bước/lượt. Loss validation dùng teacher forcing chỉ để phân xử điểm hòa, không thay thế việc đánh giá sinh tự hồi quy. Không tự sửa JSON, không dùng nhãn đích để ràng buộc giải mã.

## Hàng đợi và kết quả

Thứ tự: `MLP/42 → GNN/42 → MLP/123 → MLP/3407 → GNN/123 → GNN/3407`. Hai lượt đầu là pilot với cùng cấu hình và ngân sách như những lượt còn lại; chúng được tính trong tổng sáu lượt, không chạy lại sau khi đạt ngưỡng. Mỗi lượt hoàn tất đều phải vượt ngưỡng validation trước khi chạy lượt kế tiếp. Nếu MLP/42 không đạt, GNN/42 cũng chưa được chạy.

Nếu không đạt, launcher trả mã lỗi và ghi `quality_gate_failed`, giữ checkpoint, lịch sử và dự đoán để chẩn đoán; không chạy tiếp và không mở nhãn test. Chạy lại cùng lệnh không bỏ qua lỗi chất lượng. Các ngưỡng này là kiểm tra tối thiểu của cơ chế dự đoán, không phải chứng nhận chất lượng tổng quát hóa.

Chỉ sau khi sáu lượt đạt ngưỡng và được khóa checkpoint, chương trình mới đọc nhãn test, tạo cache đầu vào test, đánh giá sáu checkpoint và tổng hợp mean/sample standard deviation. Kết quả cuối cũng được in ra terminal.

- Output: `runs/level2/magicbrush-manual-experiment-v2/`.
- Báo cáo cuối: `summary.json`, `SUMMARY.txt`; trạng thái hiện tại: `progress.json`.
- Mỗi lượt: `<model>/<seed>/{best.pt,last.pt,result.json,predictions-val.json,completed-receipt.json}`; các file test chỉ xuất hiện sau hàng rào kiểm tra.
- Log: `logs/magicbrush-manual-experiment-v2/launcher.log`.
- Cache riêng: `data/processed/level2/magicbrush-manual-experiment-v2/{clip-trainval,clip-test}`.

Nếu mất kết nối hoặc tiến trình bị ngắt, chạy lại cùng lệnh để tiếp tục từ epoch đã lưu. Trạng thái model, optimizer, RNG và trọng số checkpoint được chọn nằm chung trong một file lưu nguyên tử. Lượt đã hoàn tất được kiểm tra hash và bỏ qua; lỗi chất lượng vẫn dừng hàng đợi. Launcher dùng chung khóa GPU với các thí nghiệm của dự án.

## Provenance và giới hạn

Config và release lock nằm ở `configs/level2/magicbrush-manual-experiment-v2{,.lock}.json`; code mới chỉ ở `tools/magicbrush_experiment_v2/`. Các tiện ích v1 được tái sử dụng qua import và khóa hash; không sửa `src/`, code/config/manifest v1 hay notebook cá nhân.

V1 đã đánh giá cùng tập test này; không mô tả nó như một tập test mới chưa từng được xem. Các thay đổi v2 dựa trên lỗi validation của v1, và việc chọn checkpoint/ngưỡng chạy tiếp v2 chỉ dùng validation. Giữ nguyên bộ chấm per-operation PRF1, exact match, tỷ lệ lỗi, baseline không chỉnh sửa và thống kê nhiều seed. Các nhãn là AI-assisted; xác nhận tay 20/20 không được suy rộng thành độ chính xác cho toàn bộ 437 mẫu.

`preflight.json` ghi kiểm tra dữ liệu, hash và môi trường trên CPU. `verification.json` ghi kết quả kiểm thử. Các thử nghiệm học một ví dụ giả lập chỉ xác nhận cơ chế huấn luyện và giải mã chạy được; không chứng minh rằng tăng ngân sách sẽ khắc phục chất lượng trên MagicBrush. Assistant chưa chạy tác vụ GPU nào cho v2.
