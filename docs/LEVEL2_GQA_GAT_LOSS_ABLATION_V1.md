# GQA · GAT tắt KG với CE và logit-adjusted loss

Kiểm tra trên custom GQA xem điều chỉnh loss từng giúp cải thiện mR@50 trên VG150 có lặp lại hay không. Đơn vị so sánh là hai lượt GAT với cùng seed, cùng khởi tạo và cùng chuỗi cặp huấn luyện trong các epoch chung. Dùng ba kết quả CE đã hoàn tất từ GQA v3 (PyTorch 2.7.0+cu128). Nhánh adjusted dùng venv riêng với PyTorch 2.7.0+cu128, torchvision 0.22.0+cu128, PyG 2.6.1 và W&B 0.18.7. Launcher đối chiếu hash kết quả và checkpoint CE trước khi train, sau đó kiểm tra khởi tạo và chuỗi lấy mẫu ở các epoch chung. [Cấu hình GQA v3](../configs/level2/gqa-full-v3-bs512.json) · [Đối chứng VG150](../src/level1_pair_kg_matrix.py)

Từ gốc repo, dùng môi trường GQA và tài khoản W&B đã đăng nhập:

```bash
bash scripts/run_gqa_loss_ablation_v1.sh train
```

Launcher chỉ train adjusted, tuần tự các seed 42, 123, 3407 rồi dừng. Cùng 69.942 ảnh train, 5.000 ảnh validation, cache CLIP đóng băng, batch 512 ảnh cho cả train/val, AdamW với LR 0,001, weight decay 0,01, clip gradient norm 5. GAT có 2 lớp, 4 head, hidden 256, dropout 0,1. Tối đa 200 epoch, patience 15, min delta 0. Chọn checkpoint theo validation GC mR@50, giữ epoch sớm nhất khi hòa.

Adjusted loss dùng CE với soft target và logits cộng 0,25 × log(prior) trong bước tối ưu. Prior có 311 lớp gồm background. Mỗi cặp dương chia đều một đơn vị khối lượng cho các predicate riêng biệt. Đếm background đúng theo sampler (tối đa 3 lần số cặp dương, hoặc tối đa 32 khi ảnh không có cặp dương), rồi cộng 1 mỗi lớp trước khi chuẩn hóa. Chỉ ước lượng prior từ dữ liệu train. Khi chấm cả hai mô hình, evaluator nhận logits gốc.

Output ở `runs/level2/gqa-gat-loss-v3-cu128/`. Mỗi seed trong thư mục `adjusted/` có `best.pt`, `last.pt` và kết quả validation riêng. File `validation-comparison.json` ghi chênh lệch adjusted trừ CE theo điểm phần trăm cho GC R/mR@50/100 sau mỗi cặp; bảng cuối có mean và sample SD của ba hiệu số. Đánh giá test là bước riêng sau khi chốt ba checkpoint adjusted. W&B online ghi từng epoch. Đường dẫn log trong `logs/` được in khi khởi động. Ctrl+C hoặc SIGTERM cho phép lưu checkpoint; chạy lại cùng lệnh để tiếp tục.

Các smoke test dùng seed 42 và dữ liệu tạm cho nhánh adjusted. CPU dùng 16 ảnh mỗi split, GPU dùng 512 ảnh mỗi split để thử một batch đủ kích thước, một epoch. Launcher tự dọn thư mục checkpoint tạm khi kết thúc. Kết quả smoke dùng để kiểm tra luồng thực thi.

```bash
bash scripts/run_gqa_loss_ablation_v1.sh smoke-cpu
bash scripts/run_gqa_loss_ablation_v1.sh smoke-gpu
```


### Dựng lại môi trường

Launcher dùng trực tiếp Python trong `data/cache/gqa-gat-loss-torch270/venv`. Để dựng trên Linux x86_64 với CPython 3.12 từ gốc repo:

```bash
python3.12 -m venv data/cache/gqa-gat-loss-torch270/venv
data/cache/gqa-gat-loss-torch270/venv/bin/python -m pip install -r requirements-gqa-gat-loss-cu128.txt
data/cache/gqa-gat-loss-torch270/venv/bin/python -m pip check
```

Các pin chính lấy từ snapshot requirements trong W&B của GAT GQA v3. Snapshot ghi torchvision 0.22.0; môi trường này chọn wheel CUDA 12.8 cho cùng phiên bản. Launcher kiểm tra phiên bản trước khi nạp dữ liệu và ghi runtime vào experiment.json.
