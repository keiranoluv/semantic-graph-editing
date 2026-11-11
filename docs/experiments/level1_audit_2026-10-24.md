# Kiểm tra nền mức 1 — 2026-10-24

Đây là ghi nhận kiểm tra code, môi trường và artifact, chưa phải đánh giá
full validation hoặc test mới.

## Môi trường kiểm tra

- Linux x86_64, Python 3.12.3, virtualenv `myenv`.
- torch 2.7.0+cpu, torch-geometric 2.6.1, numpy 2.2.6, h5py 3.13.0,
  nltk 3.9.1, pytest 8.3.5.
- `pip check`: không có dependencies hỏng.
- Máy có RTX 5090, driver 590.48.01. Không chạy train GPU trong đợt này.
- Dependencies được pin tại `requirements.txt` và `requirements-dev.txt`.
  Không khẳng định khôi phục đúng môi trường đã train baseline trước đây.

## Dữ liệu

- HDF5 khớp hash tham chiếu trong script download.
- Tạo lại metadata/split vào thư mục tạm và so sánh nội dung: khớp cả hai.
- Split: train 57.723, val 5.000, test 26.446; kiểm tra không trùng row
  nằm trong hàm build manifest.
- Hash các input được ghi tại [TEAM_SETUP_LEVEL1.md](../TEAM_SETUP_LEVEL1.md).
- Không thay đổi file nguồn và không đánh giá held-out test.

## Checkpoint baseline

Đã mở `best.pt` với `weights_only=True` và so sánh config với `config.json`,
epoch với argmax validation mR@50 trong `history.json`.

| Model | Seed | Best epoch | Epochs trong history | Config và best epoch khớp |
|---|---|---|---|---|
| Geometry MLP | 42 | 98 | 100 | Có |
| Geometry MLP | 123 | 99 | 100 | Có |
| Geometry MLP | 3407 | 100 | 100 | Có |
| Semantic MLP | 42 | 88 | 98 | Có |
| Semantic MLP | 123 | 99 | 100 | Có |
| Semantic MLP | 3407 | 92 | 100 | Có |
| GAT | 42 | 31 | 46 | Có |
| GAT | 123 | 100 | 115 | Có |
| GAT | 3407 | 51 | 66 | Có |

Cả 9 config đều ghi train_limit=57.723, val_limit=5.000, box_policy=clip,
batch_size=512 và negative_ratio=3. GAT có accumulation_steps=1.
Config không lưu hash split lịch sử hoặc max_epochs: chưa xác minh được
split thực tế lúc train và ngân sách tối đa chỉ từ các file này.

## Kiểm tra đã chạy

- Full suite ban đầu: **150 passed**.
- Sau bổ sung export/model mới: **163 passed**.
- Ruff, Black check và isort check cho code cũ: PASS.
- MLP real-data CPU smoke: 3 optimizer steps, loss hữu hạn, PASS.
- TransE WordNet transductive dry-run: train/val/test=926/115/115, PASS.
- C.2 export từ checkpoint WordNet transductive epoch 10: 124 class có
  mapping, tensor `[151,64]`, chưa human-reviewed.
- Ba real-data CPU smoke GAT+KG, KG-off, edges-off: mỗi run 8 ảnh train,
  8 ảnh validation, 1 epoch; loss/validation hữu hạn và lưu checkpoint.
- Evaluator chung đã chạy thử trên checkpoint GAT+KG smoke và baseline
  MLP/GAT. Các report 8 ảnh chỉ dùng kiểm tra kỹ thuật.
- Đánh giá lại độc lập **cả 9 checkpoint baseline trên toàn bộ 5.000 ảnh
  validation** bằng evaluator chung. Mọi R@50/100 và mR@50/100 khớp số liệu
  ở best epoch trong history (sai khác dấu phẩy động không đáng kể).
  Các report riêng và file `baselines_3seed_val5000_summary.json` nằm dưới
  `reports/level1/`; xem bảng ở `docs/experiments/level1_baseline_verified.md`.
- Trích 5 hit và 5 missed GT ở R@50 từ GAT seed 42 trên pool 100 ảnh
  validation để smoke công cụ phân tích; chưa có nhận xét thủ công.
- Giả lập ngắt trước lưu checkpoint epoch 2 rồi resume từ `last.pt`:
  history và toàn bộ tensors của model khớp chính xác run CPU 2 epochs
  liên tục cùng cấu hình. Không suy ra đảm bảo tương đương GPU.

Không báo PASS full pre-commit workflow vì chưa chạy toàn bộ các hooks.
Không suy ra chất lượng mô hình từ các smoke metrics.
