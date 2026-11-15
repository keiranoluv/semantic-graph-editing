# AlignG + PE-PLPR — custom GQA batch512, hàng đợi qua đêm

User yêu cầu nâng toàn bộ batch train/validation lên512 và một lệnh tự chạy
đủ các lượt. Namespace mới giữ nguyên source/config/run batch16 và P0/P1;
không thay đổi `src/`, checkpoint, cache hoặc notebook cá nhân.

## Recipe

Config: `configs/level2/aligng-gqa-full-v2-bs512.json`.
Custom GQA69942train/5000val, C311, CLIP/GloVe/split/sampler/loss/evaluator cũ.
Train **512**, validation **512**, effective **512**, không gradient accumulation.
AdamW lr0.001, weight_decay0.01, clip5; float32/deterministic, không AMP/TF32.
D256/2heads/GRU/dropout0.2, context đủ không phụ thuộc nhãn; P0 adapter nguyên vẹn.
Max200epoch, early stopping patience15/min_delta0, best validation GCmR@50,
hòa giữ epoch sớm hơn. Checkpoint mỗi10batch và cuối epoch/khi yêu cầu dừng.

Thứ tự tự động: AlignG42 → PE-PLPR42 → AlignG123 → PE-PLPR123 →
AlignG3407 → PE-PLPR3407. Một model/GPU tại mỗi thời điểm; giữ shared GPU lock
suốt queue. W&B **online bắt buộc** `keiranoluv-hcmus/semantic-graph-editing`.
Không fallback offline hoặc chạy tiếp khi lỗi. Không huấn luyện lại5model
GQA lịch sử đã dùng512; queue này chứa hai biến thể so sánh AlignG hiện tại.

Output mới: `runs/level2/aligng-gqa-bs512-v1/full/{model}/{seed}/`.
Khởi tạo thí nghiệm mới với batch512, không resume trọng số/AdamW batch16.
Chạy lại cùng lệnh: verify rồi skip lượt hoàn tất, resume lượt chưa xong,
hoặc bắt đầu lượt tiếp theo. Source/config/identity phải giữ nguyên.

## Một lệnh do người dùng chạy sau commit và GPU capacity PASS

Từ repo root:

```bash
bash scripts/run_aligng_gqa_bs512.sh
```

Lệnh kiểm traCPU/source/cache/pilot/capacity trước rồi tạo tmux detached
`aligng-gqa-bs512`. Terminal có thể đóng sau khi lệnh thông báo Started;
giữ máy hoạt động. Nếu session tồn tại thì không khởi tạo queue trùng.

Theo dõi bằng `tail -f logs/aligng-gqa-bs512.log` hoặc
`tmux attach -t aligng-gqa-bs512`. Trạng thái bền vững:
`runs/level2/aligng-gqa-bs512-v1/full/queue-status.json`.
Ctrl+C trong tmux yêu cầu checkpoint và dừng cả queue; job kế không tự bắt
đầu sau lệnh dừng. Với lỗi (OOM, W&B, identity...), queue dừng và ghi failed.

Sau đủ6lượt, summary mean/sampleSD qua3seed được xuất; kiểm sampler prefix
và initial state matched. Không tự chạy test hoặc VG150.

## Kiểm tra trước khi bàn giao

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 myenv/bin/python -m pytest -q tests/aligng_gqa_bs512_v1 tests/aligng_gqa_v1 tests/aligng_adapted_v1 tests/training/test_gqa_full.py tests/training/test_gqa_tracking.py tests/training/test_penet.py -k 'not cuda'
myenv/bin/python -m experiments.aligng_gqa_bs512_v1.runner --stage plan
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 myenv/bin/python -u -m experiments.aligng_gqa_bs512_v1.probe
myenv/bin/python -m experiments.aligng_gqa_bs512_v1.runner --stage check
```

CPU kiểm cả queue thứ tự6lượt, summary, restart/skip, checkpoint mutation,
interruption/error không nối job và wiring capacity probe. GPU probe được
assistant thực hiện theo quyền test đã cấp, sau khi code đã commit.
Dùng ba batch thật512ảnh từ audit cũ: max context, max sampled query ở train
qua first shuffle seeds42/123/3407, và max context của validation. Giữ cache
và case source hash. Mỗi model/case1warmup+2measured steps, temporary optimizer
updates khi train; không W&B, không ghi checkpoint train chính. Validation
probe gồm forward/CE, không tính CPU ranking. Đây không phải cận VRAM cho
mọi shuffle/epoch có thể có. Kết quả lưu tại
`runs/level2/aligng-gqa-bs512-v1/capacity.json`; thiếu/khác identity thì queue
không chạy. Báo cáo nhỏ riêng sẽ lưu evidence trong Git.

Source fingerprint mở rộng qua các file mới và launcher shell, không làm
thay đổi identity P1. Hai cặp pilot P1 được xác minh read-only trước khi
chạy, không ghi đè summary cũ. Main vẫn dùng engine P1 đã kiểm exact resume,
W&B log sau durable checkpoint và loss/evaluator gốc.
