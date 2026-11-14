# GQA AlignG batch512 — sẵn sàng hàng đợi6lượt

User yêu cầu tất cả batch train/validation512, commit trước và một lệnh tự
chạy toàn bộ hai model × ba seed qua đêm. Assistant chỉ chạy bounded GPU
capacity check sau commit code, **chưa khởi chạy train chính hoặc W&B**.

Code/config/launcher commit: `2f17cb4`. Scope source mới
`experiments/aligng_gqa_bs512_v1`, giữ nguyên toàn bộ P0/P1 và `src/`.
Config `configs/level2/aligng-gqa-full-v2-bs512.json`:
train/val/effective **512/512/512**, không accumulation, AdamW0.001/decay0.01,
max200epoch/patience15, best val GCmR@50, checkpoint10batch. Custom GQA
69942train/5000val, C311; giữ cache/GloVe/sampler/loss/evaluator.

## Bằng chứng

- CPU **125 passed,0 failed,7 deselected CUDA**. Queue thực trên graph
  synthetic6lượt, resume/skip, summary mean/SD, fail/stop không nối job,
  tampered checkpoint rejection; kế thừa kiểm P0/P1/loss/tracker/resume.
- Black/isort/Ruff, shell syntax và commit hooks PASS. Smoke tmux detached
  chạy trên socket/session riêng chỉ ghi handshake, không khởi chạy training.
- `runner --stage check` CPU đã READY sau kiểm source/cache, hai cặp pilot
  cũ và probe512; không tạo online W&B run trong kiểm này.
- GPU RTX5090: **6/6 case PASS**. Hai model ×3batch thật512ảnh: train max
  context193448pairs, train max sampled117012querypairs, val197568pairs.
  Dùng case selection audit cũ và xác minh cùng cache; không đọc test.
  Mỗi case1warmup+2measured. Temporary optimizer updates khi train; không
  ghi checkpoint train chính. Loss/gradient hữu hạn, không OOM.

| Model | Max train allocated GiB | Max train reserved GiB | Val allocated GiB |
| --- | ---: | ---: | ---: |
| pe-plpr-aligng | 6.211 | 6.982 | 1.368 |
| pe-plpr | 7.793 | 8.895 | 2.594 |

Đây là bộ nhớ tensor/allocator CUDA trên các batch được chọn, không phải
cận trên cho mọi shuffle và không gồm CPU ranking/matcher ở validation.
Chi tiết loss, norm, thời gian, image indices ở [capacity.json](capacity.json).
Không dùng phép đo này để tuyên bố cải thiện chất lượng hay tốc độ32lần.

## Một lệnh user chạy

Từ repo root:

```bash
bash scripts/run_aligng_gqa_bs512.sh
```

Lệnh CPU preflight rồi tạo tmux detached `aligng-gqa-bs512`. Tự chạy tuần tự
**AlignG42 → PE-PLPR42 → AlignG123 → PE-PLPR123 → AlignG3407 → PE-PLPR3407**.
Mỗi lúc một model, W&B online bắt buộc. Stop/error dừng cả queue; không retry
hoặc tự hạ batch. Khi chạy lại, verify/skip lượt hoàn tất và resume lượt dở.
Các lỗi được lưu vào `queue-status.json` và log. Không hứa hoàn tất trong
một đêm; thời gian phụ thuộc hội tụ/early stopping.

Output mới `runs/level2/aligng-gqa-bs512-v1/full/{model}/{seed}/`;
log `logs/aligng-gqa-bs512.log`. Không resume checkpoint batch16 vào recipe512.
Không chạy lại MLP/GAT/PE-CE/PE-PL/GAT+KG cũ đã dùng512; không tự chạy test/VG150.
Đọc [hướng dẫn launcher](../../../experiments/aligng_gqa_bs512_v1/README.md).

Mọi source/config/manifest/report/checkpoint lịch sử và thay đổi cá nhân được
bảo toàn. `verification.json` ghi command, hashes và số file tracked đã đối
chiếu trước/sau; source P1 vẫn kiểm được identity cũ.
