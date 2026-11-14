# Task B — tiếp tục trên GPU, v3

Người dùng đã yêu cầu dừng CPU và xác nhận chuyển **cả huấn luyện và đánh giá**
sang GPU. Tiến trình CPU v2 đã nhận SIGINT; giữ nguyên MLP/42 hoàn tất và
MLP/43 dang dở ở `runs/level2/emnlp2020-task-b-v2/`. Không trộn chúng vào trung
bình ba seed GPU. Không sửa source/config/receipt của CPU v2 hoặc MagicBrush.

V3 kế thừa protocol và [giới hạn dữ liệu v2](LEVEL2_EMNLP2020_TASK_B_V2.md).
Dùng cùng 29.868 train / 532 val, CLIP text cache CPU đã khóa, train-only
vocabulary, hidden 96, batch 128, 12 epoch, seeds 42/43/44. MLP rồi GNN,
mỗi model ba seed, tuần tự một lượt mỗi thời điểm. Huấn luyện lại từ đầu trên
GPU; không tiếp tục checkpoint CPU dang dở. Dataset/test denominator không đổi.

Code phiên bản riêng ở `tools/emnlp2020_task_b_gpu_v3/`: chuyển batch/model/target
sang device được chỉ định, sửa mask trong encoder sang đúng device. Orchestration
được version riêng vì kết quả CPU đã bind toàn bộ source hashes v2. Kiểm thử
miniature protocol chạy cả CPU/CUDA: hai model, ba seed, freeze, test barrier,
resume, chống sửa checkpoint; synthetic fixture chỉ kiểm code, không tính metric.

FP32, TF32 tắt, deterministic algorithms bật,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. Chọn checkpoint theo minimum val loss; không
mở test trước khi cả sáu checkpoint/result đã khóa. `prepare/features` tiếp tục
là xử lý CPU; không có image inference. Check CUDA trước khi train, không tự
fallback CPU. Các GPU workload trong lượt này được người dùng cho phép trực tiếp,
sau kiểm CPU và commit code.

```bash
# Kiểm CPU trước khi commit/chạy GPU
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES='' myenv/bin/python -m pytest -q \
  tests/editing/test_emnlp2020_task_b_gpu_v3.py

# Kiểm CUDA sau khi commit code
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 myenv/bin/python -m pytest -q \
  tests/editing/test_emnlp2020_task_b_gpu_v3.py

# Tự chạy đủ sáu lượt tuần tự, batch 128, seed 42/43/44, không tự mở test
CUDA_VISIBLE_DEVICES=0 myenv/bin/python -m tools.emnlp2020_task_b_gpu_v3 --stage train

# Chỉ sau khi khóa đủ sáu checkpoint; không train tiếp
CUDA_VISIBLE_DEVICES=0 myenv/bin/python -m tools.emnlp2020_task_b_gpu_v3 --stage test

# Thống kê và báo cáo, không có GPU computation
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m tools.report_emnlp2020_task_b_v2 \
  --config configs/level2/emnlp2020-task-b-gpu-v3.json \
  --output reports/level2/emnlp2020-task-b-gpu-v3
```

Config: `configs/level2/emnlp2020-task-b-gpu-v3.json`.
Output GPU riêng: `runs/level2/emnlp2020-task-b-gpu-v3/`.
Dữ liệu/features dùng lại: `data/processed/level2/emnlp2020-task-b-v2/`.
CPU/CUDA smoke test batch một mẫu/seed, không phải một benchmark real-data.
Stage train tự chuyển qua đủ sáu lượt; stage test không tự chạy lượt khác.

Inference chỉ cần source+instruction, không target:

```bash
CUDA_VISIBLE_DEVICES=0 myenv/bin/python -m tools.emnlp2020_task_b_gpu_v3 \
  --stage predict --model gnn --seed 42 --input /path/to/source-and-instruction.json \
  --prediction-output data/processed/level2/emnlp2020-task-b-v2/gpu-demo-prediction.json
```

Batch inference 1, GPU, không có hàng đợi. Invalid prediction trả target null
và lỗi; không dùng reference target để sửa prediction.

Sau khi hoàn tất, kiểm lại source/data/checkpoint/prediction mà không chạy GPU:

```bash
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m tools.verify_emnlp2020_task_b_gpu_v3
```

`artifact-receipt.json` khóa mọi output hoàn tất bằng SHA256. Runtime/source đã
train giữ nguyên ở commit `bf985cb`; commit báo cáo về sau không đổi các source
hashes đó. Để tái huấn luyện, dùng checkout commit thực nghiệm và output version
mới. `experiment.json` ghi commit tạo run và là file bất biến: gọi lại stage train
trong cùng output từ một HEAD khác sẽ bị từ chối ngay cả khi chỉ đổi báo cáo.
Lệnh verify ở trên là đường kiểm lại run hoàn tất từ commit bàn giao.
