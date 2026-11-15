# Sửa runtime ROBIN + SAM cho RTX 5090

Lượt `robin-five-v1` dừng ở SAM với `no kernel image is available for execution on the device`. Kiểm tra máy cho thấy RTX 5090, compute capability 12.0 (`sm_120`), driver 590.48.01. Môi trường v1 dùng Torch 2.5.1+cu124, chỉ có native kernels `sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90`. Đây là lỗi tương thích wheel/GPU; CPU smoke trước đó không phát hiện được. Chưa có kết quả ROBIN cho lượt bị lỗi.

[PyTorch 2.7](https://pytorch.org/blog/pytorch-2-7/) bổ sung hỗ trợ Blackwell và wheel CUDA 12.8. Runtime v2 pin cặp chính thức **torch 2.7.1+cu128 / torchvision 0.22.1+cu128** theo [trang các phiên bản PyTorch](https://pytorch.org/get-started/previous-versions/). Giữ Transformers 4.43.3 và các dependency ROBIN đã kiểm tra. Không sửa môi trường v1 hoặc `myenv`, checkpoint, cohort 5 ảnh, seed, decoding và dữ liệu cũ.

Môi trường mới nằm ở `data/cache/level3-sgg-v1/venv-cu128-v2`. Trong workspace hiện tại assistant đã cài đặt; máy mới cần:

```bash
python3.12 -m venv data/cache/level3-sgg-v1/venv-cu128-v2
data/cache/level3-sgg-v1/venv-cu128-v2/bin/python -m pip install \
  -r requirements-level3-sgg-robin-cu128-v2.txt
CUDA_VISIBLE_DEVICES='' data/cache/level3-sgg-v1/venv-cu128-v2/bin/python \
  -m tools.level3_sgg_compare_v1 preflight
```

Checkpoint vẫn dùng cache đã chuẩn bị bằng `prepare`. Runtime được ghi vào provenance gồm tên profile, phiên bản Torch/torchvision, CUDA build, compiled architectures. Trước khi nạp checkpoint cho `infer --device cuda`, adapter kiểm tra compute capability với native cubins trong wheel và hỗ trợ BF16. Sai kiến trúc sẽ dừng trước khi tạo output/nạp model, kèm tên requirements/launcher cần dùng. Guard không suy đoán PTX fallback hoặc khả năng tương thích của kernel có hậu tố kiến trúc đặc biệt.

Chạy lại **5 ảnh thật** bằng launcher mới:

```bash
bash scripts/run_level3_sgg_robin_cu128_v2.sh
```

SAM ViT-H → giải phóng → ROBIN-3B Stage 2; batch ảnh **1**, batch điểm SAM **32**, seed **42**, tối đa 2048 token mới/ảnh. Output mới: `runs/level3/sgg-compare-v1/robin-five-cu128-v2/`. Không tự chạy lượt tiếp theo, không huấn luyện. Giữ nguyên thư mục `robin-five-v1` bị lỗi để truy vết. Nếu output mới đã tồn tại, dùng biến `OUTPUT` với đường dẫn version mới; không xóa/ghi đè kết quả cũ.

Người dùng đã cho phép assistant chạy GPU test nhỏ. Lệnh kiểm tra riêng, không phải lượt inference đủ 5 ảnh:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  flock -n data/cache/level3-sgg-v1/inference.lock \
  data/cache/level3-sgg-v1/venv-cu128-v2/bin/python \
  -m tools.level3_sgg_compare_v1 gpu-smoke \
  --output runs/level3/sgg-compare-v1/gpu-smoke-cu128-user-v1
```

Phạm vi smoke: một ảnh pilot đầu tiên; kiểm tra CUDA permute/contiguous (vị trí lỗi cũ), BF16 matmul và torchvision NMS; SAM ViT-H chạy forward với **một điểm/batch 1**, giải phóng SAM, rồi ROBIN sinh **8 token** bằng mask fixture toàn ảnh. Seed 42, không hàng đợi, không train. `sam-probe.json`, `smoke.json`, `receipt.json` ghi evidence. Smoke kiểm tra tính thực thi kernel/model; chưa chứng minh graph đúng, đủ token hoặc batch điểm 32 không OOM.

Tests CPU:

```bash
CUDA_VISIBLE_DEVICES='' data/cache/level3-sgg-v1/venv-cu128-v2/bin/python \
  -m pytest -q tests/level3/test_sgg_compare_v1.py
```

37 tests gồm 30 kiểm tra SGG có trước và 7 trường hợp về pair runtime, kiến trúc CUDA, dừng sớm trước checkpoint/output. Không cần GPU cho suite này. Hướng dẫn chuẩn hóa EditGraph, kiểm duyệt, giới hạn Task B và khảo sát PGSG vẫn ở [tài liệu protocol](LEVEL3_SGG_ROBIN_V1.md).
