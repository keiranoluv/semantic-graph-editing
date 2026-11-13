# Sửa lỗi tokenizer của grounded pilot: v2

Lượt v1 dừng ở `return_offsets_mapping`: `AutoProcessor(..., use_fast=False)`
chọn cả Python tokenizer chậm, không hỗ trợ offset. Đây là lỗi adapter, chưa
phải kết quả đánh giá chất lượng nhãn. V1 đã ghi 20 source captions; chưa ghi
detection hoặc graph-edit evidence nào.

V2 giữ nguyên recipe, model revision, ngưỡng lọc, 20 cặp train và split. Tạo
processor tường minh với **GroundingDinoImageProcessor chậm + BertTokenizerFast**.
Offset lấy từ chính batch text đã padding, rồi loại khỏi kwargs trước model
forward. Kiểm query/token budget trên CPU trước tải model detector lên GPU.
Tất cả query của 20 cặp đã qua kiểm CPU; Fast/slow token IDs bằng nhau.

Không sửa source/config/manifest/cache/reports v1. V2 kiểm source fingerprint,
manifest, cohort, model/revision và cache identity của caption v1; sao chép
**nội dung caption thật** vào wrapper v2 với hash/file nguồn. Không dùng lại
detection hay nhãn theo tokenizer cũ; không sửa/xóa artifact cha để resume.

## Lệnh chạy

Assistant đã chuẩn bị manifest v2 và kiểm CPU. Muốn tái tạo trên máy này:

```bash
myenv/bin/python -m tools.annotate_magicbrush_grounded_v2 --stage prepare
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 myenv/bin/python -m tools.annotate_magicbrush_grounded_v2 --stage preflight
```

Preflight cần caption v1 đã có; không chạy inference. Kết quả nằm trong
`data/processed/level2/magicbrush-grounded-anchor-v2/cpu-preflight.json`.

**Sau commit code, người dùng chạy lại GPU bằng lệnh v2 này:**

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 myenv/bin/python -u -m tools.annotate_magicbrush_grounded_v2 --stage infer --wandb-mode online > logs/magicbrush-grounded-anchor-pilot-v2.log 2>&1
```

- Chỉ pilot **20 cặp project train**, vẫn từ official train.
- Tái sử dụng **20 caption BLIP base** từ v1; không chạy lại BLIP khi cache này
  đầy đủ. Grounding DINO tiny batch **4** → CLIP ViT-B/32 batch **64**, seed **42**,
  float32; các model chạy tuần tự. Nếu không có caption cha, code có thể chạy
  BLIP frozen để tạo phần thiếu; không tạo caption giả.
- W&B run mới: `magicbrush-grounded-anchor-v2-pilot-bs4-clipbs64-s42`.
  Không ghi tiếp lên run lỗi v1.
- Output mới: `data/processed/level2/magicbrush-grounded-anchor-v2/`.
- Không train, không tự chạy full, không có hàng đợi thí nghiệm tiếp theo.

Sau khi lệnh GPU thành công, chạy CPU:

```bash
myenv/bin/python -m tools.annotate_magicbrush_grounded_v2 --stage compile
myenv/bin/python -m tools.annotate_magicbrush_grounded_v2 --stage review
myenv/bin/python -m tools.annotate_magicbrush_grounded_v2 --stage serve --port 8768
```

Mở `http://127.0.0.1:8768/`, forward port nếu cần. Viewer, review CSV, packet và
audit v2 ở `reviews/magicbrush-grounded-human-v2/`.

Các giới hạn ngữ nghĩa, graph cục bộ chỉ có quan hệ vị trí, 64/415 request thuộc
phạm vi grammar, chưa bảo đảm ≥300 nhãn và việc người kiểm tự nhập quyết định
giữ nguyên như [runbook phương pháp](LEVEL2_MAGICBRUSH_GROUNDED_ANCHOR_V1.md).
Thay các lệnh CLI v1 trong runbook đó bằng CLI v2 ở trên.

## Kiểm thử sửa lỗi

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 myenv/bin/python -m pytest tests/editing -q
```

Regression mới dùng đúng processor factory của launcher, tái hiện slow tokenizer
ném lỗi, kiểm Fast offsets, padding khác độ dài, token IDs không đổi, `-inf`
padding của head DINO, budget quá dài và caption import giữ nguyên file cha/
từ chối sai identity. Probe thực kiểm cả 40 image/query jobs mà không tải model
weights hoặc thực hiện pretrained inference. GPU quality vẫn chờ người dùng chạy.
