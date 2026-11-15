# Level 3 — Qwen-Image-2.1 pilot v1

Thay editor ở lượt thử mới, giữ nguyên InstructPix2Pix và REACT++ cùng mọi kết quả cũ. Không đổi `src/`, dataset, source/config khóa, Level 1/2 hay Task B. Không train/fine-tune. Người dùng cho phép chạy một smoke GPU thật và, nếu smoke hoàn tất kỹ thuật, thêm tối đa 5 instruction. Không tự mở rộng benchmark.

## Nguồn chính thức và môi trường

- [Checkpoint Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1), revision `d26bb61231c349cf6b7896fa83353113880e1ba3`, license Qwen Research. Dùng đầy đủ weights BF16, không quantize, không thay bằng Qwen-Image-Edit-2511.
- [Repository Qwen chính thức](https://github.com/QwenLM/Qwen-Image-2.1).
- [Diffusers commit được khóa](https://github.com/huggingface/diffusers/tree/43e30028af5120d8b0868cb6975b9fad39cc4fab), class `QwenImage21Pipeline`. Kiểm tra API thực tế có image conditioning, không dùng nhầm pipeline chỉ text-to-image.
- Torch 2.11.0+cu128 / torchvision 0.26.0+cu128, transformers 5.23.0, accelerate 1.15.0. Env riêng `data/cache/level3-qwen21-v1/venv`; không thay env Task 3/IP2P cũ.

Full checkpoint gồm text encoder, transformer, VAE khoảng 33 GB; model CPU offload dùng ngay từ đầu vì RTX 5090 có 32 GB VRAM và còn cần activation. VAE tiling bật. Tất cả component eval/frozen, inference mode. Không đánh đồng “chạy không OOM với offload” với “đã thử tải tất cả lên GPU”.

```bash
python3 -m venv data/cache/level3-qwen21-v1/venv
data/cache/level3-qwen21-v1/venv/bin/pip install -r requirements-level3-qwen21-v1.txt
data/cache/level3-qwen21-v1/venv/bin/python -m tools.level3_qwen21_v1 prepare
CUDA_VISIBLE_DEVICES='' data/cache/level3-qwen21-v1/venv/bin/python -m tools.level3_qwen21_v1 preflight
CUDA_VISIBLE_DEVICES='' data/cache/level3-qwen21-v1/venv/bin/python -m pytest tests/level3/test_qwen21_v1.py -q
```

SHA256, size và URL revision của tất cả file checkpoint ở `configs/level3/qwen21-assets-v1.json`. `prepare` chỉ tải thiếu, không ghi đè file có sẵn; hash mismatch dừng và cần thư mục phiên bản mới. Inference offline, kiểm tra hash trước khi tạo output. Phiên bản Diffusers được kiểm tra từ VCS metadata đã cài. Toàn bộ versions thực tế được lưu trong từng run.

## Giao thức cố định

`configs/level3/qwen21-pilot-v1.json` khóa receipt của IP2P `pilot-five-seed42-v1` và 10 yêu cầu người dùng đã duyệt. Chỉ đọc ảnh nguồn PNG đã dùng cho IP2P; sao chép nguyên bytes vào thư mục mới và xác minh hash. Prompt không dịch/viết lại sau khi xem output.

Qwen: **BF16, seed thực tế 42 cho mỗi instruction, 40 bước, batch 1**, `true_cfg_scale=1.0`, `use_kv_cache=true`. Giữ `output_resolution=1024` theo default API đã pin: pipeline tự resize conditioning và sinh output theo diện tích/aspect ratio, làm tròn bội 32. Không mặc định ép ảnh Qwen xuống 512 vì đó là cấu hình của IP2P. Ghi kích thước output thật và giữ PNG gốc.

IP2P cũ dùng FP16, 30 bước, max side 512; base seed 42 nhưng actual seed theo hash task ID, ví dụ xóa ghế vàng là 1006579066. Qwen dùng literal seed 42 theo yêu cầu. **Cùng input/prompt, khác compute, output resolution và actual image seed**; không gọi đây là benchmark bằng nhau về tất cả yếu tố.

Smoke bắt buộc `sidewalk-b1bb633496439f-t3-1`, ảnh `sidewalk-b1bb633496439f`, prompt:

> Remove the yellow chair in front of the shop. Keep everything else unchanged.

Năm task bổ sung được chọn trước khi sinh Qwen: thay ghế xanh bằng chậu cây (ảnh 1), xóa xe máy trái (ảnh 2), xóa dù đỏ (ảnh 3), thêm chim (ảnh 4), thêm người (ảnh 5). Không chọn lại theo chất lượng đầu ra. Pilot phải xác minh receipt smoke hoàn tất 40 bước, output không rỗng, CUDA, đúng task/config/asset hash. Thành công kỹ thuật **không gán PASS chất lượng**.

## Lệnh GPU và giới hạn thực thi

Hoàn tất kiểm tra CPU và commit code trước GPU. Các lệnh dưới đây được người dùng cho phép thực hiện trong yêu cầu này. Mỗi lệnh một lượt rõ phạm vi, không tự nối smoke sang pilot, không seed queue. Dùng chung khóa inference với SGG để không chạy chồng mô hình. Đây là thử nghiệm editor nhỏ, không train và không bật W&B; provenance/telemetry lưu local.

```bash
# Một ảnh, một instruction, batch 1, seed 42, 40 bước.
bash scripts/run_level3_qwen21_v1.sh smoke

# Chỉ sau smoke thành công: tối đa 5 instruction, tuần tự batch 1.
bash scripts/run_level3_qwen21_v1.sh pilot
```

Outputs lần lượt `runs/level3/qwen21-v1/smoke-v1/` và `pilot-v1/`. Thư mục phải chưa tồn tại. Dùng `OUTPUT=.../smoke-v2` khi cần phiên bản mới và `SMOKE_RUN=.../smoke-v2` cho pilot tương ứng. Không xóa hoặc ghi đè run thất bại. OOM dừng lượt, lưu `failure.json`, partial results, telemetry và HTML; không tự đổi BF16, số bước, seed hay resolution để che lỗi.

Ảnh, prompt, config, source/output hashes, checkpoint revision, code hashes/commit, scheduler, env, thời gian load/inference và VRAM nằm trong run. VRAM gồm PyTorch peak allocated/reserved theo case và mẫu `nvidia-smi` mỗi khoảng 0.25 s cho toàn GPU (có thể bao gồm process khác). Thời gian inference có synchronize CUDA, bao gồm các chuyển CPU/GPU; không dùng thời gian enqueue kernel làm thời gian inference. Mẫu đo toàn GPU có thể bỏ lỡ đỉnh ngắn; giữ cả hai loại telemetry.

## HTML và chấm chất lượng

Mỗi run có `comparison.html` ba cột nguồn/IP2P/Qwen, kèm prompt và cấu hình. Không có nhãn PASS tự sinh. Reviewer phải chấm Edit Fulfillment và Preservation riêng, nhập tên/ghi nhận xét rồi xuất `qwen21-reviewed.json`; chưa chấm giữ `null/pending`. Khi inference lỗi, HTML hiện lỗi, không giả ảnh đích.

Tổng hợp smoke và pilot, chỉ đọc các run đã có receipt hợp lệ:

```bash
data/cache/level3-qwen21-v1/venv/bin/python -m tools.level3_qwen21_v1 compare \
  --runs runs/level3/qwen21-v1/smoke-v1 runs/level3/qwen21-v1/pilot-v1 \
  --output runs/level3/qwen21-v1/comparison-v1
python3 -m http.server 8003 --bind 127.0.0.1 \
  --directory runs/level3/qwen21-v1/comparison-v1
```

Mở http://localhost:8003/comparison.html. HTML nhúng ảnh preview, PNG đầy đủ vẫn nằm trong các run nguồn. Khi dùng máy từ xa cần forward port. Không upload ảnh dataset vào dịch vụ bên ngoài.

Quan sát ảnh thực tế của assistant sẽ được báo riêng, không ghi thành nhãn người dùng hay kết quả PASS. Pilot editor này chưa tự chạy thêm REACT++/CLIP/LPIPS, chưa tạo F1 hoặc Spearman, chưa thay kết quả scoring cũ. Chỉ nối Qwen vào lượt đánh giá Task 3 tiếp theo sau khi xác định chất lượng chỉnh sửa đủ hữu ích.
