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

## Kết quả thực tế ngày 15/11/2026 (Asia/Bangkok)

Code inference được commit `5b6b820` trước GPU. Preflight xác minh official weights/API/env, 10 tests CPU và hooks đạt trước inference. Sau khi phát hiện output RGBA có alpha không hoàn toàn 255, commit `f5dd55c` sửa **preview** ghép nền trắng, giữ nguyên PNG và mọi run; 11 tests CPU đạt. Không chạy lại GPU vì thay đổi này chỉ thuộc viewer. Chromium kiểm tra HTML tổng hợp: 6 case, 18 ảnh giải mã được, 24 ô chấm đều trống, không JS error. Test xuất JSON dùng fixture tổng hợp riêng trong `/tmp`, không tạo nhãn tay thật.

RTX 5090, Torch CUDA 12.8, BF16, model CPU offload, seed thực tế 42, đủ 40 bước mỗi ảnh. **6/6 inference hoàn tất kỹ thuật, 0 OOM**. Không thử chế độ tải toàn bộ model lên GPU. Smoke `smoke-v1` và lượt thêm 5 `pilot-v1` đều có receipt đã xác minh.

| Task | Thời gian inference (giây) | Output PNG | Đỉnh GPU đo bởi nvidia-smi (MiB) |
|---|---:|---|---:|
| Xóa ghế vàng — smoke | 47.97 | 1376×768 RGBA | 19268 |
| Thay ghế xanh bằng chậu cây | 45.55 | 1376×768 RGBA | 19268 |
| Xóa xe máy bên trái | 50.62 | 1376×768 RGBA | 19220 |
| Xóa dù đỏ | 26.32 | 1376×768 RGBA | 19220 |
| Thêm chim trên mái | 26.63 | 1184×896 RGBA | 19448 |
| Thêm người trước bảo tàng | 26.35 | 1184×896 RGBA | 19434 |

Tổng thời gian trong runner: smoke 48.76 s, lượt 5 mẫu 176.30 s; không gồm hash preflight và startup trước khi bắt đầu đo. Đỉnh toàn GPU cao nhất 19448 MiB ≈ 18.99 GiB. Có khởi động/cache khác nhau giữa các mẫu; không suy diễn tốc độ trên tập lớn từ số đo này. `torch_dtype` báo deprecated nhưng vẫn hoạt động với API đã pin. Không có lỗi OOM hoặc giảm dtype/steps để chạy qua.

Quan sát dưới đây do **assistant xem ảnh thật**, không phải nhãn PASS/FAIL của người dùng:

| Mẫu | Edit Fulfillment quan sát được | Preservation / lỗi cần lưu ý |
|---|---|---|
| Xóa ghế vàng | Ghế vàng biến mất | **Ghế xanh bên trái cũng bị xóa**, ngoài yêu cầu. IP2P cũ giữ nguyên ghế vàng. |
| Thay ghế xanh | Có chậu cây ở vị trí ghế xanh | Giữ ghế vàng, người ngồi và cửa hàng ở mức bố cục. IP2P cũ biến cả cảnh thành sân vườn. |
| Xóa xe máy trái | Xe bên trái biến mất | Còn cụm ba xe bên phải/cây/mặt tiền; chữ và texture có tái tạo. IP2P cũ sinh cảnh phố khác. |
| Xóa dù đỏ | Không còn dù đỏ | Người, xe máy, xe dừa, ghế tím còn hiện diện; chi tiết dừa/chữ/texture thay đổi. IP2P cũ vẫn có dù và biến đổi nhiều object. |
| Thêm chim | Một chim lớn trên trời | Bố cục mái/tường/cây còn; thấy đốm màu nhỏ trên trời và thay đổi texture mái. Kích thước chim cần người duyệt. IP2P cũ sinh nhiều chim nhỏ. |
| Thêm người | Người đứng giữa lối đi | Bố cục tòa nhà/cây/cỏ còn, nhưng chữ trên biển có biến dạng. IP2P cũ không thấy người được thêm. |

Kết quả quan sát cho thấy Qwen đáng tiếp tục đánh giá hơn IP2P trên những mẫu này, đặc biệt thay ghế, xóa xe và xóa dù. Tuy nhiên, lỗi xóa nhầm ghế xanh cho thấy **chưa thể coi preservation đáng tin cậy**. Không gán PASS tổng hợp, không báo tỉ lệ thành công chất lượng hoặc F1. Các lựa chọn trong HTML và `review-template.json` đều `pending/null` để người dùng chấm.

HTML tổng hợp: `runs/level3/qwen21-v1/comparison-v1/comparison.html`. [Validation JSON](LEVEL3_QWEN21_VALIDATION_V1.json) ghi hashes, telemetry từng mẫu và quan sát chi tiết. Logs/screenshot: `runs/level3/qwen21-v1/validation-v1/`. Cả hai bộ ảnh cũ/mới đều giữ nguyên; không mở rộng quá 5 instruction bổ sung, không chạy scoring SGG mới trong lượt này.
