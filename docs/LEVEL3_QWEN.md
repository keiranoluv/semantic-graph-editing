# Qwen Level 3 — một bộ 200 yêu cầu

Entry point hiện tại là `scripts/run_level3_qwen.sh`: dùng **một manifest 200 yêu cầu**, một output có resume, một giao diện review. Manifest `configs/level3/qwen200-cohort-v1.json` giữ nguyên 100 yêu cầu đầu và 100 yêu cầu bổ sung. Có 100 ảnh nguồn, mỗi ảnh có hai yêu cầu độc lập; cả hai đều chỉnh từ ảnh gốc. Không dùng target cũ làm source cho câu mới.

## Gom kết quả và mở review

Từ thư mục gốc repository:

```bash
bash scripts/run_level3_qwen.sh collect
bash scripts/run_level3_qwen.sh serve
```

Mở **http://localhost:8008/comparison.html**. HTML nằm ở `runs/level3/qwen200/seed42/comparison.html`. `serve` thu thập một lần rồi mở server, `collect` chỉ cập nhật dữ liệu trên CPU. Hai lệnh này không khởi tạo model, CUDA hoặc W&B. Có thể gom trong khi launcher cũ đang chạy: chỉ nhận case đã hoàn tất và xác minh receipt/hash. Cả 200 yêu cầu luôn có trong giao diện, ảnh chưa hoàn tất được hiển thị là chưa có và khóa ô chấm. Không chia giao diện thành lượt cũ/mới.

Khi inference cũ hoàn tất, chạy lại `collect` rồi bấm **Cập nhật** trên trang đang mở. Nút này đọc snapshot mới, không tự chạy collect/inference. Nếu dùng server mới thì `serve` đã bao gồm collect. Dừng server bằng Ctrl+C; đổi cổng bằng `serve --port 8009`. Nếu làm việc từ máy khác, forward đúng cổng của server.

Chỉ có hai nhãn **Đúng yêu cầu** và **Bảo toàn**, giá trị Đạt/Không đạt/Chưa chấm; không tự chấm PASS. Bấm ảnh để mở PNG gốc đầy đủ. Xuất `qwen200-reviewed.json` để lưu nhãn lâu dài. Nhãn gắn với task, instruction và hash source/target nên thêm kết quả không làm mất nhãn đã chấm hoặc áp nhầm sang ảnh khác.

Nút **Nhập nhãn JSON** nhận file 200 yêu cầu mới hoặc các file Qwen-only cũ (`qwen-only-review-v2`) có hash khớp output gốc. LocalStorage cũ tự được nhập khi cùng origin (host và cổng) và xác minh được hash. Khác cổng/trình duyệt phải xuất rồi nhập JSON; không có chia sẻ localStorage tự động giữa cổng 8004 và 8008. Nhãn trên ảnh chưa hoàn tất, nhãn khác mô hình hoặc binding/hash không khớp bị từ chối. Các file nhãn người dùng đã có được giữ nguyên.

## Inference và resume một lệnh

Sau khi tiến trình inference hiện tại kết thúc:

```bash
bash scripts/run_level3_qwen.sh
```

Lệnh nhận kết quả hợp lệ từ `runs/level3/qwen100-v1/seed42-v1` và `runs/level3/qwen200-extension-v1/seed42-v1`, rồi chạy **chỉ yêu cầu còn thiếu** trong bộ 200. Nếu đủ 200 thì kết thúc trên CPU, không tải model hoặc mở W&B. Dùng lại cùng lệnh/cùng `OUTPUT` để resume; không cần tạo thư mục mới. Nếu đã resume launcher cũ sang output khác, thêm đường dẫn đó:

```bash
bash scripts/run_level3_qwen.sh infer --import-run runs/level3/qwen200-extension-v1/seed42-resume-01
```

Cấu hình cố định: official **Qwen/Qwen-Image-2.1**, revision `d26bb61231c349cf6b7896fa83353113880e1ba3`, Diffusers và checkpoint đã pin, **BF16, batch size 1, seed 42, 40 steps**, output resolution 1024, CPU offload và VAE tiling. Dùng môi trường `data/cache/level3-qwen21-v1/venv`, CUDA 12.8/RTX 5090 như các lượt trước. W&B online project `level3-qwen200`, ghi metadata/timing/VRAM, không upload ảnh. GPU lock chung từ chối ngay nếu tiến trình khác đang dùng, không xếp hàng chờ. Chạy tuần tự các yêu cầu thiếu rồi dừng; không có lượt/seed khác tự chạy tiếp, không tự chạy REACT++ hoặc scoring.

Output mặc định **`runs/level3/qwen200/seed42/`**. Mỗi lần chạy có `attempts/attempt-NNNN` chứa config, code hashes, môi trường, pipeline/scheduler, W&B và execution/failure. Mỗi case mới nằm trong `cases/<task_id>/attempt-NNNN`, lưu instruction, PNG nguồn/đích, thời gian, VRAM, số bước thực tế và receipt. Case hoàn tất bất biến; nếu bị ngắt, lần sau tạo attempt riêng, không ghi đè/xóa dấu vết lỗi. `index.json`, `results.json`, `review-data.json` và `snapshot.json` là các snapshot có thể cập nhật của toàn bộ 200 yêu cầu. Mỗi case nhập giữ `case_dir` và receipt/provenance ở nơi tạo ban đầu; không di chuyển hoặc xóa output gốc.

Muốn chạy lại toàn bộ 200 từ đầu, chủ động chọn output riêng và bỏ nhập kết quả:

```bash
OUTPUT=runs/level3/qwen200/full-rerun-01 bash scripts/run_level3_qwen.sh infer --no-import
```

Đây là lệnh chạy lớn do người dùng tự thực hiện. `--no-import` không xóa thành công đã có trong chính output đó; khi chạy lại lệnh vẫn resume. Không huấn luyện/fine-tune. Thành công kỹ thuật không phải đánh giá chất lượng; không tự tạo nhãn hoặc điểm F1.

## Dọn dung lượng và giữ khả năng tái lập

Không tạo thêm bản sao manifest instruction: dùng trực tiếp cohort 200 đã có. Giao diện dùng JSON riêng, cache thumbnail một lần cho mỗi ảnh, và hard link PNG gốc thay vì chép lại hoặc nhúng base64 lặp vào HTML. Hard link dùng chung nội dung trên đĩa; các ảnh native là artifacts bất biến, không mở rồi lưu sửa trực tiếp.

`python -m tools.level3_qwen_unified.cleanup --report runs/level3/qwen200/<tên-báo-cáo-mới>` lập kế hoạch trên CPU. Thêm `--apply` để gộp lưu trữ các PNG trùng SHA-256 trong **hai output đã hoàn tất** Qwen100/REACT100 được allowlist. Công cụ giữ mọi đường dẫn và nội dung, từ chối output có tiến trình ghi, kiểm tra lại receipt trước/sau, ghi journal và số byte thực thu hồi. Không đụng lượt extension đang chạy, checkpoint, dataset, nhãn hay output ngoài danh sách.

Các module/config/launcher lịch sử `qwen21_v1`, `qwen100_v1`, `qwen200_v1` còn là phụ thuộc được pin trong provenance và resume cũ; backend thống nhất cũng tái sử dụng runtime chính thức đã kiểm tra. Vì vậy không xóa hoặc sửa chúng trong đợt chuyển giao này. Các trang lịch sử là artifacts của thực nghiệm cũ, không còn là giao diện chính. Dọn code hoặc manifest bằng cách xóa ngay sẽ phá tái lập và có thể ảnh hưởng job đang chạy. Thay đổi riêng của người dùng, Level 1/2/Task B, notebook cá nhân và dữ liệu đã khóa nằm ngoài phạm vi dọn.

## Kiểm tra CPU

```bash
CUDA_VISIBLE_DEVICES='' data/cache/level3-sgg-pretrained-v2/venv/bin/python -m pytest -q \
  tests/level3/test_qwen_unified.py tests/level3/test_qwen200_v1.py \
  tests/level3/test_qwen100_v1.py tests/level3/test_qwen21_v1.py
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m pytest -q tests/level3/test_qwen_unified_browser.py
CUDA_VISIBLE_DEVICES='' bash scripts/run_level3_qwen.sh preflight
```

Browser test cần Playwright/Chromium, dùng `--disable-gpu`. Test editor là fixture CPU để kiểm tra import/resume/OOM giả lập, không phải smoke GPU hoặc bằng chứng chất lượng Qwen. Chưa chạy GPU bằng launcher thống nhất; các ảnh được nhập là kết quả inference thật từ launcher cũ. Graph/scoring cũ của 100 cặp vẫn được giữ nguyên; gom Qwen không tự mở rộng các kết quả graph này.
