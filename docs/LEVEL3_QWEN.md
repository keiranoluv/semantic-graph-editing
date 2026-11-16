# Qwen Level 3 — một bộ 200 yêu cầu

Hướng dẫn chính: [pipeline hai bước](LEVEL3_PIPELINE.md).

Entry point hiện tại là `scripts/run_level3_qwen.sh`: dùng **một manifest 200 yêu cầu**, một output có resume, một giao diện review. Manifest `configs/level3/qwen200-cohort-v1.json` giữ nguyên 100 yêu cầu đầu và 100 yêu cầu bổ sung. Có 100 ảnh nguồn, mỗi ảnh có hai yêu cầu độc lập; cả hai đều chỉnh từ ảnh gốc. Không dùng target cũ làm source cho câu mới.

## Gom kết quả và mở review

Từ thư mục gốc repository:

```bash
bash scripts/run_level3_qwen.sh collect
bash scripts/run_level3_qwen.sh serve
```

Mở **http://localhost:8008/comparison.html**. HTML nằm ở `runs/level3/qwen200/seed42/comparison.html`. `serve` thu thập một lần rồi mở server, `collect` chỉ cập nhật dữ liệu trên CPU. Hai lệnh này không khởi tạo model, CUDA hoặc W&B. Có thể gom trong khi launcher cũ đang chạy: chỉ nhận case đã hoàn tất và xác minh receipt/hash. Cả 200 yêu cầu luôn có trong giao diện, ảnh chưa hoàn tất được hiển thị là chưa có và khóa ô chấm. Không chia giao diện thành lượt cũ/mới.

Bộ hiện tại đã đủ 200/200 kết quả. Khi có lần chạy mới hoặc resume, chạy lại `collect` rồi bấm **Cập nhật** trên trang đang mở. Nút này đọc snapshot mới, không tự chạy collect/inference. Nếu dùng server mới thì `serve` đã bao gồm collect. Dừng server bằng Ctrl+C; đổi cổng bằng `serve --port 8009`. Nếu làm việc từ máy khác, forward đúng cổng của server.

Chỉ có hai nhãn **Đúng yêu cầu** và **Bảo toàn**, giá trị Đạt/Không đạt/Chưa chấm; không tự chấm PASS. Bấm ảnh để mở PNG gốc đầy đủ. Xuất `qwen200-reviewed.json` để lưu nhãn lâu dài. Nhãn gắn với task, instruction và hash source/target nên thêm kết quả không làm mất nhãn đã chấm hoặc áp nhầm sang ảnh khác.

Nút **Nhập nhãn JSON** nhận file 200 yêu cầu mới hoặc các file Qwen-only cũ (`qwen-only-review-v2`) có hash khớp output gốc. LocalStorage cũ tự được nhập khi cùng origin (host và cổng) và xác minh được hash. Khác cổng/trình duyệt phải xuất rồi nhập JSON; không có chia sẻ localStorage tự động giữa cổng 8004 và 8008. Nhãn trên ảnh chưa hoàn tất, nhãn khác mô hình hoặc binding/hash không khớp bị từ chối. Các file nhãn người dùng đã có được giữ nguyên.

## Inference và resume một lệnh

Lệnh được giữ để tái lập hoặc resume khi cần:

```bash
bash scripts/run_level3_qwen.sh
```

Launcher chỉ resume kết quả đã có trong chính lượt đó và infer các yêu cầu thiếu. Không tự nhập kết quả từ lượt khác. Nếu đủ 200 thì kết thúc trên CPU, không tải model hoặc mở W&B. Output hiện tại vẫn có các tham chiếu hợp lệ đến artifact lịch sử; các tham chiếu và ảnh gốc đó được giữ nguyên.

Cấu hình cố định: official **Qwen/Qwen-Image-2.1**, revision `d26bb61231c349cf6b7896fa83353113880e1ba3`, Diffusers và checkpoint đã pin, **BF16, batch size 1, seed 42, 40 steps**, output resolution 1024, CPU offload và VAE tiling. Dùng môi trường `data/cache/level3-qwen21-v1/venv`, CUDA 12.8/RTX 5090 như các lượt trước. W&B online project `level3-qwen200`, ghi metadata/timing/VRAM, không upload ảnh. GPU lock chung từ chối ngay nếu tiến trình khác đang dùng, không xếp hàng chờ. Chạy tuần tự các yêu cầu thiếu rồi dừng; không có lượt/seed khác tự chạy tiếp, không tự chạy REACT++ hoặc scoring.

Output là **`runs/level3/qwen200/<RUN_ID>/`**, mặc định `RUN_ID=seed42`. Mỗi lần chạy có `attempts/attempt-NNNN` chứa config, code hashes, môi trường, pipeline/scheduler, W&B và execution/failure. Mỗi case mới nằm trong `cases/<task_id>/attempt-NNNN`, lưu instruction, PNG nguồn/đích, thời gian, VRAM, số bước thực tế và receipt. Case hoàn tất bất biến; nếu bị ngắt, lần sau tạo attempt riêng, không ghi đè/xóa dấu vết lỗi. `index.json`, `results.json`, `review-data.json` và `snapshot.json` là các snapshot có thể cập nhật của toàn bộ 200 yêu cầu. Mỗi case nhập giữ `case_dir` và receipt/provenance ở nơi tạo ban đầu; không di chuyển hoặc xóa output gốc.

Muốn chạy lại toàn bộ 200 từ đầu, chủ động chọn output riêng và bỏ nhập kết quả:

```bash
RUN_ID=reproduce01 bash scripts/run_level3_qwen.sh
```

Đây là lệnh chạy lớn do người dùng tự thực hiện. Chạy lại cùng `RUN_ID` sẽ resume. Bước SGG dùng cùng `RUN_ID` tự đọc đúng các ảnh của lượt này; không cần truyền đường dẫn riêng. Không huấn luyện/fine-tune. Thành công kỹ thuật không phải đánh giá chất lượng; không tự tạo nhãn hoặc điểm F1.

## Dọn dung lượng và giữ khả năng tái lập

Không tạo thêm bản sao manifest instruction: dùng trực tiếp cohort 200 đã có. Giao diện dùng JSON riêng, cache thumbnail một lần cho mỗi ảnh, và hard link PNG gốc thay vì chép lại hoặc nhúng base64 lặp vào HTML. Hard link dùng chung nội dung trên đĩa; các ảnh native là artifacts bất biến, không mở rồi lưu sửa trực tiếp.

`python -m tools.level3_qwen_unified.cleanup --report runs/level3/qwen200/<tên-báo-cáo-mới>` lập kế hoạch trên CPU. Thêm `--apply` để gộp lưu trữ các PNG trùng SHA-256 trong **hai output đã hoàn tất** Qwen100/REACT100 được allowlist. Công cụ giữ mọi đường dẫn và nội dung, từ chối output có tiến trình ghi, kiểm tra lại receipt trước/sau, ghi journal và số byte thực thu hồi. Không đụng lượt extension đang chạy, checkpoint, dataset, nhãn hay output ngoài danh sách.

Sau khi xác minh hai lượt cũ hoàn tất, đã bỏ 12 file của luồng hai giai đoạn: hai launcher, core/runner/entry point Qwen100, package extension Qwen200 và tests riêng của chúng. Runtime hiện tại không cần các file này; infer/resume/review cho 200 yêu cầu đều dùng entry point chung. Trong lần dọn tiếp theo, helper preview/receipt/runtime đã được tách nguyên logic vào `tools/level3_shared`, bỏ package Qwen100 và runner pilot. Các core đã khóa và requirements cũ còn dùng chung; xem [trạng thái sau dọn](LEVEL3_CLEANUP.md). `requirements-level3-qwen100-v1.txt` là bộ dependencies Qwen + W&B để tái tạo môi trường hiện tại.

Source của 12 file đã dọn vẫn có nguyên byte trong Git tại commit `6c52315debb8730ad1f997632e1a341baf80d605`; danh sách, SHA-256 và Git blob ở [LEVEL3_QWEN_RETIREMENT_V1.json](LEVEL3_QWEN_RETIREMENT_V1.json). Có thể đọc lại một file bằng `git show 6c52315:tools/level3_qwen100_v1/runner.py`. Đây là lịch sử của thực nghiệm, còn lệnh tái lập hiện hành là launcher 200 ở trên. Các tài liệu Qwen100/extension cũ được đánh dấu là hướng dẫn lịch sử.

Giữ nguyên cả 100 ảnh nguồn, 200 ảnh đích, instruction và mapping task–source–target, các manifest/config đã khóa, receipt, nhãn review, graph/scoring và outputs lịch sử. Các manifest 100 cũ còn dùng để đối chiếu hash/provenance và bộ chấm graph cũ, nên không phải dữ liệu thừa. Không đổi Level 1/2/Task B, notebook cá nhân hoặc các file riêng của người dùng.

## Kiểm tra CPU

```bash
CUDA_VISIBLE_DEVICES='' data/cache/level3-sgg-pretrained-v2/venv/bin/python -m pytest -q \
  tests/level3/test_qwen_unified.py tests/level3/test_qwen_artifacts_v1.py \
  tests/level3/test_qwen21_v1.py tests/level3/test_qwen_graphs_v1.py tests/level3/test_qwen_scoring_v1.py
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m pytest -q tests/level3/test_qwen_unified_browser.py
CUDA_VISIBLE_DEVICES='' bash scripts/run_level3_qwen.sh preflight
```

Browser test cần Playwright/Chromium, dùng `--disable-gpu`. Test editor là fixture CPU để kiểm tra import/resume/OOM giả lập, không phải smoke GPU hoặc bằng chứng chất lượng Qwen. Chưa chạy GPU bằng launcher thống nhất; các ảnh được nhập là kết quả inference thật từ launcher cũ. Graph/scoring cũ của 100 cặp vẫn được giữ nguyên; gom Qwen không tự mở rộng các kết quả graph này.

## Kết quả kiểm tra ngày 2026-11-16

Code `f2fed1a`: 63 test CPU và 1 test Chromium đều qua. Preflight xác minh đủ 200 yêu cầu. Snapshot thu thập khi người dùng vẫn chạy inference có 158/200 kết quả hoàn tất, 42 pending. Kiểm tra trình duyệt với ảnh thật giải mã đủ 358 lượt preview (200 source + 158 target), không lỗi JavaScript, đúng hai trường chấm và không tạo nhãn. PNG đích của giao diện là hard link tới output thật. Kết quả này là trạng thái tại thời điểm thu thập, không phải kết luận lượt GPU đã xong.

Dọn 84 bản PNG trùng nội dung, thu hồi **485.400.576 byte (462,914 MiB)**. Không mất đường dẫn hoặc thay đổi byte ảnh, receipt của Qwen100 và REACT100 vẫn khớp và xác minh toàn bộ bundle thành công. Hash source code của tiến trình extension đang chạy không đổi. Không khởi chạy GPU mới. Chi tiết và đường dẫn bằng chứng: [LEVEL3_QWEN_UNIFIED_VALIDATION_V1.json](LEVEL3_QWEN_UNIFIED_VALIDATION_V1.json).

## Hoàn tất 200 yêu cầu và dọn hai giai đoạn

Sau khi người dùng báo lượt mới xong, xác minh cả hai execution đều 100/100, `error=null`, đủ 40 steps và không còn tiến trình inference. Trang chung đã cập nhật **200/200**, gồm 100 source và 200 target; giữ hai binding group để nhập nhãn từ các lượt lịch sử. Chromium giải mã đủ 400 lượt preview và 200 link ảnh đích đều trỏ tới native output thật, không tạo nhãn chất lượng.

Dọn 12 file (55.442 byte source/test/launcher) và 10 cache bytecode; không còn import tới core/runner hai giai đoạn. 67 test CPU của Qwen, graph và scoring cùng 1 test Chromium đều qua. Lệnh `CUDA_VISIBLE_DEVICES='' bash scripts/run_level3_qwen.sh infer` nhận đủ 200 kết quả và thoát thành công, không tạo attempt hoặc chạy GPU. Các bundle Qwen100, extension100, REACT100 và scoring100 vẫn xác minh; 47 file metadata/source/nhãn hoặc thay đổi riêng được đối chiếu không đổi. Source lịch sử khớp đúng hash trong provenance khi đọc từ Git. Hồ sơ đầy đủ: [LEVEL3_QWEN_RETIREMENT_V1.json](LEVEL3_QWEN_RETIREMENT_V1.json); bằng chứng ảnh thật và kiểm tra tại `runs/level3/qwen200/retirement-v1/`.
