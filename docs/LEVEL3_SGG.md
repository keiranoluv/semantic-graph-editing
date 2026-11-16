# REACT++ — 200 yêu cầu, 300 ảnh riêng biệt

Hướng dẫn chính: [pipeline hai bước](LEVEL3_PIPELINE.md).

Đầu vào là kết quả Qwen của cùng `RUN_ID`: 100 ảnh nguồn và 200 ảnh edited. Bước SGG độc lập với bước tạo ảnh, chỉ bắt đầu khi đủ 200 kết quả Qwen. Hiện lượt `seed42` đã có đủ 300 prediction. Một lượt mới không tự nhập graph lịch sử; chạy lại cùng lượt sẽ resume phần còn thiếu.

Một cache dự đoán theo hash ảnh/cấu hình được dùng cho tất cả yêu cầu. Mỗi yêu cầu có hai bản EditGraph được gắn đúng `pair_task_id`, `image_role` và image/candidate ID; source của hai yêu cầu cùng ảnh dùng chung nội dung dự đoán. Native NPZ, decoded output, confidence và graph gốc giữ nguyên, provenance chỉ rõ nguồn tái sử dụng. Không suy ra correspondence giữa node nguồn/đích từ việc có cùng ID.

## Một launcher cho inference, resume và demo

Từ root repo, kiểm tra trên CPU trước:

```bash
CUDA_VISIBLE_DEVICES='' bash scripts/run_level3_sgg.sh preflight
```

Sau khi code đã commit, lệnh chạy SGG cho lượt hiện tại:

```bash
bash scripts/run_level3_sgg.sh
```

Mô hình **official REACT++ PSG YOLO12m ONNX**, đúng checkpoint/official preprocessing/decoder đang dùng ở run lịch sử. Batch **1**, seed **42**, box threshold **0.4**, relation threshold **0.05** (lọc theo triplet score đúng decoder hiện tại). Môi trường `data/cache/level3-sgg-pretrained-v2/venv`, CUDA **12.8**, RTX 5090; không huấn luyện. W&B online project `level3-qwen-sgg200`, theo dõi metadata, số ảnh, parse success và timing, không upload ảnh. Dùng GPU lock chung và từ chối nếu GPU job khác đang giữ lock, không xếp hàng chờ.

Output là **`runs/level3/qwen-sgg/<RUN_ID>/`**, mặc định `RUN_ID=seed42`. Launcher đọc ảnh từ `runs/level3/qwen200/<RUN_ID>/`; hai thư mục luôn cùng tên lượt. Mỗi lệnh xử lý tối đa 300 ảnh còn thiếu rồi dừng, không gọi Qwen hoặc tự chạy graph scoring. Có thể dùng `--max-new-images` để đặt giới hạn thấp hơn khi kiểm tra.

Để tái lập, chạy riêng hai bước với cùng một tên mới:

```bash
RUN_ID=reproduce01 bash scripts/run_level3_qwen.sh
RUN_ID=reproduce01 bash scripts/run_level3_sgg.sh
```

Không ghi đè output cũ. Lặp lại lệnh tương ứng để resume. Bước SGG kiểm tra model/protocol Qwen, đủ 200 yêu cầu, hash và receipt từng ảnh. Hash index được lưu trong run SGG; nếu input bị thay đổi, resume bị từ chối để tránh trộn graph của hai bộ ảnh. Không cần tạo manifest hoặc config riêng cho từng lượt.

## Giao diện graph chung

```bash
bash scripts/run_level3_sgg.sh serve
```

Mở **http://localhost:8009/comparison.html**. Một danh sách 200 yêu cầu, hai ảnh cạnh nhau, bật/tắt bounding box, graph có hướng, nhãn object/relation và confidence, liên kết ảnh native/EditGraph JSON/native NPZ. Không chia lượt cũ/mới. Giao diện có cache thumbnail và hard link native assets nên không chép lại PNG/NPZ. Ảnh và native arrays được xem là bất biến; không sửa trực tiếp file hard link.

`serve` thu thập và dựng trang trên CPU trước khi mở server. `collect` cập nhật trang mà không chạy GPU; bấm **Cập nhật** để đọc snapshot mới. Server bind localhost, có thể forward cổng; đổi bằng `serve --port 8010`, dừng bằng Ctrl+C. Giao diện ảnh Qwen để gán hai nhãn tay vẫn ở launcher `run_level3_qwen.sh serve` (8008).

## Output, provenance và giới hạn

- `index.json`: 300 ảnh, cache key, mọi task/role sử dụng ảnh đó, receipt và hashes của dự đoán.
- `results.json`, `pairs/<task_id>/`: 200 cặp nguồn/đích; mỗi cặp hoàn tất có receipt riêng. Đường dẫn `graph_path` trỏ đến EditGraph đúng task/role, source graph giữ nguyên object/edge/confidence của dự đoán gốc.
- `predictions/<native_sha256>/attempt-NNNN/`: các dự đoán mới, PNG chuẩn hóa nền trắng, native NPZ, decoded JSON, EditGraph và receipt. Thành công bất biến; lỗi không được tính là graph hoàn tất.
- `attempts/attempt-NNNN/`: config/code hashes, checkpoint metadata, môi trường, CUDA providers/device, W&B, timing, VRAM và profile ONNX chứng minh có CUDA kernels. ORT không được tự fallback toàn bộ sang CPU.
- `comparison.html`, `review-data.json`, `snapshot.json`: một demo cho 200 yêu cầu và hashes tương ứng.

REACT++ hiện không có attribute head; 10 yêu cầu đổi màu vẫn không thể đánh giá Edit Fulfillment từ attribute màu của graph này. Parse success chỉ là thành công kỹ thuật; không tự tạo precision/recall/F1, grounding/hallucination hoặc điểm consistency. Chất lượng cần graph GT hoặc review ảnh. Graph/scoring lịch sử cho 100 cặp được giữ nguyên; lượt này chỉ sinh SGG, chưa chạy chấm lại trên 200 cặp.

Config đã khóa `configs/level3/qwen-sgg-unified-v1.json` vẫn bảo vệ cohort/model/backend và lượt `seed42` cũ. Với tên lượt mới, runner dùng cùng cohort/model, tự lấy đường dẫn và hash index Qwen tương ứng, lưu cấu hình thực tế mang schema `qwen-sgg-unified-v2` vào run output. Không sửa config đã khóa, không thêm file manifest/config theo từng lượt.

## Kiểm tra CPU

```bash
CUDA_VISIBLE_DEVICES='' data/cache/level3-sgg-pretrained-v2/venv/bin/python -m pytest -q \
  tests/level3/test_qwen_sgg_unified.py tests/level3/test_qwen_graphs_v1.py tests/level3/test_qwen_unified.py
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m pytest -q tests/level3/test_qwen_sgg_browser.py
```

Tests dùng native arrays/graph tổng hợp trên CPU, không phải kết quả inference thật. Kiểm tra tái sử dụng graph nguồn, rebinding đúng ID, rejection khi hash/cấu hình thay đổi, bảo toàn relations/confidence, budget trước GPU, resume sau OOM giả lập và phân biệt graph rỗng với parse error.

## Kết quả thực tế — 2026-11-16

Đã commit implementation **`34bd343`** sau 40 test CPU và 1 test Chromium, rồi chạy GPU theo yêu cầu trực tiếp của người dùng. Chỉ chạy **100 target mới**, tái sử dụng 200 dự đoán lịch sử. **100/100** target mới parse thành công, đủ **300/300** dự đoán ảnh và **200/200** cặp graph, không lỗi/OOM. Thời gian ghi nhận gồm load/inference và ghi evidence: **28,999 giây** (không gồm bước dựng lại toàn bộ HTML sau inference). Peak tổng GPU 0 đo bằng nvidia-smi: **1.149 MiB / 32.607 MiB**, 100 mẫu đo. Profile ONNX có **116.200 CUDA kernel events** và 35.400 CPU kernel events; CPU events là các phần được ORT phân công, session CUDA được xác minh và không fallback toàn bộ sang CPU.

W&B đã đồng bộ metadata/metrics, không upload ảnh: [seed42-attempt-0001](https://wandb.ai/keiranoluv-hcmus/level3-qwen-sgg200/runs/vpse4rgm). Native arrays của 300 dự đoán, 200 pair receipts và 400 graph records đã được đối chiếu; nội dung node/edge/confidence của phần tái sử dụng không đổi. 47 file metadata/source/nhãn hoặc thay đổi riêng được đối chiếu không đổi. Test resume với GPU ẩn nhận cache 300/300, không tạo attempt mới. Kiểm tra ảnh thật trên Chromium: đủ 200 cặp, 400 preview, số box/cạnh trong DOM khớp dữ liệu, không lỗi JavaScript.

ORT in ba cảnh báo `ScatterND` về điều kiện chỉ số không trùng khi `reduction='none'`; điều kiện này chưa được kiểm chứng riêng, không được diễn giải thành chứng nhận độ chính xác/determinism. Trong automation, server HTTP tạm ghi một `ConnectionResetError` khi client chuyển/đóng request; các kiểm tra hiển thị đều qua. Các cảnh báo và phạm vi kiểm tra được ghi trong [LEVEL3_SGG_VALIDATION_V1.json](LEVEL3_SGG_VALIDATION_V1.json), bằng chứng local ở `runs/level3/qwen-sgg/validation-v1/`.

Các con số trên là hoàn tất kỹ thuật, không phải precision/recall/F1 hay chứng nhận chất lượng graph. Bộ chấm Graph Consistency cũ vẫn ở phạm vi 100 cặp; chưa tự chạy bộ chấm cho 200 cặp.
