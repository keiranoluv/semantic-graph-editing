# Level 3 sau khi dọn — 2026-11-16

Luồng hiện tại gồm Qwen-Image-2.1 cho 200 yêu cầu, REACT++ cho 100 ảnh nguồn + 200 ảnh đích, và bộ chấm Graph Consistency CPU. Giao diện Qwen và SGG đều có một danh sách 200 cặp, hỗ trợ resume theo output hiện tại. Bộ chấm có grounding được duyệt cho 100 câu đầu; chưa tự mở rộng điểm sang 100 câu mới.

## Lệnh hiện hành

```bash
# Ảnh nguồn/đích, hai trường review thủ công; localhost:8008/comparison.html
bash scripts/run_level3_qwen.sh serve

# Graph nguồn/đích của 200 cặp; localhost:8009/comparison.html
bash scripts/run_level3_sgg.sh serve
```

Hai lệnh trên chỉ dựng trang trên CPU và mở server; không chạy mô hình. Chạy từng lệnh trong terminal riêng, Ctrl+C để dừng. Bộ chấm/review đã có vẫn ở `runs/level3/qwen-graph-score-v1/score-v1` và `review-v1`; server review 8007 đang có được giữ nguyên.

Lệnh tái lập inference/resume vẫn là `bash scripts/run_level3_qwen.sh infer` và `bash scripts/run_level3_sgg.sh infer --max-new-images 100`. Qwen dùng checkpoint chính thức, BF16, batch 1, seed 42, 40 steps và CPU offload; REACT++ dùng ONNX chính thức, batch 1, seed 42. Output lần lượt là `runs/level3/qwen200/seed42` và `runs/level3/qwen-sgg/seed42`. Khi output đã đủ, cả hai bỏ qua inference. Không có hàng đợi lượt tiếp theo. Xem [Qwen](LEVEL3_QWEN.md) và [SGG](LEVEL3_SGG.md) để chạy lại từ đầu trong output mới.

## Đã dọn

- 43 file launcher, runner, giao diện và tests chỉ phục vụ pipeline cũ. Tách nguyên logic của 9 helper runtime/preview/receipt/geometry vào `tools/level3_shared`; các workflow hiện tại không còn import runner pilot.
- Hai môi trường ROBIN/SAM cũ và môi trường InstructPix2Pix; trọng số ROBIN, SAM, RelTR, InstructPix2Pix, OWLv2 và RelateAnything đã hết dùng. Giữ các asset lock, cấu hình, requirements và bản mã nguồn upstream nhỏ để truy vết thực nghiệm.
- Giao diện `qwen-review-v2/qwen100-v1` trùng với giao diện 200 cặp, chỉ chứa template nhãn trống. Xóa cả bundle presentation để không để lại receipt bị thiếu file.
- Lượt chấm chưa đóng receipt `qwen-graph-score-v1/run-20261116T021114Z`. Đã đối chiếu đủ 100 kết quả với `score-v1`: giống nhau ngoài thời gian chạy từng mẫu. Lượt này không chứa ảnh hay native graph.

Dung lượng file được giải phóng theo inode/block khoảng **36,02 GiB**. Chênh lệch dung lượng trống toàn filesystem khoảng 36,05 GiB có thể bao gồm hoạt động khác. Không có inference GPU trong đợt dọn này.

## Giữ lại

Giữ toàn bộ ảnh nguồn, ảnh edited, instruction, graph/native prediction, nhãn review và receipts của các kết quả còn dùng; giữ cả kết quả ảnh/graph baseline trước đây để đối chiếu. Môi trường/checkpoint Qwen và REACT++ đang dùng còn nguyên. Cache CLIP/LPIPS được giữ cho baseline Task 3.

Các thư mục `qwen100-v1/seed42-v1`, `qwen200-extension-v1/seed42-v1` và `qwen-react100-v1/seed42-v1` vẫn là nơi lưu artifact được index mới tham chiếu. Tên cũ thể hiện provenance; không còn yêu cầu chạy hai đợt. Không được xóa/đổi tên chúng chỉ vì có output tổng hợp mới.

Một số file dưới `level3_image_graph_v1`, `level3_qwen21_v1`, `level3_qwen_graphs_v1`, `level3_sgg_pretrained_v2` vẫn là phụ thuộc schema/runtime/decoder đã khóa của luồng hiện hành. Giữ nguyên byte những file có hash trong protocol REACT++ để không đổi cache key hoặc vô hiệu hóa 300 prediction. Các hàm đánh giá SGG bằng review tay cũng được giữ để không mất khả năng đánh giá baseline. Manifest/config/nguồn dataset đã khóa không thay đổi.

## Khôi phục và bằng chứng

[Danh sách code đã dọn](LEVEL3_CLEANUP_CODE_V2.json) ghi từng Git blob, SHA-256 và nguồn của helper đã tách. Code cũ còn ở commit `1d2d8761eaa384b195d9ef26649a678c95f14750`, ví dụ:

```bash
git show 1d2d876:tools/level3_qwen21_v1/runner.py
```

Để tái lập pipeline cũ, dùng checkout riêng tại commit lưu trữ, tái tạo môi trường và tải lại trọng số theo asset lock. Không khôi phục đè vào luồng hiện tại hoặc sửa output cũ. Việc dọn cache không xóa source khỏi Git.

[Kiểm chứng](LEVEL3_CLEANUP_VALIDATION_V2.json): 139 tests CPU, 2 tests Chromium; 9 helper có AST giữ nguyên; chấm lại trên CPU cho 100 cặp cho kết quả JSON giống hệt điểm đã lưu. Đối chiếu hash 6.217 file còn giữ gồm dataset, configs, ảnh, graph, labels, provenance và thay đổi riêng của người dùng; không có mất/thay đổi ngoài danh sách cho phép. 200 yêu cầu Qwen và 300 prediction SGG vẫn hợp lệ.

Bằng chứng chi tiết ở `runs/level3/cleanup-v2/`: inventory trước/sau, kế hoạch xóa theo file/inode, journal xóa và báo cáo kiểm chứng. Các run cũ có tài liệu và bằng chứng được giữ làm lịch sử, dù launcher tương ứng đã được nghỉ dùng.

Sau commit code `c281577`, đã chạy lại launcher với `CUDA_VISIBLE_DEVICES=''`: Qwen nhận 200/200 kết quả, SGG nhận 300/300 prediction với ngân sách ảnh mới bằng 0; không tăng số attempt. Preflight bộ chấm nhận đủ 100 cặp. Runtime Qwen vẫn nhận đúng API chính thức của bản Diffusers đã pin. Đối chiếu lại 6.217 file sau khi dựng lại giao diện không phát hiện thay đổi.
