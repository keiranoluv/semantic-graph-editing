# Kiểm chứng CPU và tình trạng so sánh SGG

Đã triển khai ROBIN-3B Stage 2 + SAM ViT-H trong `tools/level3_sgg_compare_v1`, độc lập với source/pipeline đang khóa. Đã kiểm tra repository/checkpoint chính thức, tải và xác minh SHA-256 trọng số, chuẩn bị cùng 5 ảnh để đối chiếu với RelTR + CLIP. **Chưa chạy lượt ROBIN đầy đủ trên 5 ảnh hoặc đo chất lượng graph.** Lệnh người dùng chạy và các giới hạn nằm trong [hướng dẫn](../../docs/LEVEL3_SGG_ROBIN_V1.md); bằng chứng máy đọc được ở [validation JSON](sgg-compare-v1-validation.json).

| Kiểm tra | Kết quả |
| --- | --- |
| Tests mới trong môi trường ROBIN | 30 passed |
| Ba suite Level 3 cũ trong `myenv` | 57 passed |
| Ruff và cú pháp launcher Bash | Pass |
| Preflight thật | Đúng revision/hash code, checkpoint, 30 ảnh nguồn, cohort 5 và baseline receipt |
| CPU smoke với checkpoint thật | Nạp strict SAM rồi giải phóng; nạp ROBIN, sinh 8 token trên ảnh pilot thật; tất cả tham số frozen; CUDA chưa khởi tạo |
| Sửa tương thích loader | Phục hồi chính xác 36 `gamma` ConvNeXt từ safetensors; không còn missing/unexpected keys chưa xử lý |
| HTML Chromium, GPU disabled | 5 ảnh/10 panel; bật/tắt/từng bbox; desktop và viewport 390px; không lỗi JS |
| Hash dữ liệu/source có trước tác vụ | 1.381 file giữ nguyên, gồm AGENTS.md, notebook cá nhân, locked pilot, baseline cũ |

Smoke cuối nằm ở `runs/level3/sgg-compare-v1/cpu-smoke-v4/`. Ảnh là `sidewalk-b1bb633496439f`, mask fixture toàn ảnh, native text 8 token: `Objects:\nregion1: pavement with red`. Đây là chuỗi **cố ý bị cắt ngắn** để kiểm tra luồng model trên CPU; không phải một graph hợp lệ hoặc minh chứng độ chính xác. Smoke không chạy SAM forward/segmentation trên ảnh. Không chạy GPU, không huấn luyện vision model và chưa đo VRAM/throughput GPU.

Tests mới bao phủ parsing giữ ID và hướng cạnh, thiếu/thừa region, output lỗi/cắt token, thuộc tính đúng đối tượng, confidence chưa có là null, mask/bbox/RGB pad đồng nhất cho ảnh ngang/dọc, lọc vùng giữ provenance, ảnh/cohort đúng hash, output bất biến, chất lượng chưa chạy không bị ghi 0, review gắn hash, HTML escape và pipeline giả lập đủ 5 ảnh với cả ca parse failure. Dữ liệu giả lập trong tests không đi vào báo cáo chất lượng.

Khi thử gộp toàn bộ suite trong môi trường ROBIN, hai tests **pipeline cũ** thiếu `clip`/`scipy`; chúng đã được chạy lại và pass trong `myenv` đúng với môi trường cũ. Không cài chồng hai bộ dependency để ép gộp môi trường. Tests mới yêu cầu môi trường riêng; nếu không có pycocotools thì suite mới được skip với lý do rõ ràng.

Baseline thật trên 5 ảnh có 137 node, 27 attribute, 36 directed edge và 93 node cô lập. Có trang xem ảnh tại `runs/level3/sgg-compare-v1/baseline-five-v1/comparison.html`. Đây là thống kê cấu trúc; không tính thành quality score. Output ROBIN đang là `not_run`; chưa có người chấm 10 cặp ảnh/model nên chưa thể kết luận mô hình nào tốt hơn.

PGSG đã được đối chiếu code/checkpoint chính thức và khóa metadata. Cần tách plumbing category/target của evaluator, dùng cấu hình SGDet và môi trường riêng trước khi có backend ảnh tùy ý. Không có số đo hoặc output PGSG trong vòng này.

Schema `EditGraph` đã được kiểm tra, nhưng nhãn mô tả tự do của ROBIN và attribute projection chưa tương đương representation/vocabulary của Task B. Không chạy hoặc sửa Task B, Level 1/2, `src/`, dataset đã khóa hay notebook cá nhân. Bước tiếp theo là người dùng chạy launcher một lần cho 5 ảnh, xem ảnh/grounding/ngữ nghĩa cạnh và chấm theo năm tiêu chí trước khi quyết định dùng extractor cho metric Level 3.
