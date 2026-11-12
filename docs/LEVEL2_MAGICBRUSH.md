# MagicBrush: chuẩn bị nhãn và kiểm tay Level 2

Mốc này hoàn tất phần chuẩn bị dữ liệu, pipeline và viewer. **Chưa chạy VLM,
chưa có cặp graph-edit được xác nhận và chưa có human review hoàn thành.**
602 cặp ảnh thô không đồng nghĩa đã đạt yêu cầu ≥300 cặp biến đổi đồ thị.

## 1. Dữ liệu thật và split đã khóa

- Nguồn: [MagicBrush chính thức](https://github.com/OSU-NLP-Group/MagicBrush),
  [dataset card tại revision đã khóa](https://huggingface.co/datasets/osunlp/MagicBrush/blob/6c254560879df7851233ea0fd2ae2a62764d0ef2/README.md).
- Chỉ tải bốn parquet **official train** đầu tiên theo tên. Không tải/đọc
  official dev/test hay caption JSON của test để quyết định nhãn.
- Chọn nhóm `img_id` theo SHA256 với seed 42, giữ mọi turn có trong cohort
  đã tải, dừng khi đạt ít nhất 600 cặp: kết quả **602 cặp, 912 ảnh riêng biệt**.
  Đây là subset của bốn shard, không phải mẫu ngẫu nhiên đại diện toàn dataset;
  không khẳng định session đầy đủ ngoài bốn shard đó.
- Gộp nhóm có chung `img_id` hoặc RGB pixel hash trước khi chia 70/15/15:
  **415 train / 97 val / 90 test**, 310 nhóm nguồn. Không có RGB pixel hash
  trùng giữa các split. Val/test ở đây là **split nội bộ từ official train**,
  không phải benchmark official MagicBrush.
- Chọn cố định **20 cặp train trước khi sinh graph**, theo thứ tự SHA256 trên
  toàn bộ ID train. Không lựa mẫu sau khi thấy chất lượng nhãn.
- Ảnh giữ nguyên encoded bytes từ parquet; mỗi ảnh có raw SHA256, RGB pixel
  SHA256, kích thước, shard và row provenance. Asset chỉ nằm dưới `data/`.
- Card khai báo CC BY 4.0; giữ card gốc trong raw cache và ghi nguồn khi dùng.

Các bằng chứng nhỏ được commit tại
[reports/level2/magicbrush-v1](../reports/level2/magicbrush-v1/README.md).
Manifest đầy đủ và ảnh nằm ở `data/processed/level2/magicbrush-v1/`.

Tái tạo bằng CPU, tại repo root:

```bash
myenv/bin/python -m pip install -r requirements-magicbrush.txt
myenv/bin/python -m tools.prepare_magicbrush --stage all
myenv/bin/python -m tools.magicbrush_graphs --output data/processed/level2/magicbrush-v1/edits-pending-v1.json
myenv/bin/python -m tools.review_magicbrush --stage build
myenv/bin/python -m tools.review_magicbrush --stage serve
```

Mở `http://127.0.0.1:8766`. Viewer hiện có ảnh nguồn/đích thật và câu yêu cầu;
khi chưa có graph, các ô chấm graph bị khóa. Không có hình/graph minh họa
được giả làm annotation. Dừng server bằng Ctrl+C.

## 2. Pipeline graph → alignment → edit labels

Mô hình đề xuất là [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct),
revision `66285546d2b821cf421d4f5eb2576359d3770cd3`, license `qwen-research`.
Đây là **inference với model đóng băng**, không fine-tune hay training.
Config ở [magicbrush-graphs-qwen25vl3b-v1.json](../configs/level2/magicbrush-graphs-qwen25vl3b-v1.json).

Mỗi lần model chỉ thấy **một ảnh + prompt cố định + kích thước ảnh**. Không
thấy câu chỉnh sửa, ảnh còn lại, hay vai trò source/target. Hai graph được
sinh độc lập trước khi căn chỉnh. Câu yêu cầu chỉ xuất hiện trong viewer và
bộ lọc kiểu chỉnh sửa ngoài schema, không đi vào model hoặc matcher.

[Qwen2.5-VL dùng tọa độ pixel tuyệt đối theo frame đầu vào](https://arxiv.org/html/2502.13923v1).
CPU resize mỗi ảnh theo `smart_resize`, bội số 28, pixel budget
100352–401408; processor không resize lần hai. Model xuất box pixel; parser
chuyển về `xyxy_1000` theo đúng frame, không đoán hệ tọa độ hoặc clip box lỗi.
Lưu raw response, frame size, model/config/source hash và lỗi parse. Bbox là
**dự đoán VLM, không phải object GT của MagicBrush**. API theo
[Transformers 4.51.3](https://huggingface.co/docs/transformers/v4.51.3/en/model_doc/qwen2_5_vl).

Graph gồm node `id/label/attributes/bbox` và edge `subject/predicate/object`.
Parser kiểm schema, hữu hạn, box diện tích dương, endpoint hợp lệ. Chuẩn hóa
case/khoảng trắng và deduplicate bộ ba cạnh; graph có thể bỏ sót chi tiết.

Matcher CPU dùng Hungarian với dummy unmatched; chi phí gồm IoU (0.6), lệch
nhãn (0.25), RGB histogram (0.15), khoảng cách tâm (0.1), ngưỡng 0.62.
Match có margin <0.08 và unmatched cùng nhãn được gắn cờ mơ hồ. Đây là heuristic
cho cặp chỉnh sửa ít đổi hình học, không bảo đảm correspondence chính xác.
Giữ ID source khi match, cấp ID mới cho node thêm. Chỉ giữ ID cạnh để
`replace_edge` khi cùng endpoint có correspondence một-một rõ ràng; trường
hợp nhiều cạnh dùng delete/add.

Diff dùng implementation sáu loại thao tác đã có ở `src/editing/operations.py`.
Mỗi nhãn phải áp dụng được và tái tạo chính xác target graph. Điều này kiểm
tính nhất quán cấu trúc, **không chứng minh graph đúng với ảnh/câu yêu cầu**.
No-op, >12 thao tác, alignment mơ hồ và một số yêu cầu style toàn ảnh được
thống kê riêng, chưa đủ điều kiện làm candidate. Regex style chỉ là bộ lọc
ban đầu; người kiểm vẫn cần phát hiện các yêu cầu ngoài schema bị bỏ sót.

Output giữ `human_review_status=pending`; `eligible_for_training` chỉ biểu thị
candidate qua bộ lọc tự động. `ready_for_main_training` vẫn false. Không tự
biến candidate thành nhãn đã được người duyệt chấp thuận.

## 3. Lệnh GPU bàn giao cho người dùng: pilot 20 cặp

**Chỉ người dùng chạy sau khi code đã commit và job GQA ba seed kết thúc.**
Assistant không khởi chạy lệnh này. Không tự nối tiếp full extraction/training.
Launcher dùng cùng khóa thiết bị với job GQA; nếu khóa đang bận, nó từ chối chạy.

Kiểm phạm vi bằng CPU:

```bash
myenv/bin/python -m tools.infer_magicbrush_graphs --review-only --plan
```

Lệnh GPU pilot:

```bash
mkdir -p logs
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 myenv/bin/python -u -m tools.infer_magicbrush_graphs --review-only --wandb-mode online > logs/magicbrush-graphs-pilot-v1.log 2>&1
```

Phạm vi: **40 ảnh của 20 cặp train**, model Qwen2.5-VL-3B đóng băng, bf16,
batch size **1**, seed **42**, greedy decode, tối đa 1536 token/ảnh; W&B
`keiranoluv-hcmus/semantic-graph-editing`, run name
`magicbrush-graphs-qwen25vl3b-bs1`. Batch 1 là cấu hình pilot để kiểm nhãn;
không phải benchmark throughput. Output:
`data/processed/level2/magicbrush-v1/graphs-qwen25vl3b-bs1/`.
Lần đầu cần tải weights của revision đã khóa.

Sau khi inference hoàn tất, chạy CPU:

```bash
myenv/bin/python -m tools.magicbrush_graphs --output data/processed/level2/magicbrush-v1/edits-pilot-v1.json
myenv/bin/python -m tools.review_magicbrush --stage build --edits data/processed/level2/magicbrush-v1/edits-pilot-v1.json
myenv/bin/python -m tools.review_magicbrush --stage serve
```

Dừng server cũ trước khi build/serve bản mới. Giao diện lưu CSV cùng packet hash;
đọc [hướng dẫn review](../reviews/magicbrush-human-v1/README.md). 20 cặp vẫn giữ
nguyên, kể cả graph lỗi/thiếu; không thay bằng mẫu dễ hơn. Graph lỗi vẫn pending,
được báo là thiếu bằng chứng, không được tính như review hợp lệ.

Cache/output từ chối ghi đè khác nội dung và resume bỏ qua prediction đã có,
kể cả `ERROR`. Nếu cần retry/sửa prompt, model, batch hoặc policy, tạo config,
cache và output version mới; không sửa/xóa kết quả cũ. Thay batch có thể đổi
kết quả số học nên cũng phải version. Không dùng val/test để chọn prompt/policy.

## Sau pilot nên làm gì?

1. Người trong nhóm kiểm đủ 20 cặp, chấm năm trục và lưu CSV. Báo tỷ lệ đúng,
   số sai/chưa đủ căn cứ và phân tích lỗi; chưa có nhiều reviewer thì không
   tuyên bố inter-rater agreement.
2. Sửa policy trên train nếu chất lượng chưa đạt, khóa version rồi bàn giao
   lệnh inference toàn cohort (**912 ảnh**, cache cùng identity sẽ tái sử dụng
   ảnh đã có). Pilot hiện không tự chạy bước này.
3. Kiểm lại số cặp graph-edit dùng được **≥300 sau lọc** và chia nhóm. Nếu
   thiếu, mở cohort train version mới, giữ các split cũ và audit leakage.
4. Khi bộ nhãn đủ chất lượng, triển khai baseline dự đoán thao tác từ
   **graph nguồn + embedding câu đóng băng**; target graph chỉ làm nhãn.
   Chốt canonical ID/matching cho node/cạnh mới, protocol và ba seed trước
   khi train; chấm P/R/F1 từng loại, exact match và transaction validity.
5. Khi ba seed GQA+KG xong, tổng hợp mean/sample SD và long-tail so với GAT
   cùng protocol. Chỉ đọc kết quả hiện có; mọi lần cần GPU tiếp theo vẫn bàn
   giao lệnh để người dùng chạy.

## Kiểm chứng và bảo toàn thực nghiệm cũ

```bash
myenv/bin/python -m pytest -q tests/editing/test_magicbrush_pipeline.py tests/editing/test_magicbrush_review_browser.py tests/editing/test_operations.py
```

35 tests CPU đã PASS, gồm fixture graph rõ ràng là synthetic trong thư mục
tạm, integration extraction-cache → diff → review, CSV Unicode/notes và browser
Chromium `--disable-gpu`. Browser QA riêng cũng tải đủ 40 ảnh thật của viewer
20 cặp, desktop/mobile, không lỗi JavaScript. Processor smoke test batch 2
chỉ dùng ảnh synthetic trên CPU, không tải model weights/khởi tạo CUDA.
Tests browser cần Playwright + Chromium; pytest báo skip nếu thiếu, không
được diễn giải skip thành đã kiểm browser.

Code mới nằm ngoài `src/`, không thay đổi mappings/KG/checkpoint/report lịch
sử. Không đưa notebook cá nhân vào commit. Xem audit hash tại
[verification.json](../reports/level2/magicbrush-v1/verification.json).
