# Task 3 v1 — pilot 5 ảnh với REACT++

Pipeline thử nghiệm: ảnh nguồn + yêu cầu đã duyệt → InstructPix2Pix pretrained → ảnh chỉnh sửa → REACT++ cho cả nguồn/đích → Graph Consistency, CLIPScore và LPIPS → nhãn tay → Spearman. Không huấn luyện mô hình thị giác. Đây là pilot phát triển, chưa phải kết quả hoàn thành Level 3.

## Phạm vi và dữ liệu

Dùng đúng 5 ảnh đã so sánh ROBIN-SAM/REACT++/RelateAnything trong `react-five-v1`. Receipt của nguồn và SHA256 từng ảnh được kiểm tra trước mỗi lượt. ROBIN-SAM và mọi kết quả SGG cũ giữ nguyên. Source, manifest và dataset khóa không bị sửa.

`configs/level3/task3-drafts-v1.json` chứa 10 **bản nháp do assistant viết**, 2 yêu cầu độc lập/ảnh: 5 xóa, 4 thêm, 1 thay thế. Cần người duyệt xác nhận câu yêu cầu, object/ROI và caption đích trước khi chạy chính. Caption mô tả ảnh đích được viết trước khi xem output, tránh điều chỉnh theo kết quả. Bản nháp không phải graph ground truth hay nhãn chất lượng.

Operation ở pilot này là **ý định do người duyệt xác nhận**, chưa phải dự đoán của mô hình Task B. Pilot giúp kiểm tra extractor và scoring trước khi nối mô hình dự đoán thao tác; không báo cáo độ chính xác Task B từ những thao tác này. Thuộc tính không được REACT++ checkpoint hiện tại hỗ trợ: score thuộc tính là `null`, không suy đoán màu từ tên object.

Theo `Project01_Graph_Editing.pdf` trang 3–5, bản Level 3 đầy đủ vẫn cần tối thiểu 100 ảnh có quyền sử dụng, 150–200 yêu cầu, nhãn tay, so sánh Spearman và phân tích tối thiểu 10 bất đồng. Bộ local hiện là provenance pilot; 5 ảnh/10 yêu cầu không đáp ứng các mốc đó. Không công bố lại ảnh từ bộ local này chỉ dựa vào việc có URL nguồn.

## Mô hình, phiên bản và môi trường

- Editor: `timbrooks/instruct-pix2pix`, revision `31519b5cb02a7fd89b906d88731cd4d6a7bbf88d`. Dùng Diffusers 0.35.1, Euler Ancestral, giữ safety checker. Tham khảo [model card chính thức](https://huggingface.co/timbrooks/instruct-pix2pix) và [hướng dẫn inference Diffusers](https://huggingface.co/docs/diffusers/api/pipelines/pix2pix).
- SGG: REACT++ PSG YOLO12m ONNX, kế thừa nguyên checkpoint/config đã khóa của `level3_sgg_pretrained_v2`. Đây là vocabulary PSG cố định, không phải open vocabulary. Lưu raw NPZ, graph EditGraph, confidence, bbox và provenance.
- CLIP: `openai/clip-vit-base-patch32`, revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`. Công thức reference-free theo [CLIPScore](https://github.com/jmhessel/clipscore), adapter dùng HF CLIP/processor; không tuyên bố bitwise tương đương script gốc.
- LPIPS: AlexNet v0.1 theo [PerceptualSimilarity chính thức](https://github.com/richzhang/PerceptualSimilarity), commit và hash của cả trunk/linear weights được khóa.

Tất cả URL revision, kích thước và SHA256 nằm trong `configs/level3/task3-assets-v1.json`. `prepare` chỉ tải tài nguyên; `preflight` kiểm tra đầy đủ hash, nguồn ảnh và source SGG. Không tự tải trong inference. Môi trường riêng không đổi env cũ:

```bash
python3 -m venv data/cache/level3-task3-v1/venv
data/cache/level3-task3-v1/venv/bin/pip install -r requirements-level3-task3-v1.txt
data/cache/level3-task3-v1/venv/bin/python -m tools.level3_task3_v1 prepare
data/cache/level3-task3-v1/venv/bin/python -m tools.level3_task3_v1 preflight
```

Máy hiện tại đã có env và weights. Yêu cầu artifacts nguồn/SGG ở các đường dẫn cũ; không thay thế chúng bằng graph tự tạo. Torch 2.7.1+cu128, torchvision 0.22.1+cu128, ONNX Runtime GPU 1.23.2; launcher CUDA kiểm tra hỗ trợ `sm_120` cho RTX 5090. CPU dùng float32, CUDA dùng float16 cho editor.

## Duyệt yêu cầu trong trình duyệt

Đã tạo `runs/level3/task3-v1/authoring-v1/authoring.html`. Có thể mở trực tiếp file hoặc phục vụ local:

```bash
python3 -m http.server 8001 --bind 127.0.0.1 \
  --directory runs/level3/task3-v1/authoring-v1
```

Mở **http://localhost:8001/authoring.html**. Nếu máy chạy ở xa, forward cổng 8001 về máy cá nhân. Vùng vàng đánh dấu object cần xóa/thay hoặc vùng thêm. Duyệt cả 10 câu, sửa caption/operation nếu cần, nhập tên người duyệt và xuất `tasks-reviewed.json`. Giao diện có nhập lại file đã xuất. Không đổi câu yêu cầu mà giữ operation không tương ứng.

Biên dịch file người dùng đã duyệt, thay đường dẫn ví dụ bằng file thật:

```bash
data/cache/level3-task3-v1/venv/bin/python -m tools.level3_task3_v1 compile \
  --submission /path/to/tasks-reviewed.json
```

Output `runs/level3/task3-v1/tasks-approved-v1/` chứa tasks, bản nộp, provenance, receipt. Kiểm tra đúng 10 task IDs, đúng ảnh/hash, context hiện hành, approved và reviewer. Không nhận bản nháp làm input chạy chính. Các thư mục output luôn tạo mới; nếu đã tồn tại phải dùng phiên bản mới qua `--output`, không xóa/ghi đè.

Tạo trang authoring phiên bản mới nếu cần:

```bash
data/cache/level3-task3-v1/venv/bin/python -m tools.level3_task3_v1 author \
  --output runs/level3/task3-v1/authoring-v2
```

## Kiểm tra CPU và GPU smoke

```bash
CUDA_VISIBLE_DEVICES='' data/cache/level3-task3-v1/venv/bin/python -m pytest \
  tests/level3/test_task3_v1.py tests/level3/test_sgg_pretrained_v2.py -q
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  data/cache/level3-task3-v1/venv/bin/python -m tools.level3_task3_v1 smoke \
  --device cpu --output runs/level3/task3-v1/cpu-smoke-v2
```

Mỗi smoke chạy đúng một yêu cầu đầu trên một ảnh: batch 1, seed gốc 42, **2 bước, cạnh dài 256**. Seed ảnh là 32-bit đầu SHA256 của `42|task_id`, lưu `actual_seed`. Thực thi tuần tự editor → REACT++ nguồn/đích → CLIP → LPIPS, giải phóng từng mô hình. Không W&B, không queue lượt khác, không đo chất lượng. Smoke có `evaluation_eligible=false`; lệnh report từ chối sử dụng.

Lệnh GPU smoke (thực hiện sau CPU checks và commit code):

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  data/cache/level3-task3-v1/venv/bin/python -m tools.level3_task3_v1 smoke \
  --device cuda --output runs/level3/task3-v1/gpu-smoke-v1
```

Nếu output đã tồn tại, chọn `gpu-smoke-v2` thay vì chạy đè. Kết quả thực tế được ghi riêng trong tài liệu validation khi hoàn tất.

## Chạy chính — người dùng tự chạy

Đăng nhập W&B trong env trước lượt chính. Runner bắt buộc online, project `level3-task3-v1`, không upload ảnh local vào W&B. Mỗi lần gọi chỉ chạy **một seed, 5 ảnh nguồn, 10 chỉnh sửa độc lập**, batch 1; tự chạy các stage trong lượt đó, **không tự chạy seed tiếp theo**. Không train/fine-tune.

```bash
data/cache/level3-task3-v1/venv/bin/wandb login
bash scripts/run_level3_task3_v1.sh 42
```

Thông số cố định: 30 bước, max side 512 (kích thước làm tròn bội 8), guidance 7.5, image guidance 1.5. Thứ tự: IP2P tạo 10 ảnh → giải phóng → REACT++ 5 nguồn resize + 10 đích → giải phóng → CLIP → giải phóng → LPIPS. Dùng chung khóa inference với launcher SGG để tránh chạy đồng thời. Seed mỗi ảnh được suy ra ổn định từ seed gốc và task ID, không phụ thuộc thứ tự tác vụ.

Output mặc định `runs/level3/task3-v1/pilot-five-seed42-v1/`. Hai seed dự phòng 123 và 3407 chỉ chạy khi người dùng chủ động gọi riêng. Có thể đổi `TASKS` và `OUTPUT` để dùng phiên bản approved/output mới:

```bash
TASKS=runs/level3/task3-v1/tasks-approved-v2 \
OUTPUT=runs/level3/task3-v1/pilot-five-seed42-v2 \
  bash scripts/run_level3_task3_v1.sh 42
```

Run lưu checkpoint/code hashes, commit, môi trường, cấu hình thực tế, seed, W&B URL, ảnh, raw NPZ, graph/confidence, resize-control, thời gian và receipt. Lỗi fatal lưu `failure.json` với partial results; không tự resume/ghi đè. Safety-filtered/parse-error được giữ trong mẫu số thất bại.

## Graph Consistency v1

Trích lại **nguồn đã resize cùng kích thước ảnh đích**; không so graph full resolution cũ với ảnh editor 512 trực tiếp. ID object gốc chuyển sang nguồn resize bằng matching nhãn và bbox. Lưu node mất sau resize; nếu anchor thao tác mất thì `null` và lý do, không tự chuyển yêu cầu sang object khác.

Matching dùng Hungarian một-một, nhãn chuẩn hóa case/whitespace phải trùng, normalized IoU ≥ 0.30, trọng số `1 + IoU`, có dummy để không ép ghép. Flag ambiguous nếu lựa chọn cạnh tranh có IoU cách dưới 0.05; đây vẫn là điểm heuristic, cần xem bằng chứng khi có cờ. Không chỉ đếm node/cạnh.

- Xóa: anchor không còn match **và** số instance cùng nhãn giảm. Không xem di chuyển object cũ là xóa.
- Thêm: có instance chưa match, đúng nhãn, tâm bbox trong ROI đã duyệt **và** số instance nhãn đó tăng. Chỉ miễn phạt một instance mới; object thừa khác vẫn ảnh hưởng preservation.
- Thay thế: trung bình điều kiện xóa và thêm đúng nhãn trong bbox nguồn.
- Quan hệ: hai endpoint phải còn match, kiểm tra đúng directed triple và trạng thái present/absent. Đảo chiều không đạt. Pilot 10 draft hiện chưa bao phủ relation edit.
- Thuộc tính: unsupported → `null`.

Phần không bị thao tác tác động dùng F1 overlap node và F1 overlap directed edge (`2TP/(2TP+FP+FN)`), rồi lấy trung bình các thành phần xác định. Hai tập cùng rỗng là undefined, không tự cho 1. Loại object/quan hệ bị chỉnh sửa khỏi preservation; mất object khác hoặc phát sinh object/edge không liên quan đều bị phạt.

`GraphConsistency = 0.5 × edit_score + 0.5 × preservation_score`; thiếu một thành phần thì `null`. **F1 overlap này đo bảo toàn giữa hai graph dự đoán, không phải precision/recall/F1 extractor so với graph GT.** Không tạo graph GT và không suy ra độ đúng thị giác từ confidence hay số node/edge. Trường hợp xóa cả cảnh có thể đạt edit nhưng preservation = 0, tổng tối đa 0.5; tổng không phải xác suất đạt chung.

CLIPScore = `2.5 × max(cos(CLIP(image_target), CLIP("A photo depicts " + target_caption)), 0)`, thang 0–2.5. Caption quá dài bị từ chối, không âm thầm truncate. LPIPS dùng ảnh nguồn resize và đích, RGB [-1,1], cùng kích thước; lưu raw distance và `negative_lpips` để hướng cao-tốt khi so tương quan. LPIPS nhỏ có thể chỉ phản ánh ảnh không đổi, không chứng minh đã thực hiện yêu cầu.

## Nhãn tay và báo cáo

Mở `review.html` trong output chính: chấm riêng thực hiện yêu cầu và bảo toàn, ghi lý do và reviewer. Trang này ẩn điểm tự động. Đạt chung = cả hai cùng đạt. Chưa chấm giữ `pending`; không tự điền nhãn. Xuất `outcome-reviewed.json`. `scores.html` riêng hiển thị ảnh/graph hai phía, bbox dạng dữ liệu và bằng chứng tính điểm để xem sau khi chấm.

```bash
data/cache/level3-task3-v1/venv/bin/python -m tools.level3_task3_v1 report \
  --run runs/level3/task3-v1/pilot-five-seed42-v1 \
  --review /path/to/outcome-reviewed.json \
  --output runs/level3/task3-v1/report-seed42-v1
```

Report kiểm tra receipt/hash từng kết quả và đủ case IDs. Spearman của graph, CLIPScore, -LPIPS dùng **cùng tập con** có nhãn và cả ba score hữu hạn; báo n và IDs. Dưới 3 mẫu hoặc nhãn/score hằng số → `rho=null`. Các chỉnh sửa cùng ảnh không độc lập; pilot chỉ mang tính mô tả, không khẳng định ý nghĩa thống kê hoặc thắng baseline.

Danh sách tối đa 10 bất đồng là gợi ý theo chênh lệch rank, trường `analysis` để trống, `ten_disagreements_analyzed=false`. Cần người viết phân tích sau; danh sách không có nghĩa đã phân tích 10 ca. Report hiện theo từng seed, chưa gộp nhiều seed làm một mẫu độc lập.

## Giới hạn cần kiểm tra trên pilot

REACT++ được chọn theo duyệt trực quan; graph ít object chỉ giúp đọc dễ, không chứng minh chính xác hơn. Closed vocabulary, bỏ sót object, nhãn sai và quan hệ không ổn định có thể làm sai điểm. Ghép nhãn/IoU yếu khi ảnh thay đổi bố cục mạnh hoặc nhiều instance giống nhau; cờ ambiguity không khắc phục hoàn toàn identity. Sự biến mất của detection không là bằng chứng pixel đã xóa. Category-count guard bảo thủ, có thể từ chối chỉnh sửa đúng nếu detector bỏ sót object khác.

ROI chỉ kiểm tra vị trí thêm/thay, không xác minh đầy đủ mọi chi tiết ngôn ngữ (kích thước/màu/quan hệ mong muốn). Pilot này chưa kiểm định độ tin cậy extractor bằng graph GT, chưa nối predictor Task B và chưa đủ nhãn để chọn trọng số dựa trên thống kê. Giữ ngưỡng/trọng số cố định cho pilot; thay đổi sau khi xem nhãn phải version riêng và báo rõ dùng development data.

## Validation thực tế — 2026-11-15

Implementation commit `1c41aa4`; 53 CPU tests đạt, preflight hash, Bash syntax và toàn bộ pre-commit hooks đạt. Chromium đã kiểm tra 10 ROI, xuất authoring JSON và nhãn pass/fail bằng fixture tổng hợp trong `/tmp`; không tạo approved pack hay nhãn tay thật. Lỗi HTML option “Đạt” tìm thấy qua browser test đã được sửa trước commit.

Inference thật hoàn tất trên CPU (`cpu-smoke-v2`) và RTX 5090 (`gpu-smoke-v1`), mỗi lượt một ảnh/2 bước/256, batch 1, seed gốc 42, seed ảnh 1006579066. Tổng thời gian stage lần lượt khoảng 5.91 s và 3.44 s (không gồm preflight). ONNX profile GPU ghi 2,324 CUDA kernel events và 708 CPU events; có CPU fallback cho một số ops. Safety checker không gắn cờ ở hai lượt. Receipt của cả hai lượt đã xác minh. Trang GPU scores hiển thị hai ảnh và hai directed graphs; trang review ẩn điểm tự động.

Nguồn full resolution có 9 node; nguồn resize 256 có 12 node/27 cạnh, cả 9 anchor gốc còn match. Graph đích smoke CPU có 8 node/16 cạnh; GPU có 4 node/4 cạnh. Đây là evidence về độ nhạy theo preprocessing/sampling, **không phải phép đo độ chính xác**. CPU float32 và CUDA float16 dùng RNG theo device, không đảm bảo ảnh sinh giống nhau dù cùng seed. Điểm số kỹ thuật chi tiết trong [validation JSON](LEVEL3_TASK3_VALIDATION_V1.json) không được dùng làm kết quả benchmark.

Warnings thực tế: `torch_dtype` deprecated, LPIPS upstream gọi torchvision `pretrained` deprecated, ONNX ScatterND cảnh báo giả định indices không trùng. Không sửa upstream đã khóa để che warnings. Logs/screenshots nằm ở `runs/level3/task3-v1/validation-v1/`; các lượt cũ được giữ nguyên. Chưa chạy chính 10 chỉnh sửa, chưa kiểm chứng W&B online của lượt chính, chưa có nhãn tay/Spearman hợp lệ. Bước tiếp theo là người dùng duyệt 10 yêu cầu rồi chạy launcher seed 42.
