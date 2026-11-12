# MagicBrush: pilot gán nhãn tự động có bằng chứng ảnh, v1

Hướng đã chọn: **tự động, lọc nghiêm ngặt, người kiểm một mẫu**. Thay việc sinh
hai scene graph độc lập rồi lấy hiệu bằng pipeline neo object theo ảnh nguồn.
Đây là pilot để kiểm chứng phương pháp mới, chưa phải bộ ≥300 nhãn đã đạt chất lượng.
Không dùng nhãn Qwen v1/v2 để chạy full hoặc train trong nhánh này.

## Phương pháp và phạm vi

1. BLIP base, frozen, chỉ caption **ảnh nguồn**. NLTK POS tagger tách cụm danh từ
   để đề xuất object ngữ cảnh. Caption không được coi là bằng chứng có object.
2. Bộ phân tích câu lệnh theo grammar công khai đề xuất object cần kiểm tra.
   Grounding DINO tiny tìm box của các query trong cả hai ảnh, cùng vocabulary.
   Không đưa toàn bộ câu yêu cầu vào model để nhờ model tự dựng đáp án.
3. Frozen CLIP ViT-B/32 kiểm tra crop, correspondence và màu; kiểm thêm thay đổi
   pixel và phân bố HSV. Instruction chỉ chọn điều kiện cần kiểm, không thay thế
   bằng chứng ảnh đích.
4. Graph nguồn gồm object đã grounding và quan hệ **thứ tự tâm bbox**:
   `left_of`, `right_of`, `above`, `below`. Đây là **graph cục bộ, phạm vi vị trí**,
   không phải graph quan hệ ngữ nghĩa đầy đủ. Không suy đoán `on`, `holding`,
   `displaying`, hành động, tiếp xúc hay chiều sâu từ bbox.
5. Giữ nguyên ID, thuộc tính và geometry của ngữ cảnh chỉ khi crop cùng tọa độ
   đã vượt kiểm tra ổn định. Box jitter ở đổi màu không tạo edge edits.
   Object thay lớp tại cùng vùng dùng `replace_node` và giữ ID. ID object nguồn
   không bị đổi chỉ vì detector đánh số ảnh đích khác.
6. Kiểm độc lập việc thêm/xóa: score detector của lớp ở ảnh đối diện phải thấp,
   **đồng thời** crop phải thay đổi, CLIP appearance khác và CLIP class contrast
   đủ lớn. Detector bỏ sót một object không tự tạo nhãn xóa. Đổi màu cần CLIP
   margin và tỷ lệ pixel màu phù hợp. Movement cần correspondence duy nhất và
   thay đổi quan hệ vị trí đúng yêu cầu.
7. Loại khỏi tập candidate nếu correspondence mơ hồ, reference không duy nhất,
   ngữ cảnh thay đổi, query focus chồng chéo, có detection đích không giải thích
   được, hoặc thay đổi ảnh ngoài vùng edit vượt ngưỡng. Target graph là giao dịch
   cục bộ trên nguồn, không lấy hiệu hai graph tự do.

Ngưỡng ghi trong `configs/level2/magicbrush-grounded-anchor-v1.json` và khóa bằng
hash. Đây là heuristic, **không phải xác suất nhãn đúng**. Exact reconstruction chỉ
kiểm đúng giao dịch, không chứng minh graph/nhãn đúng ngữ nghĩa.

Sáu loại thao tác trong schema cũ được tái sử dụng. Pilot có thể thiếu support
thực tế cho một số loại; không tạo thêm thao tác để đủ thống kê.

Các nguồn chính thức, revision đã khóa trong config:

- [BLIP image-captioning base](https://huggingface.co/Salesforce/blip-image-captioning-base),
  revision `82a37760796d32b1411fe092ab5d4e227313294b`.
- [Grounding DINO tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny),
  revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`.
- CLIP dùng lại weights trong `data/cache/vg150-clip-vit-b32-v1/weights/`,
  SHA256 ghi trong config, không ghi đè cache cũ.
- NLTK tagger có SHA256 của ZIP và từng JSON trong code; `prepare` chỉ tải tài
  nguyên POS nhỏ này nếu thiếu. Không nâng cấp môi trường Python hiện tại.

Đề của thầy cho phép caption parsing khi tạo graph trước/sau (PDF trang 2);
Qwen không bắt buộc. Phạm vi spatial graph của pilot cần được ghi rõ khi báo cáo.
Đã kiểm tra adapter RelTR nhưng không chọn làm phương án duy nhất: vocabulary
object đóng và không có attribute head không phù hợp nhiều yêu cầu đang có.

## Dữ liệu và giới hạn hiện tại

- Giữ nguyên 602 cặp gốc, ảnh, instruction, hash và split theo nhóm.
- Chuẩn bị chỉ phân tích **415 project-train instructions**, đều lấy từ official
  MagicBrush train. Không đọc official dev/test; không chọn ngưỡng bằng val/test.
- Grammar v1 hỗ trợ **64/415** train requests: 18 add, 8 delete, 25 replace,
  13 color; không có move request được grammar này nhận trong train hiện tại.
  Đây là phạm vi câu lệnh, **chưa phải 64 nhãn đúng**.
- 351 request còn lại có lý do unsupported trong `preparation.json`: reference
  đại từ/mơ hồ, nhiều object/count, nhiều thao tác, modifier/placement chưa xác
  minh được, style/action/attribute ngoài màu hoặc cú pháp chưa hỗ trợ. Giữ lại
  ảnh để mở rộng phương pháp sau; không đánh đồng pseudo-label lỗi với cặp ảnh lỗi.
- Chọn 20 ID trong số request được hỗ trợ bằng hash(seed, pair_id), **trước
  inference và trước biết chất lượng**, lưu cả 20 ID review cũ để truy vết.
  Không thay cặp bị fail bằng một cặp dễ hơn sau khi có output.
- Hai ảnh có thể khác resolution (ví dụ 500px → 1024px). So sánh trong frame
  256×256 resize bilinear được ghi rõ; aspect ratio đổi quá 0.01 thì defer.
  Không giả định có registration/segmentation GT.

Không có CSV review thực được tạo sẵn. Candidate vẫn có
`human_review_status=pending`, `eligible_for_training=false`,
`automatic_approval=false`. Code này không mở khóa trainer cũ.

Các lượt spot-check Qwen được phản ánh bằng mô tả lỗi của người dùng; chưa có N
và CSV chấm đủ để kết luận tỷ lệ sai của toàn bộ dữ liệu là 100%.

## Lệnh chạy theo thứ tự

Chạy từ root repo, dùng môi trường `myenv` đã có. Theo `AGENTS.md`: commit code
trước, **người dùng tự chạy lệnh GPU**. Chỉ `--stage infer` sử dụng GPU.

**1. CPU, tái tạo manifest và xem phạm vi.** Idempotent; không ghi đè artifact
khác nội dung. Assistant đã thực hiện bước này.

```bash
myenv/bin/python -m tools.annotate_magicbrush_grounded_v1 --stage prepare
myenv/bin/python -m tools.annotate_magicbrush_grounded_v1 --stage plan
```

**2. Người dùng chạy GPU pilot 20 cặp, không train.** BLIP → DINO → CLIP chạy
từng model một và giải phóng GPU giữa các stage. Batch caption/detector **4**,
CLIP crops/text **64**, seed **42**, float32. Chưa benchmark VRAM/throughput;
không suy từ card 32GB rằng đã kiểm chứng batch lớn hơn. Có thể download frozen
model weights ở lần đầu. Chỉ chạy một lệnh này, không có hàng đợi full/train.
DINO tắt custom CUDA kernels để tránh phụ thuộc biên dịch extension lúc chạy.

```bash
mkdir -p logs
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 myenv/bin/python -u -m tools.annotate_magicbrush_grounded_v1 --stage infer --wandb-mode online > logs/magicbrush-grounded-anchor-pilot-v1.log 2>&1
```

Tên W&B: `magicbrush-grounded-anchor-v1-pilot-bs4-clipbs64-s42`, tags có
`frozen-inference`, `pilot-20`, `no-training`. Nếu offline, thay bằng
`--wandb-mode offline`. Muốn đổi batch/ngưỡng phải dùng config/output **version
mới**; không sửa config đã khóa rồi resume cùng cache.

Cache caption, detection và crop evidence nằm dưới
`data/processed/level2/magicbrush-grounded-anchor-v1/`. Rerun cùng lệnh sẽ kiểm
hash và dùng lại cache đã có. Lỗi model không bị che bằng graph/nhãn mẫu.

**3. CPU, compile và mở viewer sau khi pilot hoàn tất.** Inference đã tự compile
một lần. Compile lại không chạy GPU. Những lần compile chưa đủ evidence cũng
có snapshot riêng; không ghi đè snapshot đã chấm.

```bash
myenv/bin/python -m tools.annotate_magicbrush_grounded_v1 --stage compile
myenv/bin/python -m tools.annotate_magicbrush_grounded_v1 --stage review
myenv/bin/python -m tools.annotate_magicbrush_grounded_v1 --stage serve --port 8767
```

Mở `http://127.0.0.1:8767/` (forward port nếu chạy trên máy từ xa). Viewer dùng
ảnh thật, box detector, instruction, source/target graph, thao tác và lý do bộ
lọc. Kiểm cả cặp bị loại; chấm năm trục rồi **Lưu vào reviews/**. Quyết định vẫn
do người duyệt nhập; không tự đánh dấu approved, kể cả candidate.

Output viewer/CSV/packet/audit nằm dưới `reviews/magicbrush-grounded-human-v1/`.
Packet được archive theo hash; browser không nhập lại quyết định khác phiên bản
bằng chứng. Bản viewer pending hiện tại chưa có model evidence để chấm graph.

**4. CPU, kiểm kết quả CSV thực đã lưu.** Dùng đường dẫn file thực từ nút Save,
không dùng nguyên placeholder dưới đây:

```bash
myenv/bin/python -m tools.annotate_magicbrush_grounded_v1 --stage audit --review-csv 'reviews/magicbrush-grounded-human-v1/submissions/review-<HASH>.csv'
```

Audit kiểm manifest/packet/evidence/CSV hash và ID, cần đủ 20 đánh giá determinate,
tỷ lệ correct ≥90%. Uncertain không được coi là correct. Đây là ngưỡng kiểm pilot
đã công khai, không phải chứng nhận độ tin cậy toàn dataset; không tự chạy tiếp.

## Sau pilot

Nếu nhãn qua kiểm có chất lượng nhưng thiếu số lượng: thu thêm cặp **official
train** có yêu cầu thuộc phạm vi hỗ trợ, tạo manifest mới giữ nguyên dữ liệu và
split cũ, rồi kiểm một mẫu candidate có seed cố định. Không ép 351 yêu cầu ngoài
phạm vi thành đáp án, không yêu cầu người dùng gán nhãn tay toàn bộ.

Nếu ít candidate hoặc review vẫn sai: dùng crop/score/mask và lý do reject để
xác định bottleneck; sửa thành version mới và kiểm lại trước mở rộng. Không hạ
ngưỡng chỉ để đủ ≥300, không mở khóa train bằng nhãn chưa đạt.

Chưa có full-inference launcher hoặc trainer gắn với bộ nhãn mới: cần kết quả
pilot thực trước khi khóa recipe tiếp theo. Các launcher cũ vẫn giữ nguyên để
tái lập lịch sử; không dùng chúng để bỏ qua bước kiểm chất lượng này.

## Kiểm tra CPU

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 myenv/bin/python -m pytest tests/editing/test_magicbrush_grounded_v1.py -q
```

Kiểm ID ổn định, không xóa do non-detection đơn thuần, contrast/locality, thay
lớp không delete/add lại instance, đổi màu không tạo edge jitter, movement,
ambiguous reference, query token spans và DINO `-inf` padding, mask roundtrip.
Fixture tổng hợp chỉ kiểm code, không được tính thành nhãn hay độ tin cậy thật.
Đã probe processor BLIP/DINO bằng ảnh CPU giả để kiểm shape/token offset;
không có pretrained model inference hoặc CUDA computation trong kiểm thử đó.
