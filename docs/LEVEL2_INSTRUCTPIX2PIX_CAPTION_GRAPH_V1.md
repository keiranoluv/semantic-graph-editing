# InstructPix2Pix caption graph v1 — Level 2 Task B

Triển khai CPU độc lập trong `tools/instructpix2pix_caption_graph_v1/`, đối chiếu
`Project01_Graph_Editing.pdf`, trang 2–5. Không sửa `src/`, notebook cá nhân,
manifest/config/checkpoint hoặc kết quả GQA/VG150/MagicBrush cũ.

**Pilot thực tế 13/11/2026: 100 mẫu, 0 candidate, 100 bị từ chối. Chưa có bộ
300 cặp và chưa có human review.** Không mở rộng từ pilot thất bại này.
[Báo cáo và bằng chứng](../reports/level2/instructpix2pix-caption-graph-v1/README.md).

## Nguồn đã kiểm chứng

[HF dataset](https://huggingface.co/datasets/timbrooks/instructpix2pix-clip-filtered/tree/aa665b890915f7a42f8615bee868a9f3447e178f),
revision `aa665b890915f7a42f8615bee868a9f3447e178f`: **313.010 dòng, chỉ
official train, 262 shard Parquet**. Không có cột ID riêng; giữ ID tổng hợp từ
revision + tên shard + chỉ số dòng trong shard, global row và tên ảnh gốc.

| Trường | Kiểu HF / kiểu vật lý đã đọc |
|---|---|
| `original_prompt` | string / Arrow string |
| `edited_prompt` | string / Arrow string |
| `edit_prompt` | string / Arrow string |
| `original_image` | Image / struct(bytes: binary, path: string) |
| `edited_image` | Image / struct(bytes: binary, path: string) |

Card ghi **custom license**, hướng dẫn xem licensing file nhưng revision này
không có file LICENSE/COPYING và không có license tag. Điều khoản chưa được
xác minh; không suy diễn CC-BY/Apache. Snapshot card/hash và tree nằm trong
provenance. Số byte trong card không nhất quán với tree; adapter dùng size
thực của từng file và row count từ footer, không tin số download_size của card.

Adapter dùng HTTP Range đúng revision, cache bất biến kèm SHA256 từng range.
Đọc footer toàn bộ shard để kiểm schema/count, rồi chỉ chiếu caption và hai
trường đường dẫn ảnh ở shard có mẫu được chọn. Từ chối server bỏ qua Range
và mọi request range >8 MiB; không tải cột image bytes hay toàn bộ dataset.
Ảnh là stage tùy chọn: `/rows` → kiểm ba caption khớp dòng Parquet đã khóa →
tải riêng hai ảnh → hash byte/RGB. Nếu viewer trả 500/429/ảnh hỏng, lưu lỗi;
không fallback tải các row group ảnh hàng trăm MB. Khi gặp 429, code hiện tại
dừng gửi request mới và ghi các mẫu còn lại là deferred; retry phải gọi riêng.
Viewer API không nhận revision, nên kiểm caption chỉ bảo vệ identity văn bản;
bytes/RGB được khóa từ lần tải, không khẳng định ảnh giống byte của Parquet.

## Parser và giới hạn được khóa

Dùng NLTK 3.9.1 + English averaged perceptron POS tagger đã có trong repo,
kiểm ba hash bằng `tools.magicbrush_grounded_v1.language`. `prepare` tải model
CPU nhỏ này nếu thiếu và kiểm hash trước khi dùng. Không thêm spaCy hay model
thị giác. Tokenizer/ngữ pháp hoàn toàn xác định; POS thành công không đồng
nghĩa graph đúng. Mọi từ phải được grammar tiêu thụ.

Hỗ trợ:

- Cụm danh từ singular/common noun; article tùy chọn, danh sách adjective nhỏ
  có trong `parser.ATTRIBUTES`, một màu; compound trong whitelist công khai.
- Một object, danh sách object nối `and` không có scope quan hệ, hoặc một
  quan hệ không gian có hướng. Chuẩn hóa `grey` → `gray`, không đoán synonym.
- `on`, `under`, `beside/next to`, `in/inside`, `near`, `behind`, `above`,
  `below`, `in front of`, `to the left/right of`, `on top of`, `underneath`.
- Cấu trúc `NP is color`; trạng thái tĩnh `sitting/standing/parked` được giữ
  trong evidence, chỉ cho qua nếu hai caption giữ nguyên trạng thái này.
- Anaphora duy nhất: `NP with NP on it [in NP]`, khi head là một object rõ
  ràng. Ví dụ sofa/pillow/living room được giải đầy đủ; pronoun khác bị chặn.
- Instruction đơn: replace/swap/change/turn, make/paint/color, add/insert,
  remove/delete/erase, move/put/place + spatial predicate, take/move NP off NP.
  `off` chỉ hỗ trợ xóa cạnh `on` khi cả hai object vẫn có trong caption đích.

Từ chối style/global/meta caption, số/đa số lượng, tên riêng/compound không
hỗ trợ, punctuation ngoài dấu chấm cuối, action phức tạp, pronoun chưa giải,
hai object cùng nhãn, nhiều spatial clause mơ hồ hoặc nhiều thay đổi không
được instruction giải thích. Đồ thị là **phần ngữ nghĩa caption được grammar
hỗ trợ**, không phải graph toàn cảnh và chưa phải ground truth nhìn từ ảnh.

Alignment không dựa vào ID serialized: match singleton label chuẩn hóa và
vai trò tương thích; replacement cần đúng một cặp mất/thêm, câu lệnh khớp,
vai trò và context cạnh giữ nguyên. Cạnh match theo cặp endpoint có hướng
duy nhất. `diff_aligned_graphs` sinh nhãn; instruction chỉ kiểm toàn bộ diff,
không sinh thêm object/attribute/edge. Giới hạn 20 node/60 edge **cả hai graph**,
1–12 operation, tối đa 2.048 action token. Validator áp dụng transaction thật,
chuẩn hóa bằng codec cũ, kiểm round-trip decoder và khớp target chính xác.

## Chọn cohort, chia tập và review

Seed 42 tạo permutation toàn bộ 313.010 global row; **100 ID đầu được khóa
trước khi parse**, không thay mẫu thất bại. Prefix này giữ nguyên khi tăng
count. Các family của diagnostic dành cho development/train; pilot không có
validation/test. Khi mở rộng, union các mẫu có cùng caption nguồn chuẩn hóa,
upstream source handle hoặc bất kỳ source/target RGB hash nào trùng nhau.
Các component còn lại chia bằng SHA threshold 70/15/15. `load_examples`
kiểm lại family/caption/handle/pixel leakage và membership.

Split này riêng cho IP2P, dùng schema train/val/test của repo; không áp dụng
random record split hoặc sửa splitter cũ. Caption/hash family có thể gộp quá
nhiều mẫu; không cam kết nhận ra ảnh cùng family nhưng khác cả caption,
handle và pixel. Chưa đủ ảnh thì chưa qua cổng export. Project test không
được đưa vào parser/compiler; không dùng các mẫu test để sửa rules.

Review tái sử dụng viewer, CSV schema và năm trục đánh giá của MagicBrush,
thêm hai caption, parsing/alignment evidence và chỉ báo ảnh thiếu. Hai cohort:

1. Hash sample ngẫu nhiên 20 trong 100 diagnostic, trước lọc. Giữ cả mẫu
   parse thất bại; có đủ ảnh thì người duyệt có thể đánh Sai/Chưa rõ.
2. Hash sample độc lập tối đa 20 candidate train, chọn sau lọc để đo correctness
   **có điều kiện**. Nếu trùng cohort 1 chỉ cần kiểm một lần.

CSV phải có reviewer thật, đủ năm trục và evidence/packet/image hash đúng.
Không cho chấm Đúng một mẫu không tạo được candidate; không cho chấm khi
thiếu hai ảnh. Hash khóa cả caption, instruction, provenance, graph và ảnh.
Mỗi compilation/viewer/packet/CSV submission bất biến; thay dữ liệu hoặc code
thì dùng version mới. Tải bổ sung ảnh tạo compilation mới; review cũ không
được tái dùng với snapshot khác. Không tự duyệt hoặc tạo tên reviewer.

Audit báo riêng sample-wide success trên cohort diagnostic đã kiểm đủ và
correctness trên candidate determinate. Unreviewed/uncertain được giữ trong
thống kê, không trình bày conditional accuracy thành độ đúng của toàn dataset.
Chưa có người thứ hai thì không báo inter-rater agreement. Cổng yêu cầu đủ
20 diagnostic đã review, ít nhất 20 candidate determinate, conditional correct
fraction ≥0,8. Đây là quality threshold protocol v1, không phải metric model.

## Lệnh tái lập trên CPU

Từ repo root, dùng môi trường hiện có và cài các dependency bổ sung nếu thiếu:

```bash
myenv/bin/python -m pip install -r requirements-instructpix2pix-caption-graph-v1.txt
export CUDA_VISIBLE_DEVICES=''
IP2P_OUT=data/processed/level2/instructpix2pix-caption-graph-v1/pilot
myenv/bin/python -m tools.instructpix2pix_caption_graph_v1 inspect --output "$IP2P_OUT"
myenv/bin/python -m tools.instructpix2pix_caption_graph_v1 plan --output "$IP2P_OUT" --count 100 --seed 42
myenv/bin/python -m tools.instructpix2pix_caption_graph_v1 prepare --output "$IP2P_OUT" --images
myenv/bin/python -m tools.instructpix2pix_caption_graph_v1 compile --output "$IP2P_OUT"
myenv/bin/python -m tools.instructpix2pix_caption_graph_v1 review-build --output "$IP2P_OUT"
myenv/bin/python -m tools.instructpix2pix_caption_graph_v1 review-serve --output "$IP2P_OUT" --port 8774
```

`pilot --output "$IP2P_OUT" --seed 42 [--images]` gộp plan 100 → prepare →
compile. Không tự phục vụ viewer, review, expand hay train. Mỗi stage dừng sau
công việc đã chỉ định. Metadata cache/pinned source hỗ trợ resume khi mất mạng;
label cache giữ nguyên input hash, code/package identity; mismatch bị chặn.

Retry ảnh sau khi hết rate limit, rồi compile và build packet mới:

```bash
myenv/bin/python -m tools.instructpix2pix_caption_graph_v1 prepare --output "$IP2P_OUT" --images --retry-images
myenv/bin/python -m tools.instructpix2pix_caption_graph_v1 compile --output "$IP2P_OUT"
```

Khi có nhiều compilation, truyền `--compilation PATH` rõ ràng vào các stage
review/export. Console compile in đường dẫn đầy đủ. Viewer ở localhost, hỗ
trợ xuất CSV và lưu submission bất biến. Dùng đường dẫn thực từ review-build:

```bash
IP2P_COMP=data/processed/level2/instructpix2pix-caption-graph-v1/pilot/compilations/ea344083afddf4f8c317d5a97fe9d261a8ea8eb14682a7707424e1b9fa6272f6
IP2P_REVIEW=reviews/instructpix2pix-caption-graph-v1/ea344083afddf4f8c317d5a97fe9d261a8ea8eb14682a7707424e1b9fa6272f6
myenv/bin/python -m tools.instructpix2pix_caption_graph_v1 review-audit --output "$IP2P_OUT" --compilation "$IP2P_COMP" --review-directory "$IP2P_REVIEW"
```

Không truyền CSV thì audit chỉ báo pending. `review-template.csv` để trống,
không phải approval. Khi có CSV thật, thêm `--reviews-csv PATH_TO_REAL_CSV`.

## Mở rộng và xuất dữ liệu — đang bị chặn

Chỉ chạy sau khi pilot có chất lượng và review đạt cổng. Không dùng command
này để bỏ qua kết quả 0/100 hiện tại. Mặc định không tải ảnh nếu thiếu `--images`.

```bash
IP2P_CSV=PATH_TO_REAL_REVIEW_CSV
IP2P_EXPANDED=data/processed/level2/instructpix2pix-caption-graph-v1/expanded-5000
myenv/bin/python -m tools.instructpix2pix_caption_graph_v1 expand --output "$IP2P_EXPANDED" --pilot-output "$IP2P_OUT" --compilation "$IP2P_COMP" --review-directory "$IP2P_REVIEW" --reviews-csv "$IP2P_CSV" --count 5000 --seed 42
```

Đây là **5.000 raw records**, không hứa 300 candidate. Chỉ xử lý train/val;
giữ test. Parser/code/package identity phải đúng bản đã được review. Thay
grammar thì tạo version/output mới và làm lại pilot; không sửa nhãn test riêng.

Sau expand, tải ảnh riêng, compile và tạo **review mới cho expanded compilation**;
truyền compilation/directory/CSV expanded vào export:

```bash
myenv/bin/python -m tools.instructpix2pix_caption_graph_v1 export --output "$IP2P_EXPANDED" --compilation PATH_TO_EXPANDED_COMPILATION --review-directory PATH_TO_EXPANDED_REVIEW --reviews-csv PATH_TO_EXPANDED_CSV --destination data/processed/level2/instructpix2pix-caption-graph-v1/export-v1
```

Export bắt buộc ≥300 candidate sử dụng được trong train+val sau loại review
incorrect/uncertain, train/val đều khác rỗng, giao dịch/budget hợp lệ, không
leakage, ảnh đã tải/kiểm hash và human gate đạt. Không có cờ force. Output
`dataset.json` có labels, split và audit hashes; không gọi training.

Bridge Python `runner.load_examples(COMPILATION_PATH)` trả đúng
`tools.edit_model_v1.data.Example`, dùng nguyên codec/`collate_inputs`/
`collate_targets`/`EditPredictor`. `Example.model_input()` chỉ trả `source`
và `instruction`; frozen embedding sau này chỉ tạo từ các trường này.
Không dùng launcher MagicBrush cũ với manifest IP2P vì launcher đó khóa
dataset và policy khác. Chưa bàn giao launcher GPU cho nguồn dữ liệu này.

## Kiểm thử

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 myenv/bin/python -m pytest -q
myenv/bin/python -m black --check tools/instructpix2pix_caption_graph_v1 tests/editing/test_instructpix2pix_caption_graph_v1.py
myenv/bin/python -m isort --check-only tools/instructpix2pix_caption_graph_v1 tests/editing/test_instructpix2pix_caption_graph_v1.py
myenv/bin/python -m ruff check tools/instructpix2pix_caption_graph_v1 tests/editing/test_instructpix2pix_caption_graph_v1.py
```

54 test mới kiểm parser thành công/thất bại, đủ sáu operation, alignment đổi
thứ tự ID, drift, pronoun/style/instance trùng, transaction sai, canonical ID,
Parquet thật trên mock HTTP Range, cache corruption, plan/resume, image failure/
rate limit/retry, review tampering, grouped leakage, không compile test và
collation predictor trên CPU. Fixture không được tính vào kết quả dataset.
