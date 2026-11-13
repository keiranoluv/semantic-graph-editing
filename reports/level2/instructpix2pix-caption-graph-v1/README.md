# Pilot InstructPix2Pix caption graph v1 — 13/11/2026

**100 mẫu thật đã xử lý; 0/100 candidate được chấp nhận, 100/100 bị từ chối.**
Pilot chưa thành công và chưa đủ điều kiện mở rộng/train. Không thay các mẫu
thất bại bằng mẫu dễ hơn. Không chạy GPU, VLM, sinh ảnh hoặc downstream training.

## Nguồn và cohort thực tế

[Dataset HF đã khóa](https://huggingface.co/datasets/timbrooks/instructpix2pix-clip-filtered/tree/aa665b890915f7a42f8615bee868a9f3447e178f):
revision `aa665b890915f7a42f8615bee868a9f3447e178f`, **313.010 dòng official
train, 262 shard**; có đủ ba trường string `original_prompt`, `edited_prompt`,
`edit_prompt` và hai trường Image `original_image`, `edited_image`.
Đã kiểm card, HF API và schema/row count footer của toàn bộ shard. Image là
struct bytes/path trong Parquet. Card khai báo custom license, không thấy
licensing file hay license tag tại revision này; điều khoản chưa xác minh.
[Snapshot nguồn](source.json).

Seed 42 lấy 100 dòng đầu của permutation đồng đều toàn bộ tập, trước mọi bộ
lọc. [Cohort](cohort.json) giữ nguyên shard, local/global index, revision và
review IDs. Diagnostic families được dành cho train/development trước parse,
nên pilot này có **100 train / 0 val / 0 test**. Đây không phải bộ train đã sẵn
sàng. Không đọc test để điều chỉnh parser.

## Kết quả đo

| Stage | Qua / mẫu số toàn cohort | Tỷ lệ |
|---|---:|---:|
| Trường caption/instruction hợp lệ | 100/100 | 100% |
| Parse caption nguồn | 0/100 | 0% |
| Parse caption đích | 0/100 | 0% |
| Parse được cả hai caption | 0/100 | 0% |
| Alignment | 0/100 | 0%; chưa có mẫu đủ điều kiện đánh giá |
| Instruction consistency | 0/100 | 0%; chưa có mẫu đủ điều kiện đánh giá |
| Structural validation | 0/100 | 0%; chưa có mẫu đủ điều kiện đánh giá |
| Candidate chấp nhận | 0/100 | 0% |
| Cả hai ảnh đã cache, decode và hash | 7/100 | 7% |

Tỷ lệ conditional của alignment/consistency/structure là **null**, không
được diễn giải như 100 giao dịch sai. [Summary máy đọc được](summary.json)
có cả mẫu số toàn cohort và số mẫu thực sự đã đánh giá ở từng stage.

Lý do từ chối parser (mỗi caption có một lý do đầu tiên; nguồn/đích tính riêng):

| Lý do | Caption nguồn | Caption đích |
|---|---:|---:|
| Punctuation/số/độ dài ngoài grammar | 49 | 48 |
| Style/global/meta-caption | 27 | 27 |
| Cụm danh từ không được hỗ trợ | 20 | 22 |
| Object/compound ngoài phạm vi | 3 | 2 |
| Spatial attachment mơ hồ | 1 | 1 |

Không thao tác nào có support trên dữ liệu thật: `add_node`, `delete_node`,
`replace_node`, `add_edge`, `delete_edge`, `replace_edge` đều **0**. Kích thước
graph và số operation trung bình trên candidate là **null**, vì cohort
candidate rỗng. Sáu loại đều được kiểm bằng fixture CPU; fixture không được
tính vào 100 mẫu hoặc yêu cầu ≥300.

Kết luận từ diagnostic: grammar v1 chưa phù hợp caption web tự do của nguồn
này. Kết quả không chứng minh rằng mọi caption bị loại đều sai; chúng nằm
ngoài phạm vi mà parser hiện tại có thể diễn giải đầy đủ. Không nới rules hoặc
chỉ lấy các ca dễ để làm đẹp số pilot. Nếu tiếp tục nghiên cứu, cần grammar
version mới, provenance riêng và pilot/review lại; không mở rộng v1 hiện tại.

## Ảnh, thời gian và bộ nhớ

Lượt lấy ảnh ban đầu nhận **29 HTTP 500** (`Scan size limit exceeded`, row
group ảnh vượt giới hạn scan 300 MB) và **64 HTTP 429**; 7 cặp thành công.
Mọi lỗi có receipt riêng, không thay bằng ảnh fixture hay tự tải cả shard.
Sau lượt này, code được bổ sung dừng request mới ngay khi gặp 429; vì vậy
lượt retry mới có thể báo deferred thay cho nhiều phản hồi 429 lặp lại.
Snapshot 100 receipt gốc vẫn được giữ nguyên; report không sửa số đo cũ.

| Phạm vi số đo | Thời gian | Peak process RSS |
|---|---:|---:|
| Chiếu metadata 100 mẫu, 8 worker | 54,62 giây | 753,21 MiB |
| Lượt tải ảnh ban đầu | 55,90 giây | 630,15 MiB |
| Compile từ metadata đã cache | 0,316 giây | 724,71 MiB |

Các stage đo trong process riêng; không cộng RSS. RSS bao gồm import thư
viện của repo. Chưa đo riêng wall time dựng index 262 footer; không suy diễn
throughput mở rộng từ compile đã cache. Cache HTTP Range chứa **32.987.938
byte** payload, gồm footer và các cột văn bản/đường dẫn cần thiết; không tải
toàn bộ dataset. [Chi tiết runtime](runtime.json).

## Review, gate và tính tái lập

20 diagnostic review IDs đã khóa trước parse, chỉ **2/20** hiện có đủ hai
ảnh. Viewer đã build, có hai caption, instruction, graph/thao tác nếu có,
parsing/alignment/validation evidence và lỗi. **0 người/mẫu đã review**;
correctness/annotation reliability là **null**. Không tạo reviewer hay verdict.
Conditional candidate review cohort có 0 mẫu, không có accuracy để báo.
[Trạng thái review](review-status.json).

Packet SHA256:
`a99af8e049a453de6e0d2148f4fe257136bd48a35126f8360203bb9cdb55099a`.
Run SHA256:
`0e26fc80151dad3eae7225fadb6682beb17e649e797f744fcfef48372d38d0c0`.
Compilation:
`ea344083afddf4f8c317d5a97fe9d261a8ea8eb14682a7707424e1b9fa6272f6`.
[Run/code/package identity](run.json), [hash ledger](provenance.json).

Đã chạy lại compile từ cache và xác nhận `edits.json` giữ nguyên byte/hash.
Export bị chặn bởi review chưa đạt, <300 candidate, train/val usable rỗng.
Expand bị chặn bởi quality gate của pilot. Không có cờ bỏ qua các điều kiện.

Toàn bộ kiểm thử CPU: **595 passed, 19 skipped**, một warning `pin_memory`
do không có accelerator. Trong đó **54 test mới pass**. Black, isort, Ruff
và mọi hook pre-commit áp dụng cho các file mới đều pass. Không sửa source
hoặc artifact cũ để làm test qua.

## Tái lập và sử dụng

[Hướng dẫn đầy đủ, grammar, split/review policy và command mở rộng có gate](../../../docs/LEVEL2_INSTRUCTPIX2PIX_CAPTION_GRAPH_V1.md).
Từ repo root:

```bash
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m tools.instructpix2pix_caption_graph_v1 pilot --seed 42
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m tools.instructpix2pix_caption_graph_v1 review-serve --port 8774
```

Command đầu tái dùng dữ liệu/receipt bất biến hiện có; thêm `--images` chỉ
ở lần chuẩn bị ảnh. Retry ảnh phải dùng `prepare --images --retry-images`.
Mở rộng chỉ được gọi riêng sau pilot thành công và review thật; không tự
chạy tiếp hàng đợi, huấn luyện hay inference GPU.

Thay đổi nằm trong module mới `tools/instructpix2pix_caption_graph_v1/`, một
test file, requirements CPU riêng, tài liệu này và ignore cho viewer sinh ra.
Bridge sử dụng `Example`/codec/collator hiện có; inference input chỉ graph
nguồn + instruction embedding đóng băng. Manifest này có policy riêng,
không dùng trực tiếp launcher MagicBrush đã khóa.
