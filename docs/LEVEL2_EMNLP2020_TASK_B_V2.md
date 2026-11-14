# Task B — EMNLP 2020 crowdsourced, protocol v2

Pipeline: **source graph + instruction → sáu loại operations → target graph**.
Không dùng ảnh, scene graph extraction, MagicBrush hoặc nhãn VLM. V2 nằm riêng
trong `tools/emnlp2020_task_b_v2/`; không sửa source/config/manifest/results của
MagicBrush hoặc adapter/pilot EMNLP v1 đã khóa.

## Dữ liệu và phạm vi ground truth

Nguồn: [He et al., Findings EMNLP 2020](https://aclanthology.org/2020.findings-emnlp.87/),
[repository chính thức](https://github.com/xlhex/SceneGraphModification), revision
`6e1b396007a0c938bccda2bbddade332745b3f42`. Dùng archive đã được pilot kiểm hash:
`6e13b096f2756b638f59e38b13b80658f91f53cbfa6750a210929fa8a091d337`.
Không giải nén hoặc dùng các biến thể MSCOCO/GCC. Không phân phối raw, nhãn chuyển
đổi, checkpoint, feature cache hoặc case chứa mẫu; quyền phân phối dataset chưa rõ.

Tận dụng nguyên grammar pickle không thực thi, chroot/seccomp/RLIMIT, compiler,
correspondence, canonical identities, `Example`, transaction và evaluator v1.
Reader v2 chỉ thay giao diện danh sách index dài thành `range(count)` để đọc toàn
split trong cùng sandbox; không dùng `pickle.load` hoặc NetworkX constructor.

Giữ toàn bộ dữ liệu gốc và lý do rejection trong artifact cục bộ. Từng split có
đúng index gốc và SHA256 của ba member. Train=official train; val=official dev;
test=official test. Không chuyển mẫu giữa split. Source fingerprint bất biến với
ID/thứ tự, chỉ đọc source; cùng graph nguồn qua split thì split xuất hiện trước
được ưu tiên. Ngay cả source của mẫu bị loại bởi compiler vẫn được reserve.
Trong mỗi split giữ lần đầu của source+instruction chuẩn hóa. Không dùng target
để tạo group, chọn mẫu thay thế, sửa nhãn hoặc bù support.

Đây là **strict source-disjoint subset của các split gốc**, không phải toàn bộ
benchmark gốc. Official counts và exclusions đều được báo. Do không có image ID,
không chứng minh được chống trùng ảnh/caption ban đầu. Graph giống nhau có thể
đến từ các ảnh khác nhau: grouping bảo thủ có thể loại quá nhiều và đổi phân bố
loại chỉnh sửa. Không so trực tiếp metric với paper.

Raw là `networkx.Graph` vô hướng, nhãn `feature`; thuộc tính là node nối bằng
`att`, không có type/owner rõ ràng. Policy A giữ mọi node, mỗi link vô hướng
thành hai cung, self-loop thành một cung. F1 cạnh vì vậy chấm **cung**, không
chấm link vô hướng gốc. Không suy diễn chiều subject/object từ ID; thay màu có
thể là `replace_node` trên node thuộc tính. Không flatten làm mất identity.

Correspondence dùng unique minimum label/retained-link changes trên các ánh xạ
injective cardinality tối đa; ID bằng nhau không phải bằng chứng. Các trường hợp
hòa điểm bị loại. Đây là một giả định correspondence cấu trúc, không phải
identity lịch sử do tác giả phát hành. Target chỉ dùng offline để tạo nhãn và
chấm điểm. Mọi accepted label phải qua topology preservation, reconstruction,
codec round-trip và budgets v1. Sáu loại thao tác được giữ nguyên; loại thiếu
support có metric undefined khi cả GT/prediction đều vắng, không được gán 1.

V2 cho phép nghiên cứu tự động trên supervision từ graph chính thức, tách khỏi
human gate dùng cho export bộ dữ liệu đã kiểm tay ở v1. **Không chuyển human gate
v1 thành pass, không tạo reviewer/verdict.** Chưa kiểm tay thì semantic reliability
vẫn null. Đây chưa phải bằng chứng đã hoàn thành yêu cầu kiểm tay ≥20 mẫu; việc
thay nguồn MagicBrush/InstructPix2Pix vẫn cần ghi nhận với giảng viên.

## Mô hình và giao thức thực nghiệm

Dùng CLIP ViT-B/32 **text encoder đóng băng trên CPU**, cùng weights/hash đã có
trong dự án. Cache chỉ chứa instruction, nhãn node+attributes và predicate của
source. Val/test được áp dụng cùng hàm trích feature; không fit vocabulary,
normalization, prior hay embedding bằng nhãn val/test. CLIP giới hạn 77 token;
cache manifest đếm số text bị truncate.

Tái sử dụng MLP/GNN `EditPredictor`, attention và GRU decoder. Encoder v2 chỉ
bỏ padding thừa, có test so tensor với encoder cũ. MLP không message passing;
GNN có hai message layers. MLP vẫn thấy endpoints trong edge representation,
vì vậy đối chứng đo lợi ích message passing, không phải bỏ hết thông tin cạnh.

Decoder dự đoán chuỗi field: `kind, entity_id, payload...`; arity theo kind
để dựng lại JSON rồi dùng nguyên UTF-8 codec/evaluator v1. Token payload là
chuỗi JSON nguyên vẹn, bao gồm Unicode/escaping/attributes. Vocabulary chỉ fit
train, cộng sáu tên kind và identity token thuộc schema. Không thêm operation
synthetic vào train. Payload ngoài vocabulary ở val/test thành UNK khi tính
teacher loss; khi chấm inference vẫn giữ GT thật và toàn bộ mẫu, không bỏ OOV.
Báo OOV count và upper bound exact match của closed vocabulary.

Cấu hình đã cố định ở `configs/level2/emnlp2020-task-b-v2.json`: hidden 96,
MLP/GNN, 12 epochs, AdamW lr 0.001, batch 128, seeds 42/43/44, CPU 4 threads.
Chọn checkpoint theo minimum validation teacher-forced loss; test không tham gia
selection. Chạy **tuần tự một model/seed mỗi thời điểm**, tự chuyển qua đủ sáu
lượt trong stage train. Cache text batch 128. Không có nhánh CUDA trong runner.

Test chỉ mở sau khi đủ sáu checkpoint và result được khóa bằng SHA256 trong
`frozen-checkpoints.json`; kiểm source/config hashes và trainval receipt trước
khi đọc test. Test sources overlap train/val bị loại bằng group đã khóa, không
đổi train/val dựa vào test. Không tune hoặc chọn model theo test. Baselines:
no-op và nearest training program bằng cosine CLIP instruction + pooled source.
Validation ablations: xáo instruction với permutation cố định; bỏ cạnh khi
inference GNN. Ablation này là diagnostic tại inference, không phải retrain.

Evaluator chấm exact kind/ID/payload, duplicate predictions là FP; GT phải tái
dựng đúng target. Báo P/R/F1 sáu loại, micro, macro trên loại có support, exact
operation set, exact target, valid transaction, parse errors và OOV. Exact match
cho yêu cầu đơn giản dùng proxy **đúng một GT operation**, không khẳng định
độ đơn giản ngôn ngữ. Malformed prediction vẫn ở mẫu số và nhận FN; không sửa
prediction bằng reference target. Chọn tối đa năm đúng/năm sai theo hash ID cố
định trong từng run; giữ full predictions để đối chiếu.

## Chạy lại trên CPU

Môi trường dùng `requirements.txt` và `requirements-dev.txt`, Linux root và
libseccomp (bắt buộc cho sandbox). CLIP weights đã có ở đường dẫn config;
archive đã có ở đường dẫn adapter v1. Nếu thiếu archive dùng downloader v1 đã
kiểm checksum; không tự thay mirror/dataset. Không tải image model mới.

```bash
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m pytest -q tests/editing
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m tools.emnlp2020_task_b_v2 --stage prepare
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m tools.emnlp2020_task_b_v2 --stage features
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m tools.emnlp2020_task_b_v2 --stage train
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m tools.emnlp2020_task_b_v2 --stage test
```

Output dữ liệu/cache: `data/processed/level2/emnlp2020-task-b-v2/`.
Checkpoint, epoch losses, metrics, full predictions và case selections:
`runs/level2/emnlp2020-task-b-v2/{mlp,gnn}-seed{42,43,44}/`.
Mỗi stage chỉ chạy phạm vi đã nêu: prepare/features không train; train tự chạy
sáu lượt tuần tự, không mở test; test chỉ đánh giá checkpoint đã freeze, không
huấn luyện tiếp hoặc khởi chạy hàng đợi khác. Completed run chỉ reuse nếu hash
khớp. Thay code/config phải dùng version output mới; không ghi đè kết quả đã khóa.

Đầu vào inference là JSON có `source_graph` theo schema `EditGraph` và
`instruction`; **không cần target hoặc operations**:

```bash
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m tools.emnlp2020_task_b_v2 \
  --stage predict --model gnn --seed 42 \
  --input /path/to/source-and-instruction.json \
  --prediction-output data/processed/level2/emnlp2020-task-b-v2/demo-prediction.json
```

Inference chuẩn hóa ID chỉ từ source, trả mapping ID, operations và graph tái
dựng; prediction không hợp lệ trả lỗi và target null, không trả GT thay thế.
Batch inference một mẫu, CPU, không tự chạy hàng đợi.

Tổng hợp ba seed, audit dữ liệu và metadata không chứa raw records:

```bash
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m tools.report_emnlp2020_task_b_v2
```

Xem báo cáo tại `reports/level2/emnlp2020-task-b-v2/`. Log kiểm thử ghi rõ
lệnh nguyên bản và CPU test harness dành riêng cho test MagicBrush cũ giả lập
CUDA availability; shim chỉ thay CUDA graph introspection trong process test,
không thay source cũ hoặc che kết quả kiểm thử nguyên bản.
