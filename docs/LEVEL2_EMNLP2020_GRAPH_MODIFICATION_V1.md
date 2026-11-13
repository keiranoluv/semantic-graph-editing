# EMNLP 2020: adapter graph có sẵn cho Task B

Đây là **phương án dữ liệu thay thế, chờ giảng viên chấp thuận**, không tự
đáp ứng yêu cầu nguồn MagicBrush/InstructPix2Pix ở trang 3 của
[đề bài](../Project01_Graph_Editing.pdf). Chưa hoàn thành bộ ≥300 cặp và kiểm
tay ≥20 mẫu. Không train, inference thị giác, GPU, API trả phí hoặc tải VLM.

## Nguồn đã kiểm tra

[He et al., Findings EMNLP 2020, trang 972–990](https://aclanthology.org/2020.findings-emnlp.87/),
DOI `10.18653/v1/2020.findings-emnlp.87`;
[code chính thức](https://github.com/xlhex/SceneGraphModification) tại commit
`6e1b396007a0c938bccda2bbddade332745b3f42`.
README liên kết trực tiếp tới
[archive Google Drive](https://drive.google.com/file/d/1K2lo1Dt7GJskyUVR9x5LH-mZya28KcDY/view?usp=sharing).
Archive SHA256:
`6e13b096f2756b638f59e38b13b80658f91f53cbfa6750a210929fa8a091d337`.
Không tìm thấy LICENSE trong repository tại revision này, điều khoản dataset
trong README, hay license trong archive. Giấy phép bài báo không tự cấp quyền
phân phối dataset. Giữ dữ liệu/nhãn/viewer chứa mẫu ở máy cục bộ; cần hỏi tác
giả trước khi phân phối lại. Hash nguồn và artifact ở báo cáo inspection.

Archive duy nhất khoảng 67 MiB chứa MSCOCO, GCC và crowdsourced; không có
download crowdsourced riêng trong README. Vì gzip/tar không có random access,
phải tải archive nhưng không extract toàn bộ. Chỉ đọc ba member train
crowdsourced cho pilot, không mở nội dung dev/test hay biến thể khác.
Đếm được **30.051** graph nguồn, graph đích và dòng instruction train, thay
vì coi con số 30K trong paper là kích thước chính xác. Kích thước dev/test
thực tế chưa đo; paper mô tả khoảng 1K/1K. Liệt kê tên/kích thước member test
trong tar không phải đọc graph hoặc instruction test.

## Schema thật và giới hạn diễn giải

File `.bin` là **chuỗi pickle độc lập**, mỗi record một
`networkx.classes.graph.Graph` protocol 3, không phải pickle của một list.
Node: integer ID và `{"feature": "label"}`; cạnh: hai endpoint và
`{"feature": "predicate"}`; không có edge ID. Node thuộc tính vẫn là node,
nối bằng `att`. Không có cờ object/attribute. Graph metadata rỗng, có cached
`NodeView`. Không có cặp ảnh hoặc URL ảnh trong các record này.

Paper nói về graph có hướng và sắp topo; file phát hành là `Graph` vô hướng.
`code/data_utils.py` tạo adjacency đối xứng, còn renderer có quy tắc trình bày
riêng cho `att`. Thứ tự node có thể phản ánh convention ban đầu, nhưng code
công khai không cung cấp bảo đảm đầy đủ để phục hồi mọi chiều subject/object.
V1 không suy ra chiều chỉ từ `min(id), max(id)`. Đây là giới hạn của supervision
được xuất: chấm theo graph đã phát hành, chưa phải ground truth đồ thị cảnh
có hướng hoặc tính đúng với ảnh thật.

ID **bị đánh lại**. Ví dụ train index 27824: `grass=0, ground=1, green=2`
trở thành `grass=0, green=1` sau xóa ground; `green` không phải `ground`
dù cùng mang số 1 giữa hai graph. Có cả xóa object kèm subtree thuộc tính,
không chỉ chênh đúng một node. Cohort trước lọc được khóa bằng hash, mọi lỗi
được giữ; không lấy mẫu dễ thay thế.

## Hai policy thuộc tính

| Policy | Quy tắc | Hệ quả |
|---|---|---|
| A, được chọn | Giữ toàn bộ node/nhãn, mỗi link vô hướng thành hai cung ngược chiều; self-loop thành một cung | Bảo toàn chính xác topology lưu trong file; số thao tác cạnh thường gấp đôi link |
| B, đối chứng | Chỉ chuyển graph không có `att`; từ chối flatten khi thiếu bằng chứng owner/type | Không tự đoán object theo degree/ID/từ loại, không âm thầm mất identity |

Không có phép flatten được chứng minh là không mất thông tin trên các cấu
trúc `att` thực tế. Do đó B báo lỗi ownership thay vì tạo thuộc tính theo
heuristic. Báo riêng phân bố thao tác của tập B chuyển được; không so tỷ lệ
thành công trên hai mẫu số khác nhau mà bỏ qua các mẫu từ chối.

Một thay màu theo A là `replace_node` trên node màu, không tự sửa trường
attributes của object. Tất cả sáu phép của repository giữ nguyên semantics:
INSERT → `add_node` cùng `add_edge` liên quan; DELETE → `delete_node` cùng
`delete_edge` tường minh; SUBSTITUTE → `replace_node` khi correspondence
hợp lệ. Link đổi predicate trên cùng cặp endpoint có correspondence duy
nhất → `replace_edge`; đổi endpoint không có edge ID → delete+add. Loại
không có support trong pilot không được tạo giả để đủ sáu loại.

## Alignment và kiểm định

Không dùng integer ID bằng nhau để căn chỉnh. Với graph tối đa bảy node,
duyệt mọi correspondence injective có cardinality tối đa; ưu tiên số nhãn
thay đổi ít nhất, rồi số link giữa các node giữ lại thay đổi ít nhất. Chỉ nhận
**một** nghiệm tối ưu; hòa điểm → `ambiguous_node_identity`. Với insert/delete,
nhãn và mọi link giữa các node còn lại phải giữ nguyên. Substitute cho phép
đổi một nhãn và giữ topology. Giới hạn tìm kiếm được báo riêng, không suy
đoán với graph lớn. Nhãn trùng có thể giải bằng cấu trúc; đối xứng còn lại bị
từ chối. Đây là correspondence cấu trúc bảo thủ, không chứng minh được identity
lịch sử mà tác giả không phát hành.

Source IDs/cạnh được chuẩn hóa **chỉ từ source**. Target chỉ dùng offline
để căn chỉnh/sinh nhãn. Dùng nguyên `diff_aligned_graphs`, `apply_operations`,
`normalize_label`, UTF-8 action codec và `Example` hiện có. Kiểm topology
chính xác với raw source/target; kiểm transaction tái dựng đích; kiểm encode /
decode; không duplicate semantic edge, dangling endpoint hoặc hai lần ghi
lên một entity. Node/edge mới dùng canonical identity policy hiện có.
Budget: graph nguồn/đích ≤20 node, ≤60 cung; 1–12 thao tác; ≤2048 action tokens.

Loader mới `tools.emnlp2020_graph_modification_v1.data.load_examples` trả
cùng `Example` mà encoder/decoder Task B đang dùng. `model_input()` chỉ có
source và instruction; dùng được `feature_texts`, codec, transaction evaluator.
**Manifest ảnh của launcher MagicBrush không tương thích**: nó yêu cầu ảnh,
pixel hash và mọi record thuộc official train. Không tạo ảnh/hash giả hay
đổi official dev thành train. Chưa có launcher huấn luyện cho nguồn mới;
không truyền manifest này cho `tools.train_magicbrush_edits`.

Chia train=official train, val=official dev, test để nguyên chưa mở. Hash
source graph bất biến theo ID/thứ tự dùng làm source group; loại dev trùng
source train và source+instruction lặp, giữ nguyên diagnostics/provenance.
Không có ID ảnh nên không chứng minh được chống trùng ảnh/caption gốc ngoài
những gì graph release cho phép. Graph cùng nội dung bị gom bảo thủ, có thể
loại cả các ảnh khác nhau. Không sử dụng target để tạo source group.

## An toàn pickle và tái lập

Không gọi `pickle.load`, `Unpickler`, constructor NetworkX hay import GLOBAL
do dữ liệu chỉ định. Reader dùng `pickletools.genops` với grammar whitelist;
GLOBAL/NEWOBJ/BUILD chỉ dựng symbol/state trơ, REDUCE/persistent IDs và schema
lạ bị từ chối. Reader thật vẫn chạy trong subprocess `python -I`, chroot rỗng,
UID/GID 65534, xóa environment, đóng FD thừa; seccomp whitelist chặn open,
socket, exec và fork, giới hạn address space 512 MiB, CPU 60 giây và timeout.
Không có writable host mount/credentials. Linux root/chroot và libseccomp
là điều kiện của runner này; thiếu thì fail closed, không fallback unpickle
ngoài sandbox. Unit test còn kiểm syscall đọc file/network/exec bị chặn.

Chỉ materialize 100 chỉ số train đã chọn trước lọc theo SHA256 của
`[adapter_version, 42, "train", original_index]`. Phải quét opcode của stream
để định vị record và đếm 30.051 frame, nhưng chỉ giải cấu trúc 100 cặp đã chọn.
Provenance ghi SHA archive/member/cohort/raw/manifest/edits, revision upstream,
commit chạy và hash source cụ thể. Không ghi đè khác nội dung. Chạy lại cùng
code/input trả artifact cũ sau kiểm hash. Thay code → output version mới,
cohort vẫn cùng indices. Nếu sửa nhãn, review cũ không tự chuyển sang.

## Lệnh CPU và review

```bash
myenv/bin/python -m tools.emnlp2020_graph_modification_v1 --stage inspect --download
myenv/bin/python -m tools.emnlp2020_graph_modification_v1 --stage pilot --report-dir reports/level2/emnlp2020-crowdsourced-pilot-v1
myenv/bin/python -m tools.emnlp2020_graph_modification_v1 --stage review-build
myenv/bin/python -m tools.emnlp2020_graph_modification_v1 --stage review-serve --port 8775
```

Output mặc định: `data/processed/level2/emnlp2020-crowdsourced-pilot-v1-run1/`;
viewer `reviews/emnlp2020-crowdsourced-pilot-v1-run1/`, mở
`http://127.0.0.1:8775/`. Chỉ CPU, không model/batch GPU, seed chọn mẫu 42,
không tự chạy hàng đợi hoặc mở rộng. Raw/nhãn/viewer chứa dữ liệu không commit.

Review gồm 20 train chẩn đoán chọn trước lọc và tối đa 20 candidate chọn
độc lập; overlap chỉ kiểm một lần. Xem graph gốc/chuyển đổi, instruction,
operations, correspondence, mọi lỗi/lossy flag. Người kiểm ghi tên thật,
năm trục và kết luận, lưu CSV qua viewer. Nhấn lưu không tự duyệt cả cohort.
`source_graph`/`target_graph` đánh giá tính trung thành chuyển đổi với graph
gốc, không thay cho nhận định graph đúng với ảnh. Chưa có review thì reliability,
agreement và semantic accuracy là `null`, không ghi 0% hay giả người kiểm.

```bash
EMNLP_REVIEW_CSV=reviews/emnlp2020-crowdsourced-pilot-v1-run1/submissions/review-REPLACE_WITH_ACTUAL_HASH.csv
myenv/bin/python -m tools.emnlp2020_graph_modification_v1 --stage review-audit --reviews-csv "$EMNLP_REVIEW_CSV"
myenv/bin/python -m tools.emnlp2020_graph_modification_v1 --stage plan --reviews-csv "$EMNLP_REVIEW_CSV"
```

Gate giữ yêu cầu hiện có: chẩn đoán ≥20 mẫu kiểm hết; candidate ≥20 kết luận
đúng/sai, đúng ≥80%. `uncertain` không tính vào determinate; candidate sai/chưa
rõ bị loại khỏi export. CSV/hash/selection/five-axis phải khớp; không nhận CSV
template trống. Chưa đạt gate thì `plan`/expand/export chặn, không hạ ngưỡng.

## Mở rộng riêng sau khi kiểm pilot

Chưa chạy các lệnh sau. Chúng chỉ xử lý cohort hữu hạn, không tự chạy full
dataset hoặc test; train/dev tiếp tục tách official. 500 train +100 dev là
số **trước lọc**, không bảo đảm ≥300 dùng được. Có thể cần cohort version mới
nếu dedup/budget/review loại nhiều; không thay mẫu lỗi trong cohort đã khóa.

```bash
myenv/bin/python -m tools.emnlp2020_graph_modification_v1 --stage expand --reviews-csv "$EMNLP_REVIEW_CSV" --train-size 500 --dev-size 100 --destination data/processed/level2/emnlp2020-crowdsourced-expanded-v1
```

Expanded train giữ 100 pilot ở đầu cùng hash ordering. Export chỉ chuyển
review pilot khi **mọi evidence label đã kiểm không đổi**. Nếu dedup hoặc
conversion đổi evidence, phải build/review packet expanded mới; không ép
chuyển verdict. Mặc định rõ ràng nhất là kiểm candidate cohort expanded:

```bash
myenv/bin/python -m tools.emnlp2020_graph_modification_v1 --stage review-build --output data/processed/level2/emnlp2020-crowdsourced-expanded-v1
myenv/bin/python -m tools.emnlp2020_graph_modification_v1 --stage review-serve --output data/processed/level2/emnlp2020-crowdsourced-expanded-v1 --port 8775
EMNLP_EXPANDED_CSV=reviews/emnlp2020-crowdsourced-expanded-v1/submissions/review-REPLACE_WITH_ACTUAL_HASH.csv
myenv/bin/python -m tools.emnlp2020_graph_modification_v1 --stage export --output data/processed/level2/emnlp2020-crowdsourced-expanded-v1 --reviews-csv "$EMNLP_EXPANDED_CSV" --destination data/processed/level2/emnlp2020-crowdsourced-expanded-v1/task-b-export.json
```

Export yêu cầu ≥300 usable train+val, cả hai split có mẫu, transaction/codec
hợp lệ, không leakage và review đạt gate. Export vẫn ghi chờ giảng viên về
nguồn dataset. Các phép thử synthetic 300 chỉ kiểm contract, không tính dữ
liệu thật. Lệnh kiểm CPU:

```bash
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m pytest -q tests/editing
myenv/bin/pre-commit run --files $(git ls-files tools/emnlp2020_graph_modification_v1 tests/editing/test_emnlp2020_graph_modification_v1.py docs/LEVEL2_EMNLP2020_GRAPH_MODIFICATION_V1.md)
```

Kết quả đo pilot, phân bố sáu phép, size/operation distributions, runtime/RSS
và các giới hạn còn lại ở [báo cáo pilot](../reports/level2/emnlp2020-crowdsourced-pilot-v1/README.md).
