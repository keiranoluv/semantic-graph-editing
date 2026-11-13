# Pilot EMNLP 2020 crowdsourced v1 — 100 mẫu thật, CPU

Ngày 13/11/2026. **Dataset hứa hẹn cho supervision theo graph phát hành:
100/100 mẫu chuyển được và tái dựng đúng target. Chưa có kết luận kiểm tay
ngữ nghĩa, chưa hoàn thành ≥300 cặp, chưa được coi là đáp ứng nguồn dữ liệu
của đề gốc.** Cần giảng viên chấp thuận thay MagicBrush/InstructPix2Pix.

Code chạy pilot: **`fc9b392`**, full hash và từng file source nằm trong
[summary.json](summary.json). Cài đặt, mapping, chính sách dữ liệu và các
lệnh mở rộng ở [hướng dẫn adapter](../../../docs/LEVEL2_EMNLP2020_GRAPH_MODIFICATION_V1.md).

## Nguồn và cohort

He et al., *Scene Graph Modification Based on Natural Language Commands*,
Findings of EMNLP 2020, trang 972–990,
[paper](https://aclanthology.org/2020.findings-emnlp.87/),
[repository chính thức](https://github.com/xlhex/SceneGraphModification).
Revision upstream `6e1b396007a0c938bccda2bbddade332745b3f42`,
hash code/paper ở [upstream-evidence.json](upstream-evidence.json).
Liên kết archive được xác minh trực tiếp từ README, không dùng mirror hay
thay biến thể khi gặp lỗi.

Archive SHA256
`6e13b096f2756b638f59e38b13b80658f91f53cbfa6750a210929fa8a091d337`;
đây là checksum tính từ download qua link chính thức, không phải chữ ký do
tác giả phát hành. Archive có MSCOCO, GCC và crowdsourced. Không tìm thấy
license dataset/code rõ ràng trong README/repository/archive được kiểm tra;
**không commit hoặc phân phối raw records, nhãn hay packet chứa dữ liệu**.
Các file được commit ở đây là hash, chỉ số và thống kê của lần kiểm chứng.

Crowdsourced train thực tế có **30.051 dòng instruction và 30.051 record
trong mỗi graph stream**. Chọn đúng **100 chỉ số train trước lọc**, seed 42,
theo hash `[adapter_version, seed, split, original_index]`;
[cohort.json](cohort.json) khóa nguyên cohort, không thay mẫu lỗi qua retry.
Quét opcode toàn stream chỉ để định vị/đếm record; materialize đúng 100 cặp,
không chuyển toàn bộ dataset. Nội dung official dev/test và biến thể khác
chưa được mở. Liệt kê metadata member trong tar ở
[inspection.json](inspection.json) không được tính là đọc nhãn test.

Reader không gọi pickle constructor/REDUCE/import dữ liệu. Grammar whitelist
được thực thi trong subprocess chroot rỗng, UID/GID 65534, environment sạch,
seccomp chặn filesystem open/network/exec/fork, hạn mức address space 512 MiB
và CPU 60 giây. Chỉ nhận/xuất qua pipe, không mount host có quyền ghi.

## Kết quả tự động

| Gate | Qua / 100 | Lỗi | Chưa đánh giá |
|---|---:|---:|---:|
| Input availability | 100 | 0 | 0 |
| Graph parsing | 100 | 0 | 0 |
| Canonical schema conversion | 100 | 0 | 0 |
| Node alignment | 100 | 0 | 0 |
| Edge alignment | 100 | 0 | 0 |
| Graph diff | 100 | 0 | 0 |
| Transaction tái dựng target | 100 | 0 | 0 |
| Codec compatibility | 100 | 0 | 0 |
| Candidate acceptance | **100** | **0** | **0** |

113 source raw links và 107 target raw links được xử lý; target gồm 214
cung canonical. Không có rejection, ambiguity, lossy conversion **so với
graph đã serialize**, no-op, duplicate-source/instruction hay vượt budget
trong cohort này. Không đồng nhất những số này với độ đúng ngữ nghĩa:
instruction accuracy, image accuracy và inter-rater agreement **chưa đánh giá**.

**31/100 cặp có correspondence khác integer ID nguồn/đích.** ID bằng nhau
không phải bằng chứng cùng entity. Matching dùng nhãn và cấu trúc, chọn
nghiệm injective cardinality tối đa, ít thay nhãn/link giữ lại nhất; hòa điểm
bị từ chối. Không có bảo đảm identity lịch sử từ code tạo dataset công khai.

| Thao tác | Số lượng |
|---|---:|
| `add_node` | 47 |
| `delete_node` | 52 |
| `replace_node` | 17 |
| `add_edge` | 94 |
| `delete_edge` | 105 |
| `replace_edge` | **0 — không có support trong pilot** |

Tổng 315 thao tác. 18 mẫu có 1 thao tác, 65 mẫu có 3, 17 mẫu có 6;
trung bình **3,15**, median 3, max 6. Không tạo thao tác thay cạnh giả để đủ
loại. Một source self-loop có sẵn trong release được giữ nguyên, không tự
chỉnh dataset; self-loop được biểu diễn bằng một cung, giải thích số delete
edge lẻ thay vì luôn gấp đôi.

| Kích thước | Min–max | Trung bình | Histogram |
|---|---|---:|---|
| Source nodes | 1–3 | 2,12 | 1:28; 2:32; 3:40 |
| Target nodes | 1–3 | 2,07 | 1:24; 2:45; 3:31 |
| Source raw links | 0–2 | 1,13 | 0:27; 1:33; 2:40 |
| Target raw links | 0–2 | 1,07 | 0:24; 1:45; 2:31 |
| Source canonical arcs | 0–4 | 2,25 | 0:27; 1:1; 2:32; 4:40 |
| Target canonical arcs | 0–4 | 2,14 | 0:24; 2:45; 4:31 |

Prepare **6,62 giây**, compile cả hai policy và loader **0,12 giây**;
không tính download, import/startup, tests hay review. Parent process peak
`ru_maxrss` **673.076 KiB (~657,3 MiB)**. Reader báo `ru_maxrss` **672.564 KiB**;
giá trị này có thể bao gồm high-water mark từ fork trước exec của process
cha đã import dependencies Task B, **không phải phép đo riêng bộ nhớ lúc
parse trong sandbox**, và không so trực tiếp với RLIMIT_AS sau exec.
Các số gốc và phạm vi ở summary; không cộng peak của hai process thành
peak đồng thời. Không đo GPU/VRAM.

## Policy và tính tương thích

File thật là `networkx.Graph` **vô hướng**, node/cạnh có nhãn `feature`;
attributes là node nối `att`. Paper nói sắp topo trên graph có hướng và code
model dùng adjacency đối xứng. Chưa có bảo đảm đủ để phục hồi mọi chiều chỉ
từ ID/thứ tự; vì vậy **A giữ topology gốc, mỗi link vô hướng thành hai cung
đối xứng**, không gọi hai cung này là ground truth chiều quan hệ trong ảnh.

B từ chối 63 mẫu có `att` vì thiếu type/owner; chỉ 37 mẫu không có `att`
chuyển được mà không cần flatten. Phân bố B trên tập đó:
19 add node / 16 delete node / 1 replace node / 38 add edge /
33 delete edge / 0 replace edge. Chưa có bằng chứng flatten `att` mà không
mất identity, nên chọn A và giữ nguyên mọi node thuộc tính.

Loader version mới đọc **100 train / 0 val**, trả đúng `Example`, codec và
`model_input()` source+instruction của Task B. Target chỉ dùng offline sinh
nhãn và chấm. Manifest ảnh của launcher MagicBrush **không tương thích**;
không tạo pixel hash giả hoặc thay official split. Chưa chuẩn bị/chạy
launcher huấn luyện cho nguồn này. Mở rộng sẽ dùng official dev làm val,
giữ test riêng, loại trùng source graph/source+instruction có báo diagnostics.
Không có image ID nên chưa chứng minh được chống trùng ảnh ngoài thông tin
graph đã phát hành.

## Review, kiểm thử và bước tiếp theo

Đã build **34 mẫu phân biệt**: 20 diagnostic trước lọc và 20 candidate chọn
độc lập, giao nhau 6. Packet hash
`66fafcca4dc889fbc53b6265cfca953f2a7c7b25cac39d9c75960b0c41b2a48c`.
Đã tự động mở đủ 34 case trên Chromium tắt GPU, mỗi case bốn panel graph,
instruction/operations/correspondence/validation, không cần ảnh, không lỗi
JavaScript. **Không tạo verdict hoặc reviewer. 0 mẫu đã được người kiểm**;
gate hiện false. Không coi kiểm thử browser là manual audit.

**259 tests CPU PASS**, gồm 42 test adapter và hồi quy editing hiện có;
pre-commit áp dụng đều pass. Kiểm cả pickle độc hại/sandbox, schema thật,
sáu loại phép, thuộc tính, ID reorder/đối xứng, semantics có/vô hướng,
transaction sai, canonical source-only IDs, leakage, hash, resume, viewer/CSV
và gate export. Fixture synthetic chỉ dùng kiểm code, không góp vào 100 hay
300 mẫu. Chạy lại pilot giữ nguyên summary artifact và mọi hash output;
chi tiết ở [validation.json](validation.json), hashes ở
[output-receipt.json](output-receipt.json).

```bash
myenv/bin/python -m tools.emnlp2020_graph_modification_v1 --stage review-serve --port 8775
```

Mở `http://127.0.0.1:8775/`, kiểm cả 34 case được chọn, ghi reviewer/năm trục,
lưu CSV thật. Gate cần diagnostic kiểm hết, candidate có ≥20 kết luận và
đúng ≥80%. Dùng CSV đó chạy `review-audit` rồi `expand` theo hướng dẫn adapter.
Chưa chạy expansion/export, không tự chạy hàng đợi hoặc tác vụ GPU. Bộ ≥300
chỉ được báo hoàn thành sau khi đủ usable train+val, dedup, budgets,
transactions và review; nguồn dataset vẫn cần giảng viên chấp thuận.
