# Level 2 — khởi động và trạng thái triển khai

Mốc 2026-11-11, sau sáu commit bàn giao Level 1 (HEAD `4d34d37`). **Mốc khởi động đã có nền dữ liệu GQA và thao tác/chấm đồ thị. Cập nhật tiếp theo: đã train/eval GQA pilot MLP/GAT, nhưng chưa hoàn thành Level 2.** Các kết quả Level 1 được giữ nguyên. Tài liệu hiện tại là Markdown, chưa tạo báo cáo PDF/slide.

Cập nhật GQA pilot: **512 train / 128 validation, seed 42, 5 epoch**, CLIP đóng băng; GC R@50 MLP **30,99%**, GAT **29,87%**. Đây là kiểm pipeline một seed, chưa phải thực nghiệm chính. [Báo cáo, protocol và lệnh demo](../reports/level2/GQA_PILOT.md).

Mốc pilot đã commit tại **`f36ff65`**. Audit tiếp theo đã đọc đủ **69.942 train / 5.000 validation**, không mở annotation test: full train phủ 310 predicate, validation phủ 267; trái/phải vẫn chiếm khoảng 89%. Đề xuất ưu tiên full train sau khi đo batch graph lớn. [Báo cáo audit quy mô](../reports/level2/GQA_SCALE_AUDIT.md). Chưa khóa/chạy thực nghiệm chính.

Benchmark tiếp theo: MLP/GAT đã chạy qua batch 64 graph lớn nhất; GAT train peak allocated **6,76 GiB**. Đề xuất khởi điểm **train batch 64 / validation batch 32**, cần xác nhận launcher full train và chất lượng học. [Benchmark và dự toán có điều kiện](../reports/level2/GQA_BENCHMARK.md). Chưa có checkpoint huấn luyện chính.

Đã chốt phương pháp 2023+: **PE-Net (CVPR 2023), bản adapted to frozen-CLIP PredCls**, giữ PL/PR và công bố các thay đổi so với code gốc. **Đã port và chạy PE-CE/PL/PLPR trên pilot GQA; chưa có main study ba seed.** [Quyết định, nguồn và thiết kế đối chứng](LEVEL2_METHOD_DECISION.md).

Cập nhật triển khai: PE-PL / PE-PLPR có GC mR@50 **0,479% / 0,441%**, đổi lại R@50 thấp hơn PE-CE; chưa đủ kết luận từ một seed. PE-PLPR qua batch 64 graph lớn nhất, peak allocated **8,97 GiB**. **296 tests PASS**; có CLI demo checkpoint thật. [Báo cáo PE-Net pilot, ablation và benchmark](../reports/level2/PENET_PILOT.md).

Đã chuẩn bị **bộ chạy full GQA 15 lượt** (MLP/GAT/PE-CE/PE-PL/PE-PLPR × 42/123/3407), cache chia shard và resume giữa epoch. Mặc định 20 epoch, batch 64/32; smoke thật và 310 tests pass. **Chưa chạy full cache/train.** [Lệnh chạy và hướng dẫn resume](LEVEL2_GQA_FULL_RUN.md). GAT+KG/VG150 mới để đợt sau theo phạm vi đã chốt.

Cập nhật **full GQA v2** theo yêu cầu: trần **200 epoch**, early stopping **patience 15 / min_delta 0**, best theo validation GC mR@50 và xuất `best.pt` ngay khi cải thiện; `last.pt` giữ đủ trạng thái resume. Đã tích hợp W&B online vào `keiranoluv-hcmus/semantic-graph-editing`, kiểm online resume và 318 tests pass. **V2 là mặc định mới; chưa chạy full.** [Hướng dẫn v2](LEVEL2_GQA_FULL_V2_RUN.md).

## 1. Đề yêu cầu gì?

Đối chiếu [PDF](../Project01_Graph_Editing.pdf), trang 3–5:

| Hạng mục bắt buộc | Trạng thái |
|---|---|
| Thực nghiệm ít nhất hai dataset có sẵn | VG150 đã có; GQA đã có **pilot MLP/GAT một seed**, chưa có thực nghiệm chính ba seed/KG |
| Triển khai/tái lập một phương pháp từ 2023 trở lại đây về SGG hoặc dự đoán quan hệ có tri thức | Đã chọn **PE-Net (CVPR 2023), bản thích nghi frozen-CLIP PredCls**; đã port + pilot GQA, chưa main study |
| Ít nhất 300 cặp `(đồ thị nguồn, câu yêu cầu, thao tác đích)` từ MagicBrush/InstructPix2Pix | Chưa xây; ví dụ tổng hợp để kiểm code không được tính vào 300 cặp |
| Kiểm tay ít nhất 20 mẫu và báo cáo độ tin cậy | Chưa có; cần người trong nhóm xác nhận, không tự đánh dấu hoàn thành |
| Huấn luyện model dự đoán thao tác, nhúng câu đóng băng | Chưa triển khai/train |
| Precision/recall/F1 từng loại thao tác; exact match cho yêu cầu đơn giản | Đã có bộ chấm cấu trúc; chưa có điểm model |
| Split theo ảnh nguồn, không rò rỉ giữa các lượt chỉnh cùng ảnh; ba seed cho thực nghiệm chính | Có hàm chia nhóm và test; chưa có manifest bộ chỉnh sửa thật |
| Demo nguồn + câu → thao tác dự đoán → đồ thị đích | Có CLI minh họa cơ chế bằng ví dụ tổng hợp; chưa phải model đọc hiểu câu |

## 2. Input/output khác Level 1 thế nào?

Level 1: đặc trưng ảnh + hộp/nhãn object → predicate cho các cặp object.

Level 2: **đồ thị nguồn + câu yêu cầu → tập thao tác chỉnh đồ thị**. Áp dụng tập thao tác hợp lệ lên đồ thị nguồn để có đồ thị đích. Đồ thị đích/thao tác GT chỉ dùng làm nhãn train hoặc để chấm, không được đưa vào input suy luận. Chưa sinh ảnh ở bước này.

Ví dụ tổng hợp:

```text
Nguồn: girl-1 --wearing--> shirt-1 [red]
Yêu cầu: Change the girl's red shirt to blue.
Thao tác: replace_node(shirt-1: shirt [red] → shirt [blue])
Đích: girl-1 --wearing--> shirt-1 [blue]
```

Sửa `girl-1` thành màu xanh thay vì `shirt-1` vẫn sai, dù cùng loại `replace_node`. Giữ cạnh `wearing` vì câu không yêu cầu thay nó. CLI hiện tạo GT bằng diff giữa hai đồ thị viết sẵn, **không dự đoán thao tác từ câu**, không chứng minh model hiểu ngôn ngữ.

Chạy: `myenv/bin/python -m tools.demo_level2_edits`.

## 3. Quy ước thao tác và chấm điểm v1

Code: [operations.py](../src/editing/operations.py), [evaluation.py](../src/editing/evaluation.py), [splits.py](../src/editing/splits.py).

| Loại | Nội dung |
|---|---|
| `add_node` | Node mới, ID mới, nhãn và tập thuộc tính |
| `delete_node` | Node cũ phải khớp đầy đủ ID/nhãn/thuộc tính |
| `replace_node` | Giữ ID, thay nhãn và/hoặc thuộc tính; đổi màu thuộc loại này |
| `add_edge` | Cạnh có ID mới và bộ subject/predicate/object |
| `delete_edge` | Xóa cạnh khớp payload cũ |
| `replace_edge` | Giữ ID cạnh, thay predicate và/hoặc endpoint |

Các thao tác được áp dụng như một giao dịch: độc lập với thứ tự liệt kê, không cho hai thao tác cùng ghi lên một thực thể, không để cạnh có endpoint đã bị xóa. Xóa node phải kèm thao tác xóa/thay các cạnh liên quan; không tự xóa ngầm rồi tính là model dự đoán được các thao tác cạnh. Nếu thao tác không hợp lệ, đồ thị nguồn vẫn nguyên vẹn.

Biểu diễn chỉnh sửa dùng tập cạnh ngữ nghĩa: cho phép nhiều predicate trên một cặp, nhưng không giữ hai cạnh trùng hệt bộ ba. Đây là quy ước **khác** GT annotation của PredCls, nơi dòng trùng vẫn được giữ khi chấm recall. Khi chuyển sang dữ liệu chỉnh sửa cần ghi rõ bước deduplicate.

**Điều kiện quan trọng:** `diff_aligned_graphs` chỉ dùng khi ID giữa source/target đã được căn chỉnh. ID detector sinh riêng cho hai ảnh không mặc nhiên trỏ tới cùng đối tượng. Chưa có thuật toán căn chỉnh ảnh/đồ thị trong mốc này; cũng chưa có matching bất biến theo hoán vị ID của node mới. Cần chốt quy tắc alignment/canonical ID trước khi sinh nhãn MagicBrush và train model.

Cách chấm từng loại thao tác:

- TP: đúng loại **và toàn bộ payload**, gồm ID đối tượng/cạnh, trạng thái trước và sau. Chỉ đúng nhãn loại thao tác chưa đủ.
- FP: thao tác dư/sai; dự đoán lặp một thao tác đúng lần thứ hai cũng là một FP.
- FN: thao tác GT bị thiếu.
- Precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = 2TP/(2TP+FP+FN).
- Loại hoàn toàn vắng cả GT lẫn prediction được ghi `null`; nếu có lỗi nhưng không có hit thì điểm bằng 0. Macro F1 chỉ trên các loại có GT support; micro gộp TP/FP/FN của mọi loại, kể cả FP ở loại không có GT.
- Exact operation-set accuracy yêu cầu đúng toàn bộ thao tác, không dư/thiếu/lặp. Chấm thêm tỷ lệ tập thao tác thực thi hợp lệ và tỷ lệ đồ thị đích khớp hoàn toàn. Giao dịch sai vẫn có thể chứa một số thao tác đúng, nên operation recall và transaction validity được báo riêng.

Ví dụ: GT gồm hai thao tác A và B. Dự đoán A và C thì TP=1, FP=1, FN=1 → P=R=F1=50%, exact match=0. Đây là số minh họa, không phải kết quả học được.

## 4. GQA đã chuẩn bị

Code [adapter](../src/datasets/gqa.py), [lệnh chuẩn bị](../tools/prepare_level2_gqa.py), [report kiểm dữ liệu thật](../reports/level2/gqa-preparation-v1.json).

| Split dự án | Nguồn | Số ảnh | Object sau xử lý | Relation annotation sau xử lý |
|---|---|---:|---:|---:|
| Train | Official train trừ validation nội bộ | 69.942 | 1.148.828 | 3.541.725 |
| Validation | 5.000 ảnh từ official train | 5.000 | 82.121 | 251.006 |
| Test dự án | Official val của GQA | 10.696 | 174.241 | 534.437 |

Chia bằng thứ tự SHA-256 của `42:image_id`, không phụ thuộc thứ tự JSON. Vocabulary từ **train dự án**, không lấy từ toàn bộ official train trước khi tách validation: **1.699 object labels, 310 predicates, 617 attributes**, ID 0 để dành. Giữ nguyên chuỗi và số nhiều; đây là vocabulary tùy chỉnh đầy đủ, **không phải GQA200/GQA100**, không so trực tiếp với bảng paper dùng taxonomy khác.

Đã chuẩn hóa toàn bộ 85.638 graph: clip box hữu hạn có diện tích dương vào ảnh, bỏ box vẫn không hợp lệ, bỏ object ngoài vocabulary và quan hệ liên quan; bỏ self-relation vì candidate là N(N−1); giữ dòng quan hệ trùng và đa predicate. Report ghi từng nguyên nhân. Sau clipping có thêm box suy biến nên số bị loại có thể cao hơn lần audit chỉ đếm w/h không dương.

Validation có 25 object ngoài vocabulary bị loại; test dự án có 3. Support đã bị lọc cần công bố khi báo metric. Có 2.208/153/398 ảnh không còn relation trong train/val/test. Split vẫn giữ các ảnh này; launcher train/eval sau này phải xử lý và báo rõ số ảnh thực sự đóng góp, không âm thầm coi mọi ảnh đều có GT. Vocab class 1.699 của GQA không dùng chung bảng KG/embedding 150 lớp VG150.

ZIP ảnh được đọc trực tiếp, không cần giải nén thêm 22 GB. Đã xác nhận mọi ID trong ba split có JPEG trong archive; đã decode ảnh train ID `10` và build đồ thị PyG: **29 node, 812 cặp ứng viên, 172 GT annotation**, geometry 5 chiều. Đây là kiểm kết nối adapter, chưa phải cache CLIP hay forward model.

```bash
myenv/bin/python -m tools.prepare_level2_gqa
myenv/bin/python -m tools.demo_level2_edits
myenv/bin/python -m pytest -q
```

Manifest đầy đủ nằm tại `data/processed/level2/gqa-v1/manifest.json` (không commit dữ liệu sinh trong `data/`); report lưu hash và code tái tạo. Tool từ chối ghi đè khác nội dung; thay protocol/source thì dùng đường dẫn manifest/report mới. Dữ liệu GQA đã có nguồn và hash ở [ghi chép tải dữ liệu](datasets/LEVEL2_GQA_PSG.md).

GQA dùng ảnh Visual Genome: hai thực nghiệm riêng không chứng minh chuyển miền độc lập. Nếu trộn dữ liệu/pretrain chéo, phải audit trùng ảnh gốc giữa các dataset trước. Chưa chạy model trên test; lượt này chỉ audit annotation cấu trúc.

## 5. Thứ tự triển khai tiếp

1. **Mở rộng sau GQA pilot:** cache CLIP và launcher MLP/GAT đã chạy được trên subset cố định; tiếp theo audit quy mô/độ phủ lớp, cache cho cohort chính, mapping KG riêng, GAT+KG và ba seed. Evaluator cần tham số hóa taxonomy 310 predicate; không đổi evaluator VG150 đã khóa, không mặc định 50 lớp.
2. **PE-Net adapted (CVPR 2023):** đã có GloVe mapping có provenance, prototype head + PL/PR, test reference, pilot CE/PL/PLPR và benchmark riêng. Tiếp theo khóa launcher/recipe chính và chạy ba seed; VG150 cần đối chứng cùng supervision mới. Dùng chung frozen CLIP và evaluator; công bố pair-feature proxy, đa nhãn và mọi thay đổi. [Chi tiết quyết định](LEVEL2_METHOD_DECISION.md). EGTR/LLM4SGG đã được đối chiếu nhưng không chọn cho pipeline hiện tại.
3. **MagicBrush:** chọn subset train/dev có provenance, ít nhất 300 cặp; sinh graph nguồn/đích bằng model tiền huấn luyện hoặc caption; căn chỉnh instance trước khi diff. Giữ mọi turn của cùng `img_id`/ảnh gốc trong một split. Có thể dùng train để tạo validation nội bộ và giữ dev làm test dự án; nếu làm vậy phải gọi đúng là split dự án. Không đưa official test vào train.
4. Khóa bộ nhãn sinh tự động, lấy ít nhất 20 mẫu ngẫu nhiên để nhóm kiểm; báo tỷ lệ đúng/mức agreement và lỗi alignment/thuộc tính. Không sửa riêng nhãn test dựa trên review. Những câu sửa style/toàn cảnh không biểu diễn được bằng schema phải được thống kê, không âm thầm ép thành no-op.
5. Train baseline và mô hình graph + embedding câu frozen; đánh giá ba seed bằng P/R/F1 từng loại, exact match trên tập yêu cầu đơn giản được xác định trước và validity. Phân tách đánh giá trên graph GT/pseudo-GT với graph model dự đoán để không che lỗi đầu vào.

Nguồn đã kiểm tra:

- [MagicBrush chính thức](https://github.com/OSU-NLP-Group/MagicBrush): dữ liệu chỉnh sửa đơn/multi-turn; [Hugging Face](https://huggingface.co/datasets/osunlp/MagicBrush), revision quan sát `6c254560879df7851233ea0fd2ae2a62764d0ef2`. Chưa tải subset hoặc sinh nhãn ở lượt này.
- [EGTR — code chính thức](https://github.com/naver-ai/egtr).
- [LLM4SGG — CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Kim_LLM4SGG_Large_Language_Models_for_Weakly_Supervised_Scene_Graph_Generation_CVPR_2024_paper.html), [code chính thức](https://github.com/rlqja1107/torch-LLM4SGG).

Mốc khởi động bên trên bổ sung code nền và kiểm dữ liệu; report foundation vẫn được giữ nguyên như provenance lịch sử. Mốc pilot tiếp theo đã có điểm MLP/GAT GQA, vẫn chưa có 300 cặp đã hoàn thiện hoặc human review. [Biên bản kiểm chứng](../reports/level2/foundation-verification.json).

Kiểm chứng mốc khởi động: **260 tests PASS** (21 ca mới), mọi hook pre-commit áp dụng đều pass. Chạy lại chuẩn bị GQA giữ nguyên manifest/report; kiểm độc lập cân bằng số object/relation trước và sau xử lý ở cả ba split. Loader thật đã đọc thêm ảnh validation ID `1` với 26 node, 238 GT annotation.
