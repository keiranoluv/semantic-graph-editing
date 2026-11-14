# Task B – Nhiệm vụ 2: EMNLP 2020 crowdsourced, GPU v3

Ngày theo múi giờ người dùng: 15/11/2026. Phạm vi: **source graph + instruction →
edit operations → target graph**. Train/evaluate trên RTX 5090, CLIP text cache
đóng băng đã chuẩn bị trên CPU. Code GPU: `bf985cb`, mở rộng từ `dd5c8b1`;
full source hashes và commit chạy nằm trong `provenance.json`.
Hướng dẫn tái lập: [protocol GPU v3](../../../docs/LEVEL2_EMNLP2020_TASK_B_GPU_V3.md).

## Nguồn, dữ liệu và kiểm định

Tận dụng adapter/pilot 100 train đã có trên `main`: parser pickle không thực
thi trong chroot/seccomp, compiler, correspondence, graph diff, canonical
identity, UTF-8 action codec, evaluator và MLP/GNN+GRU. V2 thêm cách đọc toàn
split trong cùng sandbox, loader tách trainval/test, codec field để rút ngắn
chuỗi, frozen CLIP text CPU và huấn luyện/đánh giá có provenance. Không sửa
`src/`, pilot v1, MagicBrush, hoặc notebook cá nhân.

Nguồn chính thức: [He et al., Findings EMNLP 2020](https://aclanthology.org/2020.findings-emnlp.87/),
[code/data release](https://github.com/xlhex/SceneGraphModification), revision
`6e1b396007a0c938bccda2bbddade332745b3f42`. Archive SHA256:
`6e13b096f2756b638f59e38b13b80658f91f53cbfa6750a210929fa8a091d337`.
SHA được pilot tính từ download chính thức, không phải chữ ký của tác giả.
Raw, converted records, features, checkpoint và predictions chỉ giữ cục bộ;
metadata/statistics trong báo cáo không phân phối dataset chưa rõ giấy phép.

| Split | Gốc | Dùng được | Cách chọn |
|---|---:|---:|---|
| train | 30.051 | 29.868 | official train, giữ lần đầu của input trùng |
| val | 1.000 | 532 | official dev, loại source overlap train |
| test | 1.000 | 530 | official test, loại source overlap train/val đã khóa |

Train có 53 ambiguous correspondence, 7 no-op và 123 source+instruction lặp.
Dev có 5 conversion rejection trước lọc, trong đó 2 đã nằm trong 465 mẫu
source overlap train; tổng loại 468. Không chuyển mẫu qua split, không bù mẫu
đã loại, không chọn target theo metric. Fingerprint dùng source, bất biến với
ID/thứ tự; source của cả các mẫu reject vẫn được reserve để kiểm overlap.
Test loại 468 source overlap và 2 ambiguous correspondence; giữ nguyên mọi
index gốc và lý do trong local artifact, không điều chỉnh train/val từ test.

Đây là **subset source-disjoint của split gốc**, không phải số đo trên toàn
benchmark gốc. Trong toàn train/dev có 13 nhóm source+instruction giống nhau
nhưng target không đẳng cấu; 11 nhóm xuất hiện ngay trong train. Dedup giữ
lần đầu theo index gốc, không sửa hoặc chọn nhãn thuận lợi. Audit target chỉ
phục vụ mô tả, không tham gia grouping, vocabulary, feature hay selection.

Tất cả accepted label đều qua kiểm raw topology, unique alignment,
`apply_operations(source, GT)==target` và codec round-trip. Reindexed
correspondence xuất hiện ở 6.957 train, 153 val và 181 test accepted pairs: raw ID bằng
nhau không đủ để đồng nhất entity. Label là ground truth **suy ra có điều kiện
theo correspondence và representation của graph release**, không phải nhận
định đúng với ảnh hoặc identity lịch sử được tác giả cấp sẵn.

| Thao tác | Train | Val | Test |
|---|---:|---:|---:|
| add_node | 13.712 | 93 | 105 |
| delete_node | 13.706 | 357 | 363 |
| replace_node | 6.918 | 139 | 121 |
| add_edge | 27.444 | 188 | 210 |
| delete_edge | 27.489 | 715 | 728 |
| replace_edge | 140 | 8 | 6 |

Pilot v1 không có replace_edge; full train thực sự có support. 140 cung thay
cạnh train thuộc 70 cặp, mỗi link vô hướng được tính hai cung. Không tạo
support synthetic để đủ sáu loại.

## Giao thức mô hình

CLIP ViT-B/32 text đóng băng, cache 34.367 source/instruction texts, 512 chiều,
không text nào bị truncate. Không trích ảnh. Hai encoder dùng cùng đặc trưng,
decoder, vocabulary, split và ngân sách: MLP không message passing, GNN hai
message layers; hidden 96, GRU, AdamW lr 0,001, batch 128, 12 epoch,
seeds 42/43/44, GPU FP32, TF32 tắt, CPU 4 threads, một model/seed mỗi thời điểm.

Codec dự đoán field theo arity của sáu kind, rồi dựng JSON và dùng evaluator
UTF-8 cũ. Không ràng buộc/rewrite prediction bằng target. Vocabulary 4.762 token
kể cả special tokens, học từ train cộng các token schema. Trên val, 40/532 GT
program có OOV; vẫn giữ tất cả trong mẫu số. Upper bound exact match do closed
vocabulary trên val là 492/532 = 92,48%, không phải accuracy đạt được. Yêu cầu
đơn giản dùng proxy đúng một GT operation: 140 mẫu val.

Chọn checkpoint bằng minimum validation teacher-forced loss. Sau đủ sáu lượt,
khóa SHA checkpoint/result; chỉ sau đó mở official test. Không chọn epoch,
seed, vocabulary hoặc cấu hình bằng test. Phân tích 5 đúng/5 sai chọn bằng hash
ID cố định, dùng seed 42 khai báo trước, không chọn seed tốt nhất trên test.

## Kết quả GPU trên test sạch

Trung bình ± **sample standard deviation** qua ba seed; các giá trị dưới đây
là phần trăm. Cùng 530 mẫu cho mọi mô hình và baseline. Exact target graph
bằng exact operation set trong lần đo này. Cột simple chỉ có **121 mẫu**, đều
thuộc thay nhãn node trong test sạch; không so trực tiếp với accuracy toàn tập.

| Model | Micro F1 | Macro F1, 6 loại | Exact operations/target | Simple exact | Valid transaction |
|---|---:|---:|---:|---:|---:|
| MLP | 81.69 ± 0.31 | 54.25 ± 0.51 | 67.74 ± 0.68 | 25.34 ± 1.91 | 95.60 ± 0.29 |
| GNN | 82.44 ± 0.39 | 55.03 ± 0.28 | 68.55 ± 0.22 | 24.79 ± 1.65 | 95.79 ± 0.39 |

| Thao tác | Support test | MLP P / R / F1 | GNN P / R / F1 |
|---|---:|---:|---:|
| add_node | 105 | 58.53 / 60.95 / 59.72 | 62.50 / 65.40 / 63.86 |
| delete_node | 363 | 97.33 / 96.88 / 97.10 | 97.53 / 97.98 / 97.76 |
| replace_node | 121 | 30.57 / 25.34 / 27.71 | 30.59 / 24.79 / 27.38 |
| add_edge | 210 | 42.99 / 44.76 / 43.86 | 42.64 / 44.60 / 43.56 |
| delete_edge | 728 | 97.46 / 96.75 / 97.10 | 97.53 / 97.76 / 97.64 |
| replace_edge | 6 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 |

Full precision/recall/F1, TP/FP/FN, per-seed values và độ lệch chuẩn từng loại
nằm trong [metrics.json](metrics.json). Macro chỉ khoảng 54–55% so với micro
82% cho thấy ảnh hưởng của deletion chiếm đa số. GNN hơn MLP 0,75 điểm micro-F1
trên test, nhưng validation MLP nhỉnh hơn; ba seed và graph rất nhỏ chưa đủ để
kết luận GNN tổng quát tốt hơn.

Baseline no-op: F1=0, exact=0, valid=100%. Baseline nearest training program
(CLIP instruction + pooled source cosine, không dùng target test) đạt
micro-F1 **56,22%**, exact **33,77%**, valid **67,55%**. Chỉ số chính tập trung vào các thao tác thay đổi; node/edge F1 trong paper
thuộc giao thức khác, không dùng để so sánh trực tiếp.

35/530 test programs có payload OOV; giữ đủ 530 mẫu. Closed-vocabulary upper
bound exact là 495/530 = **93,40%**, không phải điểm mô hình. Phần lớn lỗi không
thể giải thích chỉ bằng OOV: cả năm ca sai được chọn dưới đây đều không OOV.

`replace_edge` có GT support 6 cung, nhưng TP=0 ở cả ba seed của cả hai model.
Ở GNN/42, một trong ba cặp dự đoán đúng loại thay cạnh nhưng sai payload; hai
cặp còn lại dự đoán xóa node/cạnh. Vì vậy kết quả là **F1=0**, không phải
undefined hoặc không có support. Không tạo nhãn giả, đổi taxonomy hay bỏ các
mẫu này để cải thiện điểm.

## Ablation validation và cách diễn giải

| Model/đầu vào | Micro F1 (%) | Exact (%) |
|---|---:|---:|
| MLP đầy đủ | 80.39 | 66.04 |
| MLP shuffled_instruction | 66.34 | 50.19 |
| GNN đầy đủ | 80.25 | 65.85 |
| GNN removed_edges | 9.15 | 5.76 |
| GNN shuffled_instruction | 67.56 | 51.69 |

Xáo instruction làm giảm khoảng 13–14 điểm F1, nhưng baseline sau xáo vẫn cao:
source/prior/đa số deletion đóng vai trò lớn. Bỏ edge mask của GNN cũng bỏ
thông tin predicate/endpoint trong memory, tạo input ngoài phân bố; đây là
**diagnostic tại inference**, không cô lập riêng message passing và không thay
thế retrain ablation. So sánh MLP/GNN đầy đủ mới giữ cùng thông tin cạnh.

## Phân tích năm đúng và năm sai

GNN/42 được khai báo trước, không chọn seed tốt nhất theo test. Trong 530 mẫu:
364 đúng, 23 parse/schema failure, 1 transaction failure, 34 sai entity identity,
19 sai tập loại thao tác, 89 sai payload. Đây là taxonomy tự động theo GT đã
chuyển đổi, không phải human review hay kết luận đúng với ảnh. Lựa chọn case
bằng hash ID sau chia nhóm đúng/sai, không thay bằng ví dụ đẹp hơn.

| ID test (đuôi) | Kết quả | Phân tích |
|---|---|---|
| 000037 | Đúng | Thêm đúng một node và hai cung của raw link mới; giữ nguyên source. |
| 000809 | Đúng | Thêm đúng node và predicate theo graph release; diễn đạt ngôn ngữ không cần trùng token predicate. Đây là đúng với GT graph, chưa phải kiểm ngữ nghĩa độc lập. |
| 000045 | Đúng | Xóa đúng node được nhắc tới và cả hai cung incident, không để dangling endpoint. |
| 000324 | Đúng | Xóa đúng node ngữ cảnh/cặp cung, bảo toàn phần còn lại gồm liên kết thuộc tính. |
| 000126 | Đúng | Đổi đúng nhãn node màu, giữ identity/topology. Thuộc tính được biểu diễn bằng node, không phải trường attributes. |
| 000115 | Sai | Chọn đúng node để thay nhưng chọn một nhãn liên quan thay vì nhãn đích: lỗi lexical payload, không phải OOV. |
| 000041 | Sai | GT khác nhau ở số ít/số nhiều; model phát lại nhãn nguồn trong replace, vi phạm điều kiện before≠after. Parser/evaluator từ chối, không tự đổi thành no-op đúng. |
| 000372 | Sai | Nhãn mới đúng nhưng gắn vào node sở hữu thay vì node cần thay; transaction vẫn hợp lệ. Exact payload/entity cần thiết để bắt lỗi grounding này. |
| 000995 | Sai | Với hai nhãn loại trang phục có quan hệ gần, model phát replace không thay nội dung nguồn; schema/precondition failure. |
| 000375 | Sai | Yêu cầu đổi màu nhưng replace giữ nguyên màu cũ; không được tính là đúng dù topology không đổi. |

ID đầy đủ thêm prefix `emnlp2020_crowd_test_`. [error-analysis.json](error-analysis.json)
giữ ID, taxonomy và operation-kind counts cho cả MLP/GNN; full instruction,
graphs, operations và correspondence chỉ nằm cục bộ trong dữ liệu/predictions.

## Tái lập, demo và trạng thái CPU

CPU v2 dừng bằng SIGINT theo yêu cầu, giữ MLP/42 hoàn tất và MLP/43 dang dở;
không trộn kết quả đó vào ba seed GPU. GPU huấn luyện lại từ đầu. Best epochs:
MLP 6/8/8; GNN 8/7/6. Mỗi lượt kể cả đánh giá val/ablation khoảng 11–19 giây;
không coi đây là benchmark hệ thống được kiểm soát.

GPU train/test source được commit **trước** khi chạy (`bf985cb`). Các kiểm CPU
EMNLP: 53 pass, 1 CUDA skip; kiểm CPU/CUDA riêng: 2 pass. Toàn bộ editing:
408 pass, 1 skip qua CPU test harness. Lệnh nguyên bản còn 13 lỗi legacy do
mock CUDA availability; không sửa code/tests MagicBrush để che lỗi. Chi tiết
và command trong [validation.json](validation.json).

Đã chạy CLI inference với **chỉ source_graph + instruction** của test index
000001 (mẫu usable đầu tiên theo index, không chọn theo metric), batch 1,
GNN/42, GPU. Output trùng program của evaluator, transaction hợp lệ nhưng sai
GT; demo chứng minh đường đi input → prediction → reconstructed graph hoạt
động, không minh họa accuracy hoàn hảo. Artifact cục bộ:
`data/processed/level2/emnlp2020-task-b-v2/gpu-demo-{input,prediction}.json`.
Không sửa notebook cá nhân. Không có tiến trình CPU/GPU hoặc hàng đợi nào cần
tiếp tục sau bàn giao. [artifact-receipt.json](artifact-receipt.json) khóa
source/data/checkpoint/predictions để kiểm lại bằng lệnh sau mà không chạy GPU:

```bash
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m tools.verify_emnlp2020_task_b_gpu_v3
```

Runtime/source giữ nguyên commit `bf985cb`. `experiment.json` ghi commit và
bất biến; muốn train lại từ HEAD khác cần output version mới. Verifier dùng
source hashes, không cần đổi HEAD về commit train để kiểm lại kết quả đã có.

## Những giới hạn cần đưa vào báo cáo môn học

1. **Representation:** graph phát hành vô hướng, không phải scene graph có hướng
   đã xác minh với ảnh. Mỗi link thành hai cung; self-loop giữ một cung. `att`
   nối node thuộc tính nhưng không phát hành object/attribute type hoặc owner.
   Không suy chiều từ ID và không flatten thuộc tính bằng heuristic. Thay màu
   có thể là thay nhãn node; metric cạnh ở đây chấm cung, không chấm raw link.
2. **Correspondence:** matching cấu trúc tối ưu duy nhất vẫn là giả định, không
   chứng minh identity lịch sử. Ambiguous matches bị loại có báo lý do; không
   dùng oracle correspondence tại inference.
3. **Graph nhỏ:** toàn source train/dev có 1–3 node. Lợi ích GNN trên graph nhỏ
   không chứng minh khả năng multi-hop hoặc mở rộng sang cảnh nhiều object.
4. **Support lệch:** replace_edge có support rất ít; lọc source overlap làm val
   nghiêng về deletion. F1 cao ở delete không đủ kết luận về insertion, thay
   nhãn mở hoặc semantic understanding.
5. **Nguồn không có image ID:** source-disjoint graph giảm trùng cấu trúc nhưng
   không chứng minh image-disjoint; có thể loại cả ảnh khác nhau có graph giống
   nhau. Không so trực tiếp với node/edge F1 hay graph accuracy trong paper.
6. **Vocabulary và decoding:** decoder closed vocabulary có trần OOV. Không có
   copy từ chuỗi instruction cho nhãn mới. Chuỗi có thể sai schema/transaction;
   valid rate và parse error cần đọc cùng F1. Không bỏ mẫu khó để tăng điểm.
7. **Kiểm tay:** chưa có human review EMNLP mới; semantic accuracy/agreement là
   null. Test cấu trúc và reconstruction không thay thế kiểm tay ≥20 mẫu của đề.
   Không chuyển review MagicBrush sang EMNLP hoặc tự tạo người kiểm.

## Vì sao chuyển hướng và phần đề gốc còn phải ghi nhận

Đề gốc trang 3–5 yêu cầu bộ ≥300 cặp từ MagicBrush/InstructPix2Pix, kiểm tay ≥20,
CLIP/Sentence-BERT đóng băng và P/R/F1 sáu thao tác. EMNLP giúp tách bài học
**dự đoán edit từ graph + ngôn ngữ** khỏi nhiễu scene graph extraction, nhưng
không tự thay thế yêu cầu nguồn hoặc kiểm tay. Cần trình bày thay nguồn với
người hướng dẫn; không ghi cả cấp độ 2 hoàn thành chỉ từ thực nghiệm này.

Bằng chứng cũ được giữ nguyên: [audit grounded pilot](../magicbrush-grounded-failure-audit-v1/README.md)
đo 0/20 mẫu tạo được nhãn do grounding/matching/ngưỡng appearance, và phân biệt
abstention với no-op. [Bản manual 602](../magicbrush-manual-full602-v1/README.md)
về sau có 437 accepted pairs, 20 train được người dùng xác nhận, nhưng coverage
và semantic representability vẫn giới hạn. [Recipe v3](../magicbrush-manual-experiment-v3/README.md)
báo closed-vocabulary chỉ biểu diễn đầy đủ 15/67 val programs và không có
replace_edge val/test. Các số này thuộc phiên bản và thời điểm riêng, không
suy thành thất bại của mọi phương pháp MagicBrush hoặc metric so sánh chéo.
Không chạy lại hay sửa bất kỳ thực nghiệm MagicBrush nào trong công việc này.

Bài học phương pháp: reconstruction kiểm tính tự nhất quán của nhãn; cần thêm
provenance, correspondence, support, OOV và đánh giá ngữ nghĩa độc lập mới biết
metric có trả lời đúng câu hỏi nghiên cứu hay không.
