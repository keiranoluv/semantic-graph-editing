> **Trạng thái: CPU v2 đã dừng theo yêu cầu người dùng.** Chỉ MLP seed 42
> hoàn tất (best epoch 6, val micro-F1 79,55%, exact 64,85%); MLP seed 43 bị
> SIGINT, chưa được tính là kết quả. Không có đủ ba seed hoặc đánh giá test
> của phiên bản CPU này. Thực nghiệm đầy đủ chuyển sang
> [GPU v3](../emnlp2020-task-b-gpu-v3/README.md), huấn luyện lại từ đầu và không
> trộn kết quả CPU vào thống kê GPU. Dữ liệu/cache CPU vẫn dùng lại theo hash.

# Task B – Nhiệm vụ 2: EMNLP 2020 crowdsourced, v2

Ngày theo múi giờ người dùng: 15/11/2026. Phạm vi: **source graph + instruction →
edit operations → target graph**, chỉ CPU. Mã triển khai/recipe ban đầu:
`dd5c8b1`; full source hashes và commit chạy nằm trong `provenance.json`.
Hướng dẫn tái lập: [protocol v2](../../../docs/LEVEL2_EMNLP2020_TASK_B_V2.md).

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

Train có 53 ambiguous correspondence, 7 no-op và 123 source+instruction lặp.
Dev có 5 conversion rejection trước lọc, trong đó 2 đã nằm trong 465 mẫu
source overlap train; tổng loại 468. Không chuyển mẫu qua split, không bù mẫu
đã loại, không chọn target theo metric. Fingerprint dùng source, bất biến với
ID/thứ tự; source của cả các mẫu reject vẫn được reserve để kiểm overlap.

Đây là **subset source-disjoint của split gốc**, không phải số đo trên toàn
benchmark gốc. Trong toàn train/dev có 13 nhóm source+instruction giống nhau
nhưng target không đẳng cấu; 11 nhóm xuất hiện ngay trong train. Dedup giữ
lần đầu theo index gốc, không sửa hoặc chọn nhãn thuận lợi. Audit target chỉ
phục vụ mô tả, không tham gia grouping, vocabulary, feature hay selection.

Tất cả accepted label đều qua kiểm raw topology, unique alignment,
`apply_operations(source, GT)==target` và codec round-trip. Reindexed
correspondence xuất hiện ở 6.957 train và 153 val accepted pairs: raw ID bằng
nhau không đủ để đồng nhất entity. Label là ground truth **suy ra có điều kiện
theo correspondence và representation của graph release**, không phải nhận
định đúng với ảnh hoặc identity lịch sử được tác giả cấp sẵn.

| Thao tác | Train | Val |
|---|---:|---:|
| add_node | 13.712 | 93 |
| delete_node | 13.706 | 357 |
| replace_node | 6.918 | 139 |
| add_edge | 27.444 | 188 |
| delete_edge | 27.489 | 715 |
| replace_edge | 140 | 8 |

Pilot v1 không có replace_edge; full train thực sự có support. 140 cung thay
cạnh train thuộc 70 cặp, mỗi link vô hướng được tính hai cung. Không tạo
support synthetic để đủ sáu loại.

## Giao thức mô hình

CLIP ViT-B/32 text đóng băng, cache 34.367 source/instruction texts, 512 chiều,
không text nào bị truncate. Không trích ảnh. Hai encoder dùng cùng đặc trưng,
decoder, vocabulary, split và ngân sách: MLP không message passing, GNN hai
message layers; hidden 96, GRU, AdamW lr 0,001, batch 128, 12 epoch,
seeds 42/43/44, CPU 4 threads, một model/seed mỗi thời điểm.

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
