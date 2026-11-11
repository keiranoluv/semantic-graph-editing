# Level 2 — chốt phương pháp 2023+: PE-Net

Ngày quyết định: **2026-11-12**, Asia/Bangkok. Chọn **Prototype-Based Embedding
Network for Scene Graph Generation (PE-Net), CVPR 2023**, của Chaofan Zheng và
cộng sự. Tên triển khai trong dự án: **PE-Net adapted to frozen-CLIP PredCls**.
Trạng thái: **đã chọn và đối chiếu paper/code; chưa triển khai/train phương pháp**.
[Paper tại CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Zheng_Prototype-Based_Embedding_Network_for_Scene_Graph_Generation_CVPR_2023_paper.html),
[bản tác giả đầy đủ](https://arxiv.org/html/2303.07096v1),
[code chính thức đã pin](https://github.com/VL-Group/PENET/tree/9c9f50777c66647799cb7a17dfb855aa98aecd1e).

Đây là quyết định **triển khai phương pháp có điều chỉnh cho pipeline chung**, không
phải cam kết tái tạo đúng điểm paper. Các thay đổi ở mục 3 phải hiện trong báo cáo
cuối; không rút gọn tên thành “PE-Net reproduced” hoặc gắn điểm của nhóm vào bảng
paper. Đề cho phép triển khai hoặc tái lập một phương pháp từ 2023, không bắt buộc
chọn đúng hai ví dụ EGTR/LLM4SGG.

## 1. Vì sao chọn

PE-Net biểu diễn quan hệ và prototype của từng predicate trong không gian chung,
đồng thời dùng Prototype-guided Learning (PL) và Prototype Regularization (PR).
Code có nhánh PredCls rõ ràng. Launcher gốc đóng băng RPN, backbone và box head;
đây là cơ sở phù hợp hơn để tách phần relation học được khỏi thị giác đóng băng.
[Kiến trúc và loss](https://github.com/VL-Group/PENET/blob/9c9f50777c66647799cb7a17dfb855aa98aecd1e/maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py#L29),
[launcher gốc](https://github.com/VL-Group/PENET/blob/9c9f50777c66647799cb7a17dfb855aa98aecd1e/tools/relation_train_net.py#L61).

Lý do lựa chọn của dự án: GQA hiện có tới 42 predicate dưới 20 GT ngay cả khi dùng
full train, trong khi trái/phải chiếm khoảng 89% GT. Mô hình prototype là hướng
có cơ chế cụ thể để kiểm tra khả năng phân biệt lớp, và có ablation tách được tác
động của các loss. Đây là **giả thuyết cần thực nghiệm**, không phải dự đoán chắc
chắn rằng PE-Net sẽ tăng mR trên GQA custom. [Audit của dự án](../reports/level2/GQA_SCALE_AUDIT.md).

| Ứng viên | Đối chiếu nguồn chính thức | Quyết định cho pipeline hiện tại |
|---|---|---|
| **PE-Net, CVPR 2023** | Có code PredCls, prototype head, PL/PR | **Chọn**, công bố thay đổi feature, đa nhãn và evaluator |
| EGTR, CVPR 2024 | Trích quan hệ từ attention của DETR; relation smoothing gắn với chất lượng detection và học đa nhiệm | Không chọn cho lượt này: cache CLIP object không có các tensor/query DETR cần thiết |
| LLM4SGG, CVPR 2024 | Dùng LLM trích/alignment triplet từ caption, rồi grounding và huấn luyện weakly supervised | Không chọn cho lượt này: trọng tâm là quy trình tạo supervision, cần thêm caption/grounding và audit rò rỉ |

Hai nhận định loại trừ là đánh giá mức phù hợp của nhóm, không phải khẳng định hai
phương pháp kia không thể điều chỉnh. [EGTR chính thức](https://github.com/naver-ai/egtr),
[LLM4SGG chính thức](https://github.com/rlqja1107/torch-LLM4SGG).
Không lấy SGDet hoặc weak-supervision paper score để so trực tiếp với PredCls của nhóm.

## 2. Phần cốt lõi phải giữ

Chuẩn code: repo `VL-Group/PENET`, commit
`9c9f50777c66647799cb7a17dfb855aa98aecd1e`. Không import toàn bộ stack detector cũ;
port relation module cần thiết sang môi trường torch hiện tại.

- Prototype object/predicate từ embedding tên lớp; giữ phép chiếu riêng cho vai
  subject và object, gated visual residual và phép fusion của tác giả.
- Relation representation đối chiếu với toàn bộ predicate prototypes bằng cosine
  có temperature học được; background vẫn có prototype ở ID 0.
- **PL:** cosine classification loss và Euclidean margin giữa relation với
  prototype dương so với các prototype âm gần nhất.
- **PR:** regularization cosine giữa prototype và Euclidean separation giữa
  các prototype. Không chỉ thay linear classifier bằng cosine rồi gọi là PE-Net.
- Giữ cấu trúc residual/norm/shared projection của relation head khi port, không
  tự ghép GAT rồi gọi hybrid đó là phương pháp gốc.

Chi tiết chuẩn để triển khai được đối chiếu trực tiếp tại
[PrototypeEmbeddingNetwork](https://github.com/VL-Group/PENET/blob/9c9f50777c66647799cb7a17dfb855aa98aecd1e/maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py#L29).
PE-Net trong dự án là một model dự đoán quan hệ độc lập cạnh các baseline. Nó
**không thay thế GAT+KG**, cũng không thực hiện graph editing từ câu yêu cầu.

## 3. Những điều chỉnh đã chọn và phải công bố

| Thành phần | Bản thích nghi của dự án |
|---|---|
| Feature object | Cùng cache CLIP ViT-B/32 đóng băng + geometry 5D như MLP/GAT; thay ROI feature của detector gốc |
| Feature union-box | **Không có crop union mới.** Dùng MLP trên concat có thứ tự `(x_subject, x_object)` làm đại diện pair visual; gọi là **pair-feature proxy**, không gọi là union feature thật |
| Text khởi tạo | Giữ hướng GloVe 6B 300D như code gốc; object/predicate embedding được học sau khởi tạo. Chưa tải/pin asset GloVe trong lượt quyết định này |
| Kích thước nội bộ | Bắt đầu adapter với semantic width 256 và projection 512 để phù hợp ngân sách, thay 2048/4096 trong code gốc; phải báo số tham số và benchmark riêng |
| Số lớp | Tham số hóa `C=51` cho VG150, `C=311` cho GQA; không giữ các hằng 51 trong loss/tensor |
| GT nhiều predicate | Dùng tập predicate dương của pair, soft CE đồng đều như GQA pilot; không ép thành một nhãn hoặc dùng predicate đúng thứ hai làm negative |
| GT trùng | Một bộ ba trùng không tăng trọng số train; evaluator vẫn giữ mọi dòng GT theo protocol dự án |
| Ranking/evaluation | Dùng evaluator GC/NG chung đã khóa; **không bật relation NMS riêng của repo PE-Net** |
| Optimizer và số epoch | Khóa sau pilot adapter; không dùng recipe SGD 60k iteration của tác giả rồi gọi là cùng ngân sách với baseline một cách tự động |

Pair-feature proxy là thay đổi kiến trúc có ý nghĩa, không tương đương crop vùng
bao hai object. Vì vậy kết luận thực nghiệm chỉ áp dụng cho **bản adapted** này;
nếu muốn claim tái lập nguyên bản, phải mở thí nghiệm riêng với feature gốc/union
thật và các baseline có cùng đầu vào. Cách chọn hiện tại giữ nguyên lượng thông
tin ảnh đầu vào giữa các model và không tạo thêm cache N(N−1) crop.

GloVe là nguồn prior ngôn ngữ bổ sung; không coi nó là KG WordNet/ConceptNet và
không tuyên bố đầu vào ngữ nghĩa giống baseline embedding ngẫu nhiên. Các ablation
cùng PE architecture ở mục 5 dùng cùng GloVe để tách tác động PL/PR. Trước train
phải audit coverage tên lớp/OOV, pin hash file và vector mapping; fallback phải
hữu hạn, xác định theo seed. Background không lấy nhãn từ held-out data.
[Hàm khởi tạo gốc](https://github.com/VL-Group/PENET/blob/9c9f50777c66647799cb7a17dfb855aa98aecd1e/maskrcnn_benchmark/modeling/roi_heads/relation_head/utils_motifs.py#L14).

## 4. Quy ước loss và những bẫy khi port

Quyết định dùng **code commit đã pin làm chuẩn implementation**, đồng thời ghi khác
biệt với biểu thức trong paper. Code hiện tại có `l21_loss / C²`, stop-gradient ở
nhánh prototype đích của PR, và lấy mean của hinge theo từng prototype cho PR
Euclidean; không trộn tùy ý với công thức paper chưa chuẩn hóa/global hinge.
[Khối loss đã kiểm](https://github.com/VL-Group/PENET/blob/9c9f50777c66647799cb7a17dfb855aa98aecd1e/maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py#L210).

Các giá trị khởi điểm lấy từ code: `k1=10`, `k2=1`, `gamma1=1`, `gamma2=7`,
`logit_scale=log(1/0.07)`, dropout 0,2. Bốn loss cộng với hệ số 1 như launcher gốc;
không thêm reweighting ở vòng triển khai đầu. Đây là cấu hình khởi điểm, cần kiểm
scale/gradient vì adapter đổi dimension và số lớp; nếu thay phải version hóa config.

Quy ước đa nhãn do dự án đặt ra:

1. Với pair có tập predicate dương `Y`, CE dùng phân phối đều trên `Y`. Pair
   background có `Y={0}`.
2. Euclidean PL: tìm tối đa `k1` prototype âm gần nhất **ngoài toàn bộ Y**. Lấy
   trung bình hinge riêng cho từng predicate dương, rồi trung bình trên pair.
   Với một nhãn, công thức phải khớp tham chiếu single-label. Không nhân trọng số
   pair chỉ vì có nhiều predicate.
3. Loại diagonal của PR theo ID rõ ràng, không dựa vào giả định một giá trị 0
   đứng đầu sort. Giữ background trong tập prototype theo code tham chiếu.
4. Dùng device từ tensor; không `.cuda()` hoặc hằng 51 rải trong loss. Normalize
   phải có epsilon; tên predicate hoàn toàn OOV không được chia cho 0 như trường
   hợp có thể xảy ra trong `rel_vectors` gốc.
5. Tính squared distance bằng phép norm/matmul có chunk nếu cần, đối chiếu số học
   và gradient với bản broadcast nhỏ. Không tạo tensor `[pairs,C,dimension]`
   khổng lồ khi GQA có 311 lớp. Không clamp/cắt logit hoặc margin âm thầm.
6. Ảnh 0/1 node và batch không có supervised pair phải có quy ước rõ: không
   tạo loss NaN, không phát sinh optimizer update chỉ từ PR trong batch rỗng.

Điểm quan trọng với VG150: historical launcher single-label bỏ pair đa nhãn.
Không đổi mã/nhãn lịch sử đó. Nếu dùng quy ước đa nhãn chung ở vòng PE-Net mới,
phải chạy baseline đối chứng tương ứng và ghi thành **protocol train mới**, giữ
split/evaluator. Các checkpoint Level 1 cũ là kết quả lịch sử, không phải đối
chứng hoàn toàn matched cho một thay đổi supervision.

## 5. Cách kiểm chứng và so sánh

**Ma trận thực nghiệm dự kiến**, không phải danh sách đã chạy:

- MLP, GAT, GAT+KG: các baseline bắt buộc trên cùng split và frozen visual cache.
- `PE-CE`: cùng adapter/prototype head, chỉ cosine CE.
- `PE-PL`: thêm Euclidean PL, chưa PR.
- `PE-PLPR`: đầy đủ PL+PR, là bản phương pháp được chọn.

Các biến thể PE phải cùng initial state khi cùng seed, cùng sampling/thứ tự ảnh,
GloVe mapping và budget; báo cả params/time/memory. Đối chứng random/shuffled text
là phần mở rộng hữu ích nhưng không được dùng để thay ablation PL/PR tối thiểu.
Không dùng kết quả benchmark MLP/GAT để mặc định PE-Net batch 64 vừa GPU.

VG150 dùng split đã khóa của Level 1; GQA dùng full 69.942 train và validation
5.000 ảnh theo manifest hiện có. Train hai dataset riêng, không chuyển checkpoint
đã train VG150 sang GQA. Seeds dự kiến **42, 123, 3407** như Level 1. Chọn checkpoint
bằng validation GC mR@50 với taxonomy cố định của từng dataset; báo GC/NG R/mR@50/100,
mean±std ba seed, support và long-tail theo tần suất **train**. Test chỉ chấm sau
khi khóa lựa chọn; không dùng test để tạo prototype, prior, mapping hay tune loss.

Tiêu chí trước khi gọi là đã triển khai:

- Unit test giá trị/gradient các loss single-label khớp code reference (dropout
  tắt, fixture xác định), cùng test riêng cho đa nhãn, GT trùng, background, empty,
  lớp cuối 50/310, CPU/CUDA và vector OOV.
- Kiểm đối xứng/chiều quan hệ đúng; gradient chỉ vào adapter/head/embedding học
  được, không vào CLIP. Sampler và topology không lộ GT khi forward.
- Pilot thật so PE-CE/PL/PLPR trên cohort đã khóa; loss/gradient hữu hạn, reload
  checkpoint tái chấm được; ghi mọi deviation, không chỉ test dữ liệu tổng hợp.
- Benchmark riêng graph lớn và kiểm hiệu quả batch/accumulation trước main runs.
- Hoàn thành ba seed và báo cáo cả trường hợp không cải thiện; không yêu cầu
  thắng baseline để công nhận implementation đúng.

## 6. Việc cần làm tiếp và trạng thái

**Bước kế tiếp:** chuẩn bị GloVe mapping có provenance, viết adapter và bốn thành
phần loss với test tham chiếu, rồi chạy pilot PE trên 512/128 ảnh thật đã có cache.
Chưa chạy full cache/full training chỉ vì đã chọn xong paper.

Quyết định này hoàn thành bước chọn phương pháp; mục “triển khai/tái lập phương
pháp 2023+” vẫn **chưa hoàn tất**. PE-Net không giải quyết thay phần 300 cặp MagicBrush,
human review 20 mẫu hoặc model thao tác đồ thị; những phần đó vẫn còn nguyên.

[Biên bản nguồn và quyết định dạng JSON](../reports/level2/method-selection-v1.json)
lưu commit/link và SHA-256 bản text code đã đọc. Source review nằm trong
`data/external/penet-review-9c9f507/` (không commit source ngoài vào repo); chưa
chạy code legacy đó. License của repo upstream là MIT; nếu port/copy mã phải giữ
notice tương ứng. Lượt này chỉ sửa tài liệu, không đổi evaluator, model, dataset,
checkpoint hoặc notebook cá nhân.
