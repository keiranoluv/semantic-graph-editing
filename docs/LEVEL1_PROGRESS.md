# Theo dõi Project 01 — Mức 1

Cập nhật: 2026-10-25.

## Cách phối hợp

- Codex sửa code/tài liệu và được chạy kiểm tra môi trường, dữ liệu, tests,
  thiết lập môi trường và smoke tests nhỏ trên CPU.
- Người dùng chạy các lệnh train GPU. Codex cung cấp lệnh và giải thích mục đích;
  không tự khởi chạy train GPU.
- Mỗi nhóm lệnh phải có mục đích, thư mục thực thi và đầu ra cần gửi lại.
- Chỉ đánh dấu kiểm tra PASS khi có đầu ra thực tế, ghi rõ môi trường và phạm vi.
- Không ghi đè baseline, checkpoint hoặc dữ liệu hiện có. Không tự commit/push.
- Không dùng kết quả test để lựa chọn mô hình hoặc điều chỉnh tham số.

## Mục tiêu mức 1

Nguồn yêu cầu: `Project01_Graph_Editing.pdf`, trang 3–5.

- VG150 PredCls; MLP phi đồ thị, GAT và GAT có tri thức nền trên cùng split.
- Liên kết nhãn với WordNet/ConceptNet, kiểm tra tay và ghép nhúng
  TransE/ComplEx vào đặc trưng đỉnh.
- Recall@50/100 và mean Recall; ít nhất 3 seeds cho thực nghiệm chính,
  báo cáo trung bình và độ lệch chuẩn.
- Ablation bỏ tri thức và bỏ hoặc xáo trộn cạnh đồ thị.
- Phân tích ít nhất 5 dự đoán đúng và 5 dự đoán sai.
- Môi trường, dữ liệu, split và lệnh chạy có thể tái lập trên máy thành viên khác.
- Báo cáo, slide, notebook/demo và bảng phân công.

Ưu tiên WordNet để hoàn thiện pipeline đầu tiên. Quyết định bổ sung ConceptNet
sau khi có kết quả và đánh giá thời gian. Không giả định hai không gian embedding
được huấn luyện độc lập đã căn chỉnh với nhau.

## Trạng thái 5 bước

| Bước | Mục đích và sản phẩm | Trạng thái | Điều kiện hoàn thành |
|---|---|---|---|
| 1. Kiểm tra nền và môi trường | Xác minh C.1, baseline, phụ thuộc, nguồn dữ liệu và split; hướng dẫn thiết lập cho nhóm | Hoàn thành kiểm tra nền; còn giới hạn về split/ngân sách lịch sử | 150 tests ban đầu PASS, split tái tạo khớp, 9 checkpoint/config/epoch khớp; xem audit |
| 2. C.2 — Xuất embedding | Chuyển checkpoint TransE thành bảng đặc trưng đúng class ID VG150, kèm mask và provenance | Hoàn thành export reviewed chính [151,64], 150/150 class | `best.pt` epoch 52, hash KG/review và tensor export khớp; `load_class_features` PASS |
| 3. GAT+WordNet | Tích hợp KG vào đặc trưng đỉnh, có cấu hình và run riêng | Code reviewed và CPU smoke hoàn thành | Reviewed GAT+KG 1 epoch CPU PASS; 3 chế độ KG-on/KG-off/edges-off draft trước đó PASS; baseline được bảo toàn |
| 4. Thực nghiệm | So sánh cùng protocol với 3 seeds; ablation tri thức/cạnh | 9 baseline đã đánh giá lại full validation; aggregator có; chờ GAT+KG GPU | Chưa có kết quả GAT+KG train dài hoặc mean/std cho nhánh KG |
| 5. Phân tích và bộ nộp | 5 đúng/5 sai, notebook, bảng kết quả, hướng dẫn, tài liệu báo cáo/demo | Có case kỹ thuật, notebook đã chạy, bản chuẩn bị báo cáo/slide; phân tích người và kết quả final còn lại | Phân tích thủ công trên model final và bộ nộp hoàn chỉnh |

## Điểm xuất phát và những điều chưa xác minh

- Đã đọc PDF, code GAT, phần nạp KG của TransE và các ghi chú handoff.
- Theo ghi chú baseline, đã có 9 runs MLP/GAT cho seeds 42, 123, 3407;
  các số liệu được ghi là validation, chưa phải đánh giá held-out test cuối cùng.
- Full suite hiện tại có 176 tests PASS; lint/format trên các file mới/sửa
  đã sửa PASS. Chưa báo cáo toàn bộ pre-commit `--all-files` PASS.
- `myenv` ban đầu chỉ có pip; đã cài dependencies và chạy CPU smoke. Sau đó đổi
  sang torch 2.7.0+cu128: CUDA available, nhận RTX 5090; full suite hiện tại 176 tests PASS.
- Mapping AI-provisional cần phân biệt với mapping được người kiểm tra chấp thuận.
- README đã cập nhật phạm vi C.1/C.2/GAT+WordNet draft; không khẳng định train GPU đã xong.
- Cần xác minh tính tương đương split, giới hạn số ảnh và protocol của các baseline.
- Baseline hiện tại không có đặc trưng vùng ảnh. Cần ghi rõ lựa chọn này trong báo cáo;
  đề cho phép điều chỉnh kiến trúc nếu giải thích cơ chế.
- Cần xác nhận với giảng viên phạm vi notebook cho mức 1: mục sản phẩm chung có
  minh họa thao tác chỉnh sửa, trong khi huấn luyện thao tác thuộc mức 2.

## Nhật ký công việc

| Ngày | Công việc | Bằng chứng / kết quả | Tiếp theo |
|---|---|---|---|
| 2026-10-24 | Đối chiếu yêu cầu PDF với repository và thống nhất cách phối hợp | Người dùng giữ quyền chạy train GPU; Codex được kiểm tra và smoke CPU | Nhóm lệnh B1-01 |
| 2026-10-24 | Tạo tài liệu theo dõi 5 bước | File này | Kiểm kê môi trường và đọc quy trình download hiện có |
| 2026-10-24 | Bước 1: cài môi trường CPU, full tests, smoke MLP, kiểm tra split/checkpoint | 150 tests PASS; 9/9 config và best epoch khớp; split/metadata tái tạo khớp | Xem `docs/experiments/level1_audit_2026-10-24.md` |
| 2026-10-24 | Thêm hướng dẫn thiết lập/download cho nhóm | `docs/TEAM_SETUP_LEVEL1.md`; requirements đầy đủ | Máy teammate chạy và đối chiếu hash |
| 2026-10-24 | C.2 draft export | 124/150 class, dim 64; 10 tests PASS | Review mapping trước kết quả cuối |
| 2026-10-24 | GAT+WordNet, KG-off và edges-off | 163 tests tổng cộng PASS; cả 3 chế độ CPU smoke PASS | Người dùng chạy train GPU |
| 2026-10-24 | Evaluator chung MLP/GAT/GAT+KG | Smoke 8 ảnh validation; có hash dữ liệu, per-predicate recall và test gate | Full validation sau train |
| 2026-10-24 | Kiểm tra resume CPU bằng giả lập ngắt ở epoch 2 | History và toàn bộ model tensors khớp run 2 epochs liên tục | Chưa suy ra tương đương GPU |
| 2026-10-24 | Chuẩn bị bảng review dễ đọc khi người dùng cần giải thích mapping | CSV v2 có 150 dòng và 150 gợi ý; 124 `selected_provisional`, 26 `deferred_unmapped`, toàn bộ quyết định và reviewer để trống | Nhóm đọc nghĩa và ghi quyết định thực tế |
| 2026-10-24 | Làm rõ 124/150 sau thắc mắc của người dùng | Kiểm tra `ai_draft_summary.json` và `deferred_review.csv`: 23 WordNet auto + 101 AI-provisional + 26 deferred có suggestion; `short.n.01` sai ngữ cảnh quần áo | Chưa đổi KG/TransE/C.2 hiện tại |
| 2026-10-24 | Kiểm tra worksheet v2 | 150 suggestions, 26 deferred kèm lý do; 171 tests PASS và pre-commit file sửa PASS | Nhóm xem/điền quyết định thực tế |
| 2026-10-24 | Chuẩn bị ví dụ train cho human review | 150/150 class có 5 image IDs và hộp bao; chỉ đọc nhãn/hộp train, không đọc predicate GT | Nhóm mở JPEG và ghi accept/defer/reject thật |
| 2026-10-24 | Kiểm tra công cụ ví dụ train | 750 dòng ví dụ gồm image ID, object index, box XYXY; 173 tests và pre-commit trên file đã sửa PASS | Người review dùng CSV cùng định nghĩa WordNet |
| 2026-10-24 | Đánh giá lại 9 baseline trên 5.000 ảnh validation | 9/9 report khớp history; bảng mean/std ở `docs/experiments/level1_baseline_verified.md` | Cần GAT+KG 3 seeds để so sánh |
| 2026-10-24 | Bước 4/5: aggregator và ví dụ | Aggregator từ chối subset và khác split; smoke 5 hit + 5 missed từ GAT seed 42 | Phân tích tay model cuối và hoàn thiện báo cáo |
| 2026-10-24 | Notebook mức 1 | Chạy thành công với GAT baseline và ảnh validation đầu tiên; visual hóa box và GT/predicted scene graph | Ảnh JPEG tùy chọn, nhóm demo với checkpoint final |
| 2026-10-24 | Chuẩn bị bộ nộp và QA sau thay đổi | `docs/LEVEL1_SUBMISSION.md`; 169 tests, lint/format và pre-commit trên file đã sửa PASS | Báo cáo/slide cuối cần GPU results và đóng góp thực tế |
| 2026-10-24 | Chuẩn bị CUDA trong `myenv` | torch 2.7.0+cu128, CUDA 12.8, RTX 5090 available; pip check và 169 tests PASS | Người dùng chạy train GPU seed 42 |
| 2026-10-25 | Tải ảnh phục vụ review WordNet | 150 JPEG train khác nhau cho 150 class, 21.5 MB; đối chiếu class ID, SHA-256, giải mã JPEG, kích thước metadata và hộp bao đều PASS; host gốc từ chối kết nối nên đọc từng file trong ZIP mirror | Nhóm xem vật thể trong hộp và ghi quyết định thật vào worksheet |
| 2026-10-25 | Tạo bản ảnh có tên class để review | 150 bản sao ở `data/raw/vg150/images_review/`, dạng `001_airplane_2349975.jpg` với số cuối là `image_id` gốc; giữ nguyên ảnh nguồn | Nhóm xem ảnh theo tên class, tra hộp bao trong `review_images.csv` khi cần |
| 2026-10-25 | Thêm trang xem hộp bao cho review | `images_review/index.html` hiển thị 150 ảnh với khung đỏ và lọc theo class ID, nhãn, image ID; `bike` trên ảnh 2376992 nằm sát mép trái phía dưới | Nhóm dùng hộp bao để xét đúng đối tượng, xem thêm ví dụ nếu ảnh bị cắt |
| 2026-10-25 | Thêm ảnh thứ hai cho class `bike` | `011_bike_2344860.jpg` và gallery có 151 thẻ; ảnh train này có xe máy đỏ, hai hộp gần trùng mang nhãn `bike` và `motorcycle` | Reviewer cân nhắc nhãn `bike` có thể đa nghĩa, chưa chốt synset |
| 2026-10-25 | Mở rộng review lên 5 ảnh/class | 750 cặp class–ảnh train, 237 JPEG gốc, 750 bản sao tên class và 750 thẻ gallery; SHA-256, giải mã JPEG, kích thước metadata và hộp bao đều PASS | Nhóm kiểm tra 5 ví dụ từng nhãn và ghi quyết định người review |
| 2026-10-25 | Kiểm tra 5 box của `building` | Cả 5 khớp HDF5/class ID 22; ảnh `1592671` chụp nội thất nhưng box phủ 99% ảnh, `2389149` box chỉ 0,5% ảnh | Ghi hai ví dụ là khó xác nhận/khả năng annotation nhiễu; không dùng riêng chúng để quyết định synset |
| 2026-10-25 | Kiểm tra 5 box của `cap` | Cả 5 khớp HDF5/class ID 25; ảnh `2347962` và `2354978` là đồ đội đầu, `2399043` là nắp chai, `2388888` khó xác định | Giữ `defer`, ghi rõ class có nhiều nghĩa, chưa chọn synset duy nhất |
| 2026-10-25 | Kiểm tra 5 box của `ear` | Cả 5 khớp HDF5/class ID 40; đều là tai ngoài của chó, người, bò, cừu, ngựa vằn | Reviewer so `ear.n.01` với `auricle.n.02`, chưa điền quyết định hộ |
| 2026-10-25 | Kiểm tra 5 box của `glass` | Cả 5 khớp HDF5/class ID 54; 2 ly uống, 2 kính đeo mắt, 1 vùng kính nhỏ chưa chắc; candidate nguồn thiếu `spectacles.n.01` | Giữ `defer`, ghi class đa nghĩa và chốt chính sách mapping trước khi chọn synset |
| 2026-10-25 | Kiểm tra 5 box của `wheel` | Cả 5 khớp HDF5/class ID 144; box chỉ bánh xe đạp, xe máy hoặc toa tàu, không bao cả xe | `wheel.n.01` phù hợp hơn `bicycle.n.01`; reviewer tự ghi quyết định |
| 2026-10-25 | Biên dịch mapping đại diện của Toan | 150/150 synset được chọn; sửa `orange` thành quả cam, `stand` thành vật đỡ; thêm candidate WordNet 3.0 cho `short_pants.n.01` với provenance; compiler 150 accept, mapping mới 150 linked_manual | Ghi rõ class đa nghĩa dùng synset đại diện; bổ sung bằng chứng/lý do từng class cho manifest C.1 reviewed; dựng KG/TransE/export mới, không tái dùng checkpoint draft 124 class |
| 2026-10-25 | Kiểm tra mapping reviewed trong `myenv` | NLTK 3.9.1 tạo 150 linked_manual; 151 dòng ngữ nghĩa trùng bản Python hệ thống NLTK 3.10.3, chỉ metadata phiên bản khác; bản `myenv` là `data/knowledge/wordnet-reviewed-toan-v1/` | Teammate dùng cùng `myenv` hoặc output dir riêng khi đổi NLTK |
| 2026-10-25 | CPU dry-run WordNet KG từ mapping Toan | 150 class resolved, 146 synset gốc riêng, 1.077 entities và 1.312 triples ở 2 hops; chưa cung cấp reviewed override/evidence gate nên `ready_for_gat_kg_training=false` | Không train final từ dry-run này; bổ sung manifest audit và dựng KG versioned sau khi nhóm chốt chính sách |
| 2026-10-25 | CPU dry-run KG với override 150 class của Toan | Cổng `--require-fully-reviewed` PASS; 146 synset gốc, 1.077 entities, 1.312 triples; `ready_for_gat_kg_training=true` | Người dùng tạo artifact KG thực tế bằng lệnh bên dưới; Codex kiểm tra đầu ra và chuẩn bị nhánh TransE/C.2/GAT+KG |
| 2026-10-25 | Chuẩn bị lệnh GPU và WandB cho nhánh KG 150 class | KG smoke riêng tạo được 1.077 entities/1.312 triples; TransE transductive dry-run PASS (1.050/131/131 triples); GAT+KG WandB offline CPU smoke 1 epoch PASS; 0 GPU train do Codex | Người dùng tạo KG chính, chạy TransE GPU theo validation MRR, export CPU, rồi GAT+KG GPU với WandB online; lệnh tại `docs/knowledge/c2_gat_wordnet.md` |
| 2026-10-25 | Sửa ngân sách TransE trước run 150 class | Run WordNet cũ 10 epochs còn tăng validation MRR tại epoch 10; thêm WandB và chọn `best.pt` theo validation MRR, early stop; CPU smoke 2 epochs và export 150/150 PASS | Người dùng chạy TransE tối đa 200 epochs, patience 20, rồi export `best.pt`; không dùng checkpoint epoch 10 mặc định |
| 2026-10-25 | Kiểm tra KG chính sau khi người dùng tạo | `data/knowledge/wordnet-kg-toan-v1/metadata.json`: 150 class resolved, 1.077 entities, 1.312 triples, `ready_for_gat_kg_training=true`; TransE chính chưa chạy | Commit thay đổi early stopping/WandB của TransE, sau đó người dùng chạy GPU theo lệnh mới |
| 2026-10-25 | Toan xác nhận bản 150 synset là chính thức | Tạo approval declaration không nhận vơ lý do riêng từng class; manifest batch khóa SHA-256 của KG/mapping/override/selection; C.1 reviewed dry-run, TransE CPU smoke 2 epochs, C.2 export 150/150 và GAT+KG CPU smoke 1 epoch PASS | Commit code reviewed và chạy TransE GPU `--mode reviewed`, export `best.pt`, rồi GAT+KG GPU không dùng `--allow-draft` |
| 2026-10-25 | Người dùng chạy TransE reviewed trên GPU | 72 epochs, best epoch 52 theo validation filtered MRR=0,046061, early stop patience 20; test filtered MRR=0,033331 (KG link prediction, không phải PredCls). Cổng review, hash KG/checkpoint và epoch `best.pt` đều PASS; WandB mode online | Người dùng export `best.pt` thành C.2 reviewed, gửi output 150/150 để kiểm tra trước GAT+KG GPU |
| 2026-10-25 | Người dùng export C.2 reviewed | `[REVIEWED] 150/150 mapped classes`, tensor [151,64], norm class trung bình 1,0; `load_class_features` PASS. GAT+KG CPU smoke 1 epoch với đúng export chính PASS | Người dùng chạy GAT+KG seed 42 GPU, rồi đánh giá PredCls full validation và ablation |
| 2026-10-25 | Giải thích KG link prediction MRR | KG 1.077 entities/1.312 triples; 629 entities chỉ có 1 cạnh train, 887 có tối đa 2. Test MRR=0,03333, Hits@10=0,05725 so với kỳ vọng random filtered khoảng 0,00704/0,00931; 100 triple train sample MRR=0,12166. IsA test MRR=0,02654, HasPart=0,04094 | MRR của TransE không phải metric VG150 PredCls; xem GAT+KG cùng KG-off để xác định giá trị của embedding |

## B1-01 — Kiểm kê trước khi cài đặt

Chạy trong `/workspace/workspace/semantic-graph-editing`.
Mục đích: lấy phiên bản Python/pip, package hiện có và nội dung hướng dẫn dữ liệu
để chuẩn bị lệnh cài đặt/download đúng với repository. Không train, không download,
không sửa hoặc xóa dữ liệu trong nhóm này.

```bash
source myenv/bin/activate
python --version
python -m pip --version
python -m pip list
git status --short
git diff --check
python - <<'PY'
from pathlib import Path

paths = [
    'requirements.txt',
    'configs/dataset/vg150.yaml',
    'scripts/download_vg150.sh',
    'docs/datasets/VG150.md',
    'docs/knowledge/c1_transe_training.md',
]
for name in paths:
    p = Path(name)
    print(f'\n===== {name} =====')
    print(p.read_text(encoding='utf-8') if p.exists() else 'MISSING')
PY
```

Nhóm kiểm tra này đã được Codex chạy sau khi người dùng cho phép. Các lệnh
được giữ làm tham khảo; không cần người dùng chạy lại. Không cần tạo lại `myenv`.

## Thiết lập máy thành viên và tải dữ liệu

Đã có hướng dẫn tại [TEAM_SETUP_LEVEL1.md](TEAM_SETUP_LEVEL1.md): Python/venv,
PyTorch CPU/CUDA, dependencies, download annotations VG150, tạo split, hash tham
chiếu, WordNet corpus và lệnh smoke. JPEG tùy chọn cho notebook có lệnh tải
theo image ID; host ảnh không kết nối được ở môi trường kiểm tra này.

Không tải lại toàn bộ dataset trong phiên này vì dữ liệu đã có. Đã xác minh
HDF5 checksum và tái tạo split/metadata từ input đang có. Tính sẵn sàng của
các URL download trên máy teammate chưa được kiểm chứng bằng một lượt tải mới.

## Bước tiếp theo do người dùng thực hiện

- Mapping 150/150 của Toan đã được xác nhận chính thức; KG và manifest reviewed
  batch đã có ở máy hiện tại. Teammate tạo lại bằng lệnh trong
  [c2_gat_wordnet.md](knowledge/c2_gat_wordnet.md).
- TransE GPU và C.2 reviewed đã hoàn tất; người dùng chạy GAT+KG GPU từ export
  `[REVIEWED] 150/150`. Lệnh tại tài liệu C.2 nêu trên; gửi output và đường dẫn
  run để Codex kiểm tra trước khi tổng hợp kết quả.
- Manifest ghi nguồn xác nhận theo lô và giới hạn thiếu lý do riêng cho phần lớn
  class. Khi báo cáo, nêu đúng giới hạn này và vấn đề nhãn đa nghĩa.
- Không đánh dấu bước 4 hoặc 5 hoàn thành trước khi có dữ liệu thực nghiệm,
  phân tích 5 đúng/5 sai và notebook/demo thực tế.

Không commit dữ liệu tải về, checkpoint, credentials hoặc đường dẫn riêng của máy.
