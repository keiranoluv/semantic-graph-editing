# Theo dõi Project 01 — Mức 1

Cập nhật: 2026-10-25.

## Cách phối hợp

- Codex sửa code/tài liệu và được chạy kiểm tra môi trường, dữ liệu, tests,
  thiết lập môi trường và smoke tests nhỏ trên CPU.
- Người dùng chạy các lệnh train/đánh giá PredCls GPU. Codex cung cấp lệnh và
  giải thích mục đích; TransE v2 GPU là ngoại lệ đã được người dùng yêu cầu chạy trực tiếp.
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
| 3. GAT+WordNet | Tích hợp KG vào đặc trưng đỉnh, có cấu hình và run riêng | Hoàn thành code reviewed, CPU smoke và 3 run KG-on GPU | Cả 3 seed có `best.pt` và dừng bằng early stopping; baseline được bảo toàn |
| 4. Thực nghiệm | So sánh cùng protocol với 3 seeds; ablation tri thức/cạnh | Đủ 15 run CLIP và 15 report evaluate độc lập trên toàn bộ 5.000 ảnh validation; summary 3 seed đã tạo | Đã đo đủ R@50/100, mR@50/100; cần khóa lựa chọn mô hình trước khi dùng held-out test |
| 5. Phân tích và bộ nộp | 5 đúng/5 sai, notebook, bảng kết quả, hướng dẫn, tài liệu báo cáo/demo | Có 5 hit/5 miss từ GAT+CLIP seed 42, ảnh review và nhận xét sơ bộ; notebook đã chạy; xác nhận tay và bộ nộp còn lại | Thành viên nhóm kiểm từng ảnh/box, hoàn thiện báo cáo, slide và demo |

## Điểm xuất phát và những điều chưa xác minh

- Đã đọc PDF, code GAT, phần nạp KG của TransE và các ghi chú handoff.
- Theo ghi chú baseline, đã có 9 runs MLP/GAT cho seeds 42, 123, 3407;
  các số liệu được ghi là validation, chưa phải đánh giá held-out test cuối cùng.
- Full suite hiện tại có 187 tests PASS; lint/format trên các file mới/sửa
  đã sửa PASS. Chưa báo cáo toàn bộ pre-commit `--all-files` PASS.
- `myenv` ban đầu chỉ có pip; đã cài dependencies và chạy CPU smoke. Sau đó đổi
  sang torch 2.7.0+cu128: CUDA available, nhận RTX 5090; full suite hiện tại 187 tests PASS.
- Mapping AI-provisional cần phân biệt với mapping được người kiểm tra chấp thuận.
- README đã cập nhật phạm vi C.1/C.2/GAT+WordNet draft; không khẳng định train GPU đã xong.
- Cần xác minh tính tương đương split, giới hạn số ảnh và protocol của các baseline.
- Ma trận annotation-only lịch sử không có đặc trưng vùng ảnh. Lúc bắt đầu pipeline chưa tải JPEG;
  237 JPEG hiện có được tải sau đó chỉ để review synset, không phải tập ảnh
  huấn luyện. PDF mục 4.1 (trang 2) nêu nhãn/vùng ảnh/hình học là cách tổ chức
  input được đề xuất và cho phép input khác; trang 3 nói kiến trúc cấp độ 1
  *nên* gồm cả ba và cho phép điều chỉnh nếu giải thích cơ chế. Bảng yêu cầu
  cấp độ 1 không đặt vùng ảnh thành điều kiện bắt buộc riêng. Cần giải trình
  lựa chọn annotation-only trong báo cáo và trao đổi với giảng viên nếu coi đó
  là thay đổi lớn; nếu quyết định thêm vùng ảnh, phải train/so sánh lại.
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
| 2026-10-25 | Chuẩn bị ma trận thực nghiệm mức 1 | Launcher ban đầu ở `tools/level1_matrix.py`, dựng 6 ablation còn thiếu (KG-off/edges-off × 3 seeds) và đánh giá 9 checkpoint trên full validation, tổng hợp mean/std; khi 3 run KG-on còn chạy, file nằm ngoài `src/` và `scripts/` nên không đổi code hash của chúng; dry-run cho 6 train và 9 eval PASS. Ba run khi đó dùng 4,5–11,9 GiB/32 GiB GPU trong mẫu 12 giây | Người dùng khởi chạy ablation sau khi KG-on hoàn tất; sau khi toàn bộ 9 run hoàn tất, chạy evaluate và gửi summary |
| 2026-10-25 | Hoàn tất 3 run GAT+WordNet reviewed GPU | Seeds 42/123/3407 dừng early stopping ở epoch 69/70/45, `best.pt` epoch 54/55/30. Validation mR@50 tại `best.pt`: 20,2079% / 20,0762% / 18,9785%; mean ± sample std 19,7542 ± 0,6750%. GAT baseline cùng seeds: 19,7963 ± 1,0300%. Các số KG-on lấy từ history/checkpoint, chưa chạy evaluator độc lập; seed 42 tăng nhưng hai seed còn lại giảm. | Người dùng chạy 6 ablation GPU, rồi đánh giá độc lập cả 9 checkpoint trên full validation; chưa dùng test để chọn mô hình |
| 2026-10-25 | Sửa launcher ablation chạy trực tiếp | Lệnh `python tools/level1_matrix.py` ban đầu lỗi `ModuleNotFoundError: src` trước khi khởi chạy GPU; thêm repo root vào import path và cho dry-run kiểm tra feature bundle. Dry-run `--max-parallel 6`, py_compile, ruff và `git diff --check` PASS | Người dùng chạy lại `train-ablations --max-parallel 6`; không có run ablation nào được tạo bởi lần lỗi trước |
| 2026-10-25 | Chuyển launcher vào `src/` theo yêu cầu người dùng | Đường dẫn hiện tại là `src/level1_matrix.py`, chạy qua `python -m src.level1_matrix`; chỉ chuyển orchestration, không sửa trainer/model. Vì `train_gat_kg.py` hash mọi file Python dưới `src/` và `scripts/`, các ablation mới sẽ ghi thêm hash của launcher so với 3 KG-on đã hoàn tất; ghi rõ khác biệt provenance này khi báo cáo | Người dùng commit file và tài liệu trước khi chạy GPU; Codex không tự commit |
| 2026-10-25 | Người dùng khởi chạy 6 ablation đồng thời | `src.level1_matrix train-ablations --max-parallel 6` in ra 6 dòng `[START]` cho KG-off và edges-off, seeds 42/123/3407; kiểm tra tiến trình thấy đủ 6 trainer đang chạy. Chưa có kết quả cuối | Không khởi chạy trùng trên máy này; chờ tất cả dừng rồi chạy stage evaluate |
| 2026-10-25 | Audit GAT+KG và TransE khi ablation đang chạy | 51 test mục tiêu PASS. C.2 [151,64], 150 class valid, background zero; vector/mask trong `best.pt` GAT+KG khớp từng phần tử. Cả 6 config ablation có cùng hash dữ liệu/KG, chỉ đổi switch dự kiến và `code_sha256` do chuyển launcher vào `src/`. Trên 8 ảnh validation, cùng trọng số: tắt KG đổi logits trung bình 1,812; tắt cạnh đổi 4,223. TransE thực hiện score `||h+r-t||`, margin loss, chuẩn hóa entity; review/KG hashes khớp. Điểm cần ghi nhận: negative sampler và validation filtered MRR đang dùng tập `known` gồm cả triple KG test; đánh giá lại `best.pt` với chỉ train+val để lọc cho validation làm MRR đổi 0,00000046. Đây là hạn chế protocol KG link prediction, không phải rò rỉ VG150 test; chưa sửa `src/` hoặc `scripts/` giữa 6 run GPU. | Sau ablation, đối chiếu KG-on/KG-off; nếu cần báo cáo KG link prediction theo protocol nghiêm ngặt, sửa bộ lọc và tái chạy C.1/C.2 bằng phiên bản mới |
| 2026-10-25 | Bốn ablation hoàn tất | `kg_off` seeds 42/3407 và `edges_off` seeds 42/123 có `last.pt` ở ngưỡng early stop; người dùng báo exit=0. Validation mR@50 tại `best.pt`: KG-off 19,6313%/18,8603%, edges-off 19,4674%/19,2297%. Cùng seed, KG-on cao hơn lần lượt +0,5766/+0,1182 và +0,7405/+0,8465 điểm %. `kg_off` seed123 và `edges_off` seed3407 còn chạy tại thời điểm kiểm tra | Chờ 2 run còn lại; chưa tính mean/std 3 seed hoặc kết luận cuối từ 4 run |
| 2026-10-25 | KG-off đủ 3 seed hoàn tất | Seed 123 exit=0, `best.pt` epoch 53, validation mR@50=18,9907%. KG-off seeds 42/123/3407: 19,6313%/18,9907%/18,8603%, mean ± sample std 19,1608 ± 0,4127%. KG-on tương ứng 19,7542 ± 0,6750%, paired delta +0,5934 điểm %, dương cả 3 seed. Đây là validation lấy từ history, chưa đánh giá độc lập; edges-off seed 3407 vẫn chạy | Chờ run cuối rồi chạy stage evaluate cho cả 9 checkpoint; không đánh giá test trước khi chốt protocol |
| 2026-10-25 | Cả 6 ablation hoàn tất | Kiểm tra đủ 9 cặp `best.pt`/`last.pt`, tất cả dừng ở `bad_epochs=15`, không còn tiến trình train. `edges_off` seed 3407 best epoch 74, validation mR@50=19,9819%. Ba seed `edges_off`: 19,4674%/19,2297%/19,9819%, mean ± sample std 19,5597 ± 0,3845%. KG-on hơn `edges_off` trung bình +0,1945 điểm %, nhưng seed 3407 thấp hơn 1,0034 điểm %. Số này lấy từ checkpoint/history, chưa phải lượt evaluate độc lập | Người dùng chạy `myenv/bin/python -m src.level1_matrix evaluate --max-parallel 3 --device cuda` để đánh giá 9 checkpoint trên toàn bộ validation và tạo summary; chưa dùng test |
| 2026-10-25 | Đánh giá độc lập cả 9 checkpoint hoàn tất | Đủ 9 report, mỗi report `split=val`, `images_evaluated=5000`, `full_split=true`; summary 3 nhóm. Validation mR@50 mean ± sample std: KG-on 19,7542 ± 0,6750%, KG-off 19,1608 ± 0,4127%, edges-off 19,5597 ± 0,3845%; GAT baseline 19,7963 ± 1,0300%. KG-on hơn KG-off +0,5934 điểm %, hơn edges-off +0,1945 điểm %; vẫn thấp hơn GAT baseline gốc 0,0421 điểm %. R@50 KG-on 64,1446%, thấp hơn KG-off 64,3164% và edges-off 64,7673%. Đây là kết quả annotation-only với embedding TransE hiện tại; không có VG150 test hoặc đặc trưng vùng ảnh | Giữ kết quả làm baseline kỹ thuật, bổ sung vùng ảnh và quyết định sửa protocol TransE trước đợt train final; không dùng test để chọn giữa các phương án |
| 2026-10-25 | Checkpoint Git trước nhánh ảnh | Người dùng commit `38b35ec` cho PDF, tài liệu và report annotation-only; Codex không tự commit | Tách rõ thay đổi CLIP/TransE v2 khỏi kết quả cũ |
| 2026-10-25 | Tải đủ JPEG VG150 theo split | 89.169/89.169 JPEG train/val/test, 12 GiB ảnh và 15 GiB archive; giải mã và SHA-256 từng ảnh, hash archive trong `images_manifest.json`. Có 15 ảnh khác kích thước metadata, gồm ảnh 61530: JPEG 1000×586 so với metadata 1024×768. Dữ liệu local, không commit | Khi crop CLIP, scale GT box theo từng trục sang kích thước JPEG; giữ hình học GT gốc |
| 2026-10-25 | CLIP vùng ảnh và TransE filter v2 | Thay placeholder bằng cache HDF5 theo GT box, CLIP OpenAI ViT-B/32 đóng băng, vector 512D chuẩn hóa L2; MLP semantic/GAT/GAT+KG cùng đọc cache. CPU smoke 1 ảnh train/14 box: 14 vector hữu hạn, norm 0,99989–1,00014 sau float16; cache chính có 1/89.169 ảnh. Negative TransE chỉ lọc train; validation train+val; test sau chọn checkpoint lọc cả ba split. Full suite 183 tests PASS, ruff/black/isort và `git diff --check` PASS; TransE reviewed dry-run PASS (1.050/131/131 triples). Chưa chạy GPU train/eval hoặc cache CLIP đầy đủ | Người dùng chạy tạo cache CLIP GPU đầy đủ, TransE GPU v2, export C.2 mới, rồi ma trận 15 run; lệnh ở `docs/experiments/level1_clip_regions.md` |
| 2026-10-25 | Người dùng khởi chạy tạo cache CLIP GPU | Kiểm tra tiến trình thấy `scripts.cache_vg150_regions --device cuda --batch-size 64` đang chạy; file HDF5 còn khóa ghi, chưa có manifest hoàn tất. Codex không khởi chạy tiến trình GPU này | Chờ dòng `[OK]` và `manifest.json`, rồi kiểm tra 89.169 ảnh/GT box trước ma trận train |
| 2026-10-25 | Làm rõ nguồn dữ liệu theo phản hồi người dùng | `scripts/download_vg150_images.py` nay khai báo trực tiếp hai URL ZIP mirror Hugging Face và `--dry-run` in URL; annotations HDF5/từ điển/metadata vẫn có URL riêng trong `scripts/download_vg150.sh`. Cập nhật hướng dẫn teammate; dry-run, ruff/black/isort và `git diff --check` PASS. Không thay `images_manifest.json` hoặc cache CLIP đang chạy | Người dùng/teammate có thể kiểm tra chính xác nguồn và hash trước khi tái lập |
| 2026-10-25 | Cache CLIP hoàn tất và audit input 15 run | `features.h5` + `manifest.json` có 89.169/89.169 ảnh được đánh dấu hoàn tất, vector 512D; đồ thị train/val/test đầu tiên đều có `x` 517D hữu hạn. Đối chiếu PDF: class ID→embedding nhãn 32D, GT box→hình học 5D, crop→CLIP 512D đóng băng, GAT+KG thêm TransE 64D; MLP không truyền tin, GAT dùng topology đầy đủ không đọc predicate GT. Hai lựa chọn cần giải thích trong báo cáo: embedding nhãn học được thay vì GloVe/CLIP text gợi ý; fully-connected candidate graph thay vì dùng cạnh quan hệ GT (tránh lộ nhãn). Checkpoint MLP/GAT CLIP nay khóa hash dữ liệu/code, summary từ chối khác input/budget; full suite 184 tests PASS, lint/format và diff check PASS | TransE v2/export mới vẫn chưa có; sau đó mới chạy ma trận 15 run và đánh giá validation |
| 2026-10-25 | CPU smoke ba mô hình với cache CLIP hoàn chỉnh | MLP, GAT và GAT+KG đều train 1 epoch CPU trên 8 train/8 val images, tạo `best.pt` và evaluate CPU 8 ảnh validation ra report hữu hạn. KG smoke dùng export WordNet cũ chỉ để kiểm tra wiring. Sửa hai help string còn ghi ResNet18. Audit rủi ro và lệnh 15 run ở `docs/experiments/level1_clip_regions.md` | Chờ TransE v2/export mới trước 15 GPU run |
| 2026-10-25 | Train TransE v2 và export WordNet | Theo yêu cầu mới của người dùng, Codex chạy đúng lệnh TransE GPU seed 42 với filter train-only/validation train+val; exit 0, early stop epoch 60, `best.pt` epoch 40, validation MRR 0,0441983 và KG test MRR 0,0260909. CPU export từ `best.pt` tạo C.2 v2 shape [151,64], 150/150 class valid. Launcher ma trận CLIP dry-run PASS với 15 jobs tuần tự; chưa chạy PredCls GPU. Run dữ liệu/W&B không commit | Người dùng commit code/tài liệu rồi chạy 15 PredCls GPU run và đánh giá validation |
| 2026-10-25 | Tăng batch validation ma trận CLIP | Sau commit `0cb72d8`, kiểm tra chưa có run PredCls GPU nào và không có trainer đang chạy. Đổi batch validation trong từng epoch và lượt evaluate độc lập từ 128/32 thành 512 cho cả 15 run; giữ batch train 512 và protocol khác như cũ | Người dùng commit thay đổi này trước khi chạy ma trận GPU; theo dõi VRAM khi chạy song song |
| 2026-10-25 | Tối ưu đọc graph trước khi khởi chạy lại 15 ô | Người dùng dừng launcher 3 MLP; Codex chuyển 3 checkpoint/log dở sang thư mục `*-stopped-before-loader-20261025`, không xóa. Đọc HDF5 bằng memmap read-only cho dataset liên tục, fallback sang HDF5 với dataset nén; có tùy chọn worker nhưng launcher đặt 0 vì benchmark 4.096 graph: 1,03 giây với memmap/0 worker, 3,36 giây với HDF5 giữ mở/0 worker, 3,69/3,52 giây với memmap/2 hoặc 4 worker. MLP, GAT, GAT+KG smoke CPU 1 epoch có history, model tensors và sampler RNG khớp chính xác trước/sau đổi loader. Full suite 187 tests, lint/format, diff check và dry-run 15 ô tối đa 5 song song PASS. Chưa chạy PredCls GPU với code mới | Người dùng commit code/tài liệu rồi khởi chạy lại cả 15 ô từ đầu với `--max-parallel 5`; theo dõi VRAM, không dùng checkpoint cũ khác code hash |
| 2026-10-25 | Dọn ba run MLP dở theo yêu cầu người dùng | Xóa đúng 3 thư mục checkpoint đã archive, 3 log, 3 thư mục W&B local và 3 symlink trỏ vào chúng; output `runs/level1-clip-v1/` và `logs/level1-clip-v1/` dành cho ma trận mới vẫn trống. Không xóa dữ liệu, cache CLIP, KG export hay W&B run trên web | Người dùng commit thay đổi loader rồi chạy mới đủ 15 ô |
| 2026-10-25 | So sánh tạm thời 9 run CLIP đã xong với annotation-only | Từ history tại best epoch, mR@50 trung bình 3 seed của MLP semantic/GAT/KG-on lần lượt 17,3703%/19,9766%/18,8485%; report annotation-only độc lập tương ứng 17,6702%/19,7963%/19,7542%, delta −0,2999/+0,1803/−0,9057 điểm %. R@50 CLIP tương ứng 62,9065%/63,2434%/63,6230%, đều thấp hơn annotation-only 64,2616%/64,2233%/64,1446%. Cùng VG150 val 5.000 ảnh và cùng seed, nhưng CLIP chưa được evaluate độc lập; KG-on còn đổi TransE/C.2 sang filtered v2, MLP đổi patience 10→15, tất cả đổi validation batch 128→512. KG-off và edges-off còn chạy nên chưa kết luận ablation. | Chờ đủ 15 run và người dùng chạy stage evaluate GPU độc lập; báo cáo paired delta với cùng protocol và ghi rõ các thay đổi ngoài CLIP |
| 2026-10-25 | Phân tích 13/15 run CLIP đã hoàn tất | MLP/GAT/KG-on/KG-off đủ 3 seed; edges-off mới xong seed 42. Mean ± sample std mR@50 từ history tại best epoch: MLP 17,3703 ± 1,1182%, GAT 19,9766 ± 0,4749%, KG-on 18,8485 ± 0,4006%, KG-off 19,6284 ± 0,2497%. KG-on trừ KG-off theo seed 42/123/3407 là −0,6106/−0,6001/−1,1289 điểm %, mean −0,7799 điểm %; R@50 lại cao hơn trung bình +0,4604 điểm %. Kiểm config cùng seed: dữ liệu, cache CLIP, KG manifest và siêu tham số trùng; chỉ `use_knowledge` khác. Edges-off seed 42 mR@50=18,7001%, thấp hơn KG-on seed 42 0,5758 điểm %, chưa đủ 3 seed. Đây là history validation, chưa phải evaluate độc lập. | Chờ edges-off seed 123/3407 dừng, rồi người dùng chạy evaluate GPU cho đủ 15 `best.pt`; xem per-predicate report trước khi giải thích chênh lệch R/mR |
| 2026-10-25 | Đủ 15/15 run CLIP, so sánh theo yêu cầu mức 1 | Không còn trainer; 15 `best.pt` có epoch trùng best mR@50 trong history, cùng hash dữ liệu/cache CLIP, train 57.723 và val 5.000 ảnh. Mean ± sample std mR@50: MLP 17,3703 ± 1,1182%; GAT 19,9766 ± 0,4749%; KG-on 18,8485 ± 0,4006%; KG-off 19,6284 ± 0,2497%; edges-off 19,6743 ± 1,1601%. GAT hơn MLP +2,6063 điểm % ở cả 3 seed. KG-on trừ KG-off −0,7799 điểm % ở cả 3 seed. KG-on trừ edges-off −0,8258 điểm % trung bình nhưng dấu không nhất quán theo seed (+0,5759/−2,1696/−0,8837). R@50 và mR@100/R@100 được tính cùng history; chưa có report evaluate độc lập hoặc 5 đúng/5 sai. | Người dùng chạy `myenv/bin/python -m src.level1_visual_matrix evaluate --device cuda --max-parallel 1` để tạo 15 report và summary, rồi phân tích từng predicate và 5 dự đoán đúng/5 sai; không dùng test để chọn mô hình |
| 2026-10-25 | Evaluate độc lập đủ 15 checkpoint CLIP | Người dùng chạy stage GPU; đủ 15 report `clip-v1-*-val5000.json` và `clip-v1-matched-3seed-val5000-summary.json`, mỗi report full validation 5.000 ảnh và cùng protocol/hash. Bốn metric R@50/100, mR@50/100 trùng chính xác history tại best epoch cho cả 15 run. GAT hơn MLP +2,6063 điểm % mR@50; KG-on kém KG-off −0,7799 điểm % ở cả 3 seed; KG-on kém edges-off −0,8258 điểm % trung bình nhưng seed 42 ngược dấu. Theo per-predicate R@50 trung bình 3 seed, KG-on hơn KG-off ở 11/49 lớp, kém ở 20 lớp và bằng ở 18 lớp; ví dụ `on` +1,26 điểm % (support 11.911), `behind` −5,16 (517), `under` −4,91 (231). Đây vẫn là validation dùng chọn best checkpoint, chưa phải held-out test. | Chọn checkpoint đại diện và kiểm tra thủ công ít nhất 5 dự đoán đúng/5 sai trên JPEG; khóa protocol trước khi đánh giá test hoặc bước mức 2 |
| 2026-10-25 | Chuẩn bị 5 đúng/5 sai với checkpoint CLIP | Bổ sung đọc và xác minh cache CLIP cho `scripts.extract_level1_examples.py`; CPU smoke trên GAT seed 42, 100 ảnh validation tạo `reports/level1/clip-v1-gat-seed42-example-candidates-val100.json` với 5 hit và 5 miss R@50. Đã xem 5 JPEG tương ứng; có miss `has`→`with` và `above`→`on` có thể là biến thể ngữ nghĩa, nên không coi mọi miss metric là lỗi thị giác. File JSON giữ `human_analysis` trống để người trong nhóm xác nhận thủ công. Không train/evaluate GPU thêm. | Teammate xác nhận hộp bao/quan hệ trên 5 ảnh, hoàn thiện ghi chú từng case và đưa vào báo cáo/demo |
| 2026-10-25 | Trang review trực quan 5 đúng/5 sai | Thêm `tools/render_level1_example_review.py` và tạo `reports/level1/clip-v1-gat-seed42-example-review.html` gồm 10 thẻ: JPEG gốc, box subject/object, GT, dự đoán, rank và lý do hit/miss theo R@50. Đây là trang review từ 5 ảnh có cả hit và miss; chưa thay thế xác nhận ngữ nghĩa bằng người trong nhóm. | Người dùng mở HTML, đối chiếu từng box/quan hệ và ghi nhận ví dụ nhãn đồng nghĩa hoặc mơ hồ trong phần phân tích báo cáo |
| 2026-10-25 | Bản PNG cho trình xem file | Giao diện người dùng mở HTML như text, nên thêm `tools/render_level1_review_png.py` để xuất hai contact sheet PNG, mỗi ảnh có 5 case với box đỏ/xanh, GT và dự đoán. Đã kiểm bằng trình xem ảnh; output `reports/level1/clip-v1-gat-seed42-review-hits.png` và `...-misses.png`. | Người dùng mở PNG trực tiếp và đánh dấu các quan hệ/box hợp lý hoặc mơ hồ; sau đó hoàn thiện phân tích 5 đúng/5 sai |
| 2026-10-25 | Ghi nhận xét sơ bộ 10 triplet | Thêm `reports/level1/clip-v1-gat-seed42-qualitative-review.md`: 5 hit/5 miss theo R@50 trên 5 ảnh validation, tách khớp GT khỏi hợp lý ngữ nghĩa. Các miss `has`→`with`, `above`→`on` có thể do cách diễn đạt; `near`→`with` là lỗi quan hệ rõ hơn. Không nhận đây là đánh giá thủ công của người trong nhóm; không suy tỷ lệ lỗi từ mẫu nhỏ. | Thành viên nhóm xác nhận bảng ảnh/box và dùng nhận xét đã kiểm trong báo cáo |
| 2026-10-25 | Kiểm khả năng thử CLIP + ConceptNet sau checkpoint Git | Snapshot ConceptNet 5.7 đã có; graph B.2 gồm 46.414 entities, 56.703 triples nhưng metadata `ready_for_gat_kg_training=false`, 150 dòng review queue chưa có quyết định/người duyệt. Bản B.2.2 bridge cũng là candidate chưa duyệt. `src/knowledge/export.py` hiện chỉ xuất C.2 WordNet; chưa có bảng embedding ConceptNet [151,64] được kiểm class ID. Vì vậy chưa có lệnh GPU CLIP+ConceptNet công bằng và hợp lệ để chạy ngay. | Sau khi người dùng commit kết quả CLIP hiện tại, chuẩn bị mapping ConceptNet được review, cổng KG/TransE và C.2 riêng; CPU smoke và dry-run trước khi gửi lệnh GPU |
| 2026-10-25 | Chuẩn bị nhánh CLIP + ConceptNet reviewed sau commit `81703ab` | Tạo `data/knowledge/conceptnet-review-v1/review.csv` 150 dòng **trống** và 1.705 evidence mẫu; 26 nhãn ưu tiên mơ hồ, 112 nhãn có nhiều URI còn lại, 12 nhãn chuẩn. Script `prepare_conceptnet_review` kiểm URI, assertion, reviewer và hash nguồn; compiler version riêng chỉ chạy khi 150 quyết định đầy đủ. Cổng C.1 ConceptNet khóa CSV/class link/manifest, C.2 xuất vector đúng class ID với mask 0 cho reject/defer, GAT+KG nhận nguồn ConceptNet. Ranking TransE được vector hóa để giảm hàng trăm nghìn batch nhỏ mỗi epoch. CPU preflight graph thật: split transductive 10%/10% thất bại, 8%/8% PASS (47.631/4.536/4.536); 192 tests PASS. Chưa chạy GPU. | Người trong nhóm review `review.csv` theo [hướng dẫn](knowledge/conceptnet_review_and_runs.md); sau đó compiler và CPU dry-run, người dùng chạy TransE GPU, export CPU, rồi 6 run PredCls GPU cần thiết và evaluate. Không dùng bridge WordNet↔ConceptNet cho nhánh này |
| 2026-10-25 | Chuẩn bị đề xuất review ConceptNet theo yêu cầu của Toan | Tạo `data/knowledge/conceptnet-review-v1/review_suggestions.ai-draft.csv` từ graph ConceptNet 5.7 và mapping WordNet đại diện đã khóa hash. Đề xuất 121 accept, 27 defer, 2 reject (`pant`, `short`), kèm assertion URI và lý do từng dòng; `review_focus.ai-draft.csv` lọc 29 dòng cần chú ý. `proposed_by=Codex_AI_provisional`, `requested_human_approver=Toan`; file `review.csv` chính thức vẫn 150 pending. Có cổng batch approval riêng: chỉ sau tuyên bố của Toan khóa hash mới tạo review version khác, ghi rõ AI draft được duyệt theo lô và `human_verified_class_count=0`. | Toan xem file đầy đủ và 29 dòng focus; nếu đồng ý, phê duyệt bản AI draft theo lô. Chưa tạo approval thay người dùng, chưa chạy GPU |
| 2026-10-25 | Kiểm lại 27 nhãn ConceptNet bị hoãn | Xem mọi cạnh được giữ lại của từng URI ứng viên, tạo bản AI v2 riêng tại `data/knowledge/conceptnet-review-ai-v2/`: thêm 13 đề xuất `accept` có URI/assertion/rationale cụ thể (`branch`, `cap`, `handle`, `orange`, `plane`, `plant`, `player`, `post`, `screen`, `ski`, `surfboard`, `tail`, `wing`). Tổng mới 134 accept, 14 defer, 2 reject. Bản v1/hash cũ giữ nguyên; bản report focus giờ có 16 nhãn. Chưa gán `reviewer=Toan`, chưa phê duyệt hay chạy GPU. | Toan xem [báo cáo 13 đã xét/16 còn cân nhắc](../reports/level1/conceptnet-ai-review-focus.md) và bản đề xuất 150 dòng; nếu đồng ý bản v2, phê duyệt theo lô để compiler ghi đúng provenance |
| 2026-10-25 | Toan chốt review ConceptNet 150 class | Toan duyệt 134 đề xuất AI v2 và trực tiếp chọn `accept` ID 11, 54, 123, 143; `reject` 12 ID focus còn lại. Tạo hồ sơ v3 riêng (`data/knowledge/conceptnet-review-toan-v3/`) có lời quyết định nguyên văn, bốn URI/assertion do Codex chọn, 16 quyết định trực tiếp và dấu vết 134 đề xuất AI; reviewer=Toan, basis `human_mixed_direct_and_ai_batch`. Hai file nguồn nhỏ `reviews/conceptnet-toan-v3/approval.json` và `selection.csv` được theo dõi bằng Git để teammate tái dựng dữ liệu bị ignore. Tổng 138 accept, 12 reject, 0 defer/pending. Compiler `data/knowledge/conceptnet-reviewed-toan-v3/` và CPU TransE dry-run PASS (47.631/4.536/4.536); 194 tests PASS. `bike`/`glass` được accept theo chỉ đạo nhưng URI trộn nghĩa; đây là rủi ro khi giải thích vector. Chưa chạy GPU, chưa commit. | Toan chạy TransE ConceptNet GPU theo [hướng dẫn](knowledge/conceptnet_review_and_runs.md); sau checkpoint, Codex có thể export C.2 CPU rồi gửi lệnh 6 run CLIP+ConceptNet GPU |
| 2026-10-25 | TransE ConceptNet v3 đã có checkpoint, C.2 CPU export hoàn tất | Sau commit `e0bfe42`, thấy `runs/transe/conceptnet-toan-v3-seed42/` đã kết thúc ở epoch 25 (early stop), chọn best epoch 5 theo val MRR=0,01567; test MRR=0,01321 trên 512 triple mẫu (1.024 truy vấn head/tail), chỉ là metric KG link prediction. `scripts.export_conceptnet` CPU tạo `data/knowledge/c2-conceptnet-toan-v3/` với tensor [151,64] và 138 valid bit; bốn class Toan accept đều valid, 12 class reject đều invalid. Cổng nguồn/review/checkpoint PASS. Chưa chạy bất kỳ GPU train/eval nào từ Codex. | Toan chạy 6 ô CLIP+ConceptNet `kg_on` và `edges_off` × 3 seed bằng launcher; sau đó chạy GPU evaluate trên validation theo [hướng dẫn](knowledge/conceptnet_review_and_runs.md) |
| 2026-10-25 | Đủ 21/21 run CLIP và full validation table | Toan hoàn tất 6 run ConceptNet và GPU evaluate; Codex đối chiếu 6 report mới với `best.pt` (metric khớp tuyệt đối), cùng 5.000 ảnh validation, cùng data/cache/protocol với 15 run CLIP cũ. [Bảng đầy đủ](../reports/level1/clip-all-21run-val5000-summary.md) ghi 21 dòng từng seed và 7 điều kiện mean±SD: ConceptNet KG-on R@50/mR@50=63,67/19,66; WordNet KG-on=63,62/18,85; KG-off=63,16/19,63; GAT=63,24/19,98. ConceptNet KG-on hơn WordNet KG-on +0,82 điểm mR@50, nhưng chỉ hơn KG-off +0,04 và kém GAT 0,31; chưa chứng minh KG giúp baseline không KG một cách vững. Đây là validation dùng chọn checkpoint, chưa phải held-out test. | Xem per-predicate và lỗi định tính trước khi chọn mô hình; nếu thử nhúng/fusion khác, giữ nguyên dữ liệu/CLIP/protocol và đánh giá 3 seed mới |
| 2026-10-25 | Chuẩn bị thử RotatE trên ConceptNet v3 | Thêm trainer dùng score PyG RotatE và negative sampler có lọc train, ranking filter cùng TransE; C.2 ghép real/imag 32+32 thành 64D L2-normalized, mask 138/150; launcher riêng ba run CLIP+RotatE KG-on (seeds 42/123/3407), `edges_off` tùy chọn. CPU dry-run graph thật PASS split 47.631/4.536/4.536 và 512 triple mẫu mỗi val/test; ba split khớp từng triple với run TransE ConceptNet seed42; full suite 195 tests PASS. Chưa chạy GPU hoặc export thực vì chưa có checkpoint RotatE. | Toan chạy một lệnh GPU train KG trong [hướng dẫn](knowledge/rotate_conceptnet_workflow.md); sau đó export CPU, ba run PredCls GPU và evaluate validation |
| 2026-10-25 | Audit RotatE v1 sau run GPU của Toan | `runs/rotate/conceptnet-toan-v3-seed42/` dừng epoch 21, best epoch 1, val MRR=0,00455, test MRR=0,00544 (mỗi split mẫu 512 triple); TransE cùng split val/test MRR=0,01567/0,01321. RotatE train loss 0,991→0,041 nhưng val MRR tụt khoảng 0,0005, Hits@1 luôn 0; median norm entity 0,071→0,847 và mean positive distance train/val ở epoch 21 khoảng 0,137/1,176 trên 1.024 triple mẫu. Dấu hiệu overfit do v1 dùng một negative + hinge loss; evaluator đã có test so khớp candidate reference. Giữ nguyên artifacts v1, không export hoặc chạy PredCls từ checkpoint này. | Thử v2 theo objective RotatE gốc; chỉ cân nhắc PredCls khi KG validation có tín hiệu |
| 2026-10-25 | Chuẩn bị RotatE v2 tách biệt | Thêm complex L1 score, khởi tạo entity có scale theo margin, 32 filtered negatives/positive, self-adversarial logistic loss; giữ split/filter KG và seed 42, thư mục run/C.2/PredCls v2 mới. Export từ chối checkpoint RotatE v1; CPU preflight graph thật, 196 tests, ruff/black/isort và diff check PASS. Đây là phương án có cơ sở từ công bố RotatE, chưa có kết quả GPU v2. | Toan chạy lệnh GPU v2 trong [hướng dẫn](knowledge/rotate_conceptnet_workflow.md); Codex đọc val MRR trước khi export C.2 |
| 2026-10-25 | RotatE v2 KG hoàn tất, C.2 đã export trên CPU | Toan chạy GPU v2, early stop ở epoch 76. Validation MRR tốt nhất theo ngưỡng `min_delta=0,0001` tại `best.pt` epoch 36: 0,0132415; epoch 37 có MRR thô 0,0132743 nhưng mức tăng dưới `min_delta` nên không thay checkpoint. Đến epoch 76 train loss 0,4893 nhưng val MRR giảm còn 0,00643: overfit sau đỉnh. TransE ConceptNet cùng split val MRR=0,0156735. CPU export C.2 v2 `[151,64]`, 138/150 class valid, invalid rows zero, valid rows norm 1; launcher ba ô PredCls dry-run PASS. KG link prediction không dự đoán chắc kết quả PredCls. | Toan có thể chạy ba ô CLIP+RotatE `kg_on` sau khi commit tài liệu cập nhật; đánh giá validation rồi so paired với ConceptNet+TransE, KG-off và GAT |
| 2026-10-25 | Chuẩn bị nhánh ComplEx ConceptNet theo đề xuất của Toan | Dùng PyG ComplEx, filtered logistic loss với 32 negative/positive và L2 regularization; ranking head/tail cùng split/filter C.1, export real+imag 32+32 thành C.2 64D với mask 138 class, launcher riêng ba ô CLIP+ComplEx KG-on. KG train và PredCls output version riêng; không chạy GPU/không tự commit. CPU preflight graph thật PASS, train/val/test split khớp từng triple với TransE và RotatE v2; 197 tests, ruff/black/isort/diff check và launcher dry-run PASS. Lệnh teammate ở [hướng dẫn](knowledge/complex_conceptnet_workflow.md). | Toan commit code/tài liệu, rồi chạy một run ComplEx KG GPU; Codex kiểm val MRR và export C.2 CPU trước ba run PredCls GPU |

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
PyTorch CPU/CUDA, dependencies, annotation VG150, split, WordNet corpus và
lệnh smoke. Ma trận CLIP cần đủ 89.169 JPEG; lệnh tải hai archive mirror,
manifest/hash và lệnh cache được ghi trong hướng dẫn teammate và
[level1_clip_regions.md](experiments/level1_clip_regions.md). Máy hiện tại đã
tải và xác minh toàn bộ; máy teammate chưa được kiểm chứng bằng lượt tải mới.

## Bước tiếp theo do người dùng thực hiện

- 15 run CLIP+WordNet và 15 lượt evaluate validation đã hoàn tất; summary và
  ví dụ 5 đúng/5 sai ở `reports/level1/`.
- Người trong nhóm điền 150 quyết định class→ConceptNet trong
  `data/knowledge/conceptnet-review-v1/review.csv`, xem bằng chứng ở file
  `evidence_samples.csv` cùng thư mục. Lệnh kiểm và trình tự TransE/PredCls ở
  [conceptnet_review_and_runs.md](knowledge/conceptnet_review_and_runs.md).
- Chỉ sau review hoàn chỉnh mới biên dịch KG ConceptNet version riêng và chạy
  TransE GPU, export C.2 CPU, 6 run PredCls GPU và evaluate trên 5.000 ảnh val.
- Khóa pipeline trước khi đánh giá VG150 test; hoàn thiện báo cáo, slide và
  notebook/demo từ các report đã có.
- Teammate tải lại annotation và JPEG theo
  [TEAM_SETUP_LEVEL1.md](TEAM_SETUP_LEVEL1.md); mapping reviewed có lệnh tái lập
  ở [c2_gat_wordnet.md](knowledge/c2_gat_wordnet.md).

Không commit dữ liệu tải về, checkpoint, credentials hoặc đường dẫn riêng của máy.
