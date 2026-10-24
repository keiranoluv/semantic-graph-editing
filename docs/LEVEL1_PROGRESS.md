# Theo dõi Project 01 — Mức 1

Cập nhật: 2026-10-24.

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
| 2. C.2 — Xuất embedding | Chuyển checkpoint TransE thành bảng đặc trưng đúng class ID VG150, kèm mask và provenance | Draft hoàn thành; reviewed chưa triển khai | 10 tests export PASS, artifact thật [151,64], 124/150 class được chọn nghĩa; cả 150 class có gợi ý trong worksheet v2 |
| 3. GAT+WordNet | Tích hợp KG vào đặc trưng đỉnh, có cấu hình và run riêng | Code draft và CPU smoke hoàn thành | 3 chế độ KG-on/KG-off/edges-off train+val CPU thành công; baseline được bảo toàn |
| 4. Thực nghiệm | So sánh cùng protocol với 3 seeds; ablation tri thức/cạnh | 9 baseline đã đánh giá lại full validation; aggregator có; chờ GAT+KG GPU | Chưa có kết quả GAT+KG train dài hoặc mean/std cho nhánh KG |
| 5. Phân tích và bộ nộp | 5 đúng/5 sai, notebook, bảng kết quả, hướng dẫn, tài liệu báo cáo/demo | Có case kỹ thuật, notebook đã chạy, bản chuẩn bị báo cáo/slide; phân tích người và kết quả final còn lại | Phân tích thủ công trên model final và bộ nộp hoàn chỉnh |

## Điểm xuất phát và những điều chưa xác minh

- Đã đọc PDF, code GAT, phần nạp KG của TransE và các ghi chú handoff.
- Theo ghi chú baseline, đã có 9 runs MLP/GAT cho seeds 42, 123, 3407;
  các số liệu được ghi là validation, chưa phải đánh giá held-out test cuối cùng.
- Full suite hiện tại có 173 tests PASS; lint/format và pre-commit trên các file
  đã sửa PASS. Chưa báo cáo toàn bộ pre-commit `--all-files` PASS.
- `myenv` ban đầu chỉ có pip; đã cài dependencies và chạy CPU smoke. Sau đó đổi
  sang torch 2.7.0+cu128: CUDA available, nhận RTX 5090; full suite hiện tại 173 tests PASS.
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

- Chọn ưu tiên train draft hay kiểm tra mapping trước khi train dài.
- Đã giải thích mapping và chuẩn bị bảng
  `data/knowledge/level1-human-review-v2/review.csv`; hướng dẫn tại
  [LEVEL1_MAPPING_REVIEW.md](knowledge/LEVEL1_MAPPING_REVIEW.md).
  Chưa coi câu hỏi của người dùng là quyết định chọn train hay review trước.
- Codex đã tiếp tục toàn bộ phần code/CPU/tài liệu sau lời nhắn "tiếp tục".
  Train GPU vẫn do người dùng khởi chạy theo lệnh cụ thể.
- Khi muốn train GPU: chuyển PyTorch trong `myenv` sang CUDA theo hướng dẫn
  thiết lập, rồi chạy seed 42 theo [c2_gat_wordnet.md](knowledge/c2_gat_wordnet.md).
- Gửi output train, lỗi nếu có, và đường dẫn run. Codex tiếp tục đọc kết quả,
  kiểm tra checkpoint/đánh giá và chuẩn bị các seeds/ablation tiếp theo.
- Không đánh dấu bước 4 hoặc 5 hoàn thành trước khi có dữ liệu thực nghiệm,
  phân tích 5 đúng/5 sai và notebook/demo thực tế.

Không commit dữ liệu tải về, checkpoint, credentials hoặc đường dẫn riêng của máy.
