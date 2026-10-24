# Bộ nộp mức 1 — bản chuẩn bị

Nguồn yêu cầu: `Project01_Graph_Editing.pdf`, trang 3–6. File này tập hợp
nội dung đã kiểm chứng và những chỗ nhóm cần điền sau train GPU/review.
Không coi các ô trống là kết quả đã có.

## Báo cáo kỹ thuật (mục tiêu 12–18 trang)

1. **Bài toán và phạm vi (1 trang).** VG150 PredCls: cho trước nhãn/hộp,
   dự đoán predicate; mức 1 tập trung vào mô hình đồ thị và tri thức nền.
   Giải thích vì sao không huấn luyện detector hoặc generator.
2. **Dữ liệu/split (2 trang).** HDF5/vocabulary/metadata và hash ở
   [TEAM_SETUP_LEVEL1.md](TEAM_SETUP_LEVEL1.md). Split theo ảnh 57.723 / 5.000 /
   26.446, `box_policy=clip`; nêu giới hạn so sánh benchmark.
3. **Mô hình (2–3 trang).** Geometry MLP 5D; Semantic MLP thêm embedding nhãn
   học được 32D; GAT truyền tin trên fully-connected directed topology không
   đọc predicate GT; GAT+WordNet thêm TransE 64D và valid bit. Đặc trưng vùng
   ảnh hiện chưa dùng, cần nêu lý do lựa chọn đơn giản hóa.
4. **Tri thức nền (2 trang).** WordNet noun senses, graph, C.1 TransE và C.2
   `[151,64]` features; 124/150 coverage của draft AI. Mô tả quy trình người
   kiểm tra tại [LEVEL1_MAPPING_REVIEW.md](knowledge/LEVEL1_MAPPING_REVIEW.md),
   số dòng thực tế được chấp nhận/defer sau khi nhóm điền CSV. Không gọi draft
   là bản đã kiểm duyệt.
5. **Protocol (1–2 trang).** Cùng data split, 3 seeds 42/123/3407, chọn
   checkpoint bằng validation mR@50, R@50/100 và mR@50/100, tính mean/sample
   std. Đưa cả ablation tắt KG và bỏ cạnh giữa đối tượng. Phân biệt `--no-knowledge`
   cùng kiến trúc với GAT baseline gốc.
6. **Kết quả (2–3 trang).** Bảng baseline dưới đây đã kiểm tra lại. Thêm
   GAT+WordNet và ablation sau khi người dùng chạy GPU; nếu làm draft thì đánh
   dấu rõ. Thêm per-predicate/long-tail khi dữ liệu thực nghiệm đủ.
7. **Phân tích lỗi (1–2 trang).** Công cụ trích case đã tạo 5 hit/5 missed mẫu
   ở [level1_case_candidates.md](experiments/level1_case_candidates.md).
   Thay hoặc bổ sung case GAT+KG final, xem JPEG và ghi nguyên nhân thực tế.
8. **Tái lập, giới hạn và đạo đức dữ liệu (1–2 trang).** Môi trường, lệnh tải,
   hash, hạn chế mapping, split/protocol, khai báo mã nguồn mở/AI theo PDF.
   Nêu rõ không dùng relation annotation validation/test để dựng KG.

### Bảng kết quả đã có (full validation, %, 3 seeds)

| Model | R@50 | R@100 | mR@50 | mR@100 |
|---|---:|---:|---:|---:|
| Geometry MLP | 49.00 ± 0.08 | 51.59 ± 0.08 | 4.95 ± 0.06 | 5.47 ± 0.08 |
| Semantic MLP | 64.26 ± 0.09 | 66.33 ± 0.02 | 17.67 ± 0.27 | 19.76 ± 0.34 |
| GAT | 64.22 ± 0.14 | 66.35 ± 0.25 | 19.80 ± 1.03 | 22.00 ± 1.07 |
| GAT+WordNet | Chờ 3 GPU runs | Chờ | Chờ | Chờ |
| Cùng kiến trúc, KG off | Chờ 3 GPU runs | Chờ | Chờ | Chờ |
| GAT+WordNet, no inter-object edges | Chờ 3 GPU runs | Chờ | Chờ | Chờ |

Số đầy đủ: [level1_baseline_verified.md](experiments/level1_baseline_verified.md).
Các checkpoint baseline cũ không ghi hash split lúc train/max_epochs; nêu
giới hạn này trước khi kết luận chênh lệch nhỏ có ý nghĩa.

## Slide và demo vấn đáp

Giới hạn theo PDF: tối đa 20 phút trình bày **gồm slide và demo**, sau đó hỏi
đáp. Gợi ý phân bổ: 12–15 phút slide, 3–5 phút notebook/demo.

Slide nên có: câu hỏi nghiên cứu; VG150 PredCls và leak-safe split; sơ đồ
MLP → GAT → GAT+KG; WordNet mapping/review; bảng 3 seeds; ablation; một hit
và một missed được nhóm xác minh; giới hạn/kết luận; phân công.

Demo: khởi động `myenv`, mở [notebook mức 1](../notebooks/level1_scene_graph_demo.ipynb)
với checkpoint đã kiểm thử và ảnh JPEG nếu đã tải. Chuẩn bị video dự phòng
≤5 phút như PDF. Notebook hiện minh họa dự đoán quan hệ cho mức 1. Mục nộp
chung trong PDF có ví dụ câu chỉnh sửa và thao tác đồ thị, trong khi mô hình
thao tác thuộc mức 2; nhóm cần hỏi giảng viên phạm vi demo mức 1 trước ngày nộp.

## Phân công cần người thật điền

| Thành viên | Dataset/split | MLP/GAT/KG | Review mapping | Thực nghiệm/đánh giá | Báo cáo/demo | Commit/minh chứng |
|---|---|---|---|---|---|---|
| Tên 1 |  |  |  |  |  |  |
| Tên 2 |  |  |  |  |  |  |
| Tên 3 |  |  |  |  |  |  |

Điền theo đóng góp thực tế, giữ lịch sử commit và bản phân công trong báo cáo.
Không tự điền tên người kiểm tra WordNet hay tuyên bố họ đã kiểm tra.

## Chốt trước nộp

- Có 3 seeds GAT+KG và ablation, cùng split/protocol; hash và config đầy đủ.
- Có review mapping được ký tên, số accept/defer rõ; kết quả draft và reviewed
  được đặt tên đúng trạng thái. Nếu chưa có reviewed, báo cáo không gọi kết quả
  GAT+KG là đã kiểm chứng nguồn tri thức.
- Đánh giá full validation; chỉ chạy test cuối sau khi đóng băng thiết kế.
- Có 5 đúng/5 sai được người thật kiểm tra ảnh và giải thích.
- Notebook chạy được trên máy nhóm; đường dẫn JPEG hoặc demo không ảnh được
  công bố rõ. Environment, data, commands và nguồn ảnh/tri thức kèm hash.
