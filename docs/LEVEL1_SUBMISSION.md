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
   ảnh hiện chưa dùng, cần nêu lý do lựa chọn đơn giản hóa. PDF mục 4.1 đề xuất
   vùng ảnh nhưng cho phép input khác; trang 3 cho phép điều chỉnh kiến trúc
   nếu giải thích cơ chế. Xác nhận với giảng viên nếu xem đây là thay đổi lớn.
4. **Tri thức nền (2 trang).** WordNet noun senses, graph, C.1 TransE và C.2
   `[151,64]` features với 150/150 class từ mapping đại diện Toan xác nhận
   chính thức. Mô tả quy trình tại
   [LEVEL1_MAPPING_REVIEW.md](knowledge/LEVEL1_MAPPING_REVIEW.md), ghi rõ xác
   nhận theo lô, thiếu lý do riêng cho phần lớn class và giới hạn nhãn đa nghĩa.
5. **Protocol (1–2 trang).** Cùng data split, 3 seeds 42/123/3407, chọn
   checkpoint bằng validation mR@50, R@50/100 và mR@50/100, tính mean/sample
   std. Đưa cả ablation tắt KG và bỏ cạnh giữa đối tượng. Phân biệt `--no-knowledge`
   cùng kiến trúc với GAT baseline gốc.
6. **Kết quả (2–3 trang).** Bảng dưới đây là đánh giá độc lập đủ 5.000 ảnh
   validation cho mỗi checkpoint, ba seed mỗi điều kiện. Thêm per-predicate/
   long-tail và kết quả test chỉ sau khi chốt pipeline final.
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
| GAT+WordNet | 64.14 ± 0.05 | 66.25 ± 0.15 | 19.75 ± 0.67 | 21.89 ± 0.59 |
| Cùng kiến trúc, KG off | 64.32 ± 0.20 | 66.41 ± 0.17 | 19.16 ± 0.41 | 21.15 ± 0.25 |
| GAT+WordNet, no inter-object edges | 64.77 ± 0.21 | 66.77 ± 0.14 | 19.56 ± 0.38 | 21.52 ± 0.32 |

Số đầy đủ: [level1_baseline_verified.md](experiments/level1_baseline_verified.md).
Cả 9 report mới và summary nằm ở `reports/level1/wordnet-reviewed-*-val5000.json`
và `reports/level1/wordnet-reviewed-ablation-3seed-val5000-summary.json`.
KG-on hơn KG-off 0,59 điểm % mR@50 trung bình, nhưng không hơn GAT baseline
gốc (19,75% so với 19,80%); lợi thế so với bỏ cạnh chỉ 0,19 điểm % và đảo
chiều ở seed 3407. Đây là pipeline annotation-only, chưa có đặc trưng vùng
ảnh; TransE hiện tại còn hạn chế bộ lọc triple KG test lúc train.
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
- Có nguồn lựa chọn 150 synset của Toan, approval declaration và manifest
  reviewed batch gắn hash KG; báo cáo nêu giới hạn thiếu lý do riêng từng class.
- Đánh giá full validation; chỉ chạy test cuối sau khi đóng băng thiết kế.
- Có 5 đúng/5 sai được người thật kiểm tra ảnh và giải thích.
- Notebook chạy được trên máy nhóm; đường dẫn JPEG hoặc demo không ảnh được
  công bố rõ. Environment, data, commands và nguồn ảnh/tri thức kèm hash.
