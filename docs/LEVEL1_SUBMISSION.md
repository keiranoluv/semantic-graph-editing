# Bộ bàn giao Mức 1 — bản gọn

Cập nhật 2026-11-11. Theo yêu cầu hiện tại, chỉ chuẩn bị báo cáo Markdown và demo đơn giản; chưa làm slide/PDF hoặc tổng hợp báo cáo kỹ thuật dài.

## Mở các file này trước

1. [Báo cáo kết quả ngắn](LEVEL1_REPORT.md): pipeline, protocol, bảng 54 run, diễn giải và giới hạn.
2. [Phân tích 5 hit / 5 miss có ảnh](../reports/level1/submission/cases/analysis.md): nhận xét AI sau khi quan sát ảnh; người trong nhóm chưa xác nhận.
3. [Phiếu review](../reports/level1/submission/cases/human-review.csv): để nhóm ghi người review, xác nhận hộp/GT và sửa nhận xét.
4. [Notebook demo reference-v1](../notebooks/level1_reference_demo.ipynb): nạp GAT+CLIP, suy luận CPU một ảnh và giải thích ranking/R@50.
5. [So sánh GC và NG cho đủ 54 checkpoint](../reports/level1/reference-comparison/comparison.md): ng-R/ng-mR ở K=50/100, mean/std và chênh lệch theo seed.
6. [Long-tail: head/body/tail trên cả 54 checkpoint](../reports/level1/long-tail-reference-v1/analysis.md): chia nhóm từ train, mR theo nhóm ở K=50/100, mean/std, paired delta và số annotation từng lớp.

7. [Zero-shot trên 9 checkpoint fusion](../reports/level1/zero-shot-reference-v1/analysis.md): tổ hợp chưa gặp trong annotation train, GC/NG@50/100, ví dụ chấm, mean/std và paired delta.

## Phần đã thực hiện

- 27 run chính, 12 run pair/loss và 15 run fusion đã được chấm bằng reference-v1 trên đủ 5.000 ảnh validation, dùng GPU. Có 3 seed, mean/sample std, paired delta và nguồn report/checkpoint.
- Cùng 54 checkpoint đã chấm thêm `reference-ng-v1` bằng GPU, cho phép nhiều predicate trên một cặp. Hai protocol có bảng riêng và bảng đối chiếu; không train/chọn lại checkpoint.
- Sáu run relation prior được để riêng do bất nhất L1/L2; không xóa hay train lại. [Danh sách](../reports/level1/submission/excluded-relation-prior.json).
- 10 case được chọn bằng quy tắc cố định; ảnh có hộp subject/object, GT, dự đoán và thứ hạng. Có kiểm tra tự động hit/miss khớp evaluator.
- Notebook chạy thành công từ đầu tới cuối trên CPU tại workspace hiện tại. Mặc định ảnh 89, GT `shelf → on → table`: dự đoán `on` hạng 64 nên missed@50; R@50 ảnh 70%, R@100 ảnh 90%.
- Bộ test hiện tại: **239 PASS**, gồm kiểm tra phân nhóm chỉ từ train, xử lý lớp vắng và phép trung bình long-tail. Cả hai evaluator đã khớp upstream trên 100 ca tổng hợp + 16 ảnh thật.
- Phân tích long-tail đã hoàn tất cho 54 checkpoint, dùng report có sẵn. Head 10 lớp chiếm 88,40% annotation train; tail 20 lớp chỉ 1,46%. Loss điều chỉnh cải thiện tail ở cả GC/NG; pair fusion KG thật hơn shuffled ở tail NG@50 nhưng chưa hơn baseline không KG.

- Zero-shot đã chấm GPU cho baseline / pair KG thật / pair KG xáo trộn × ba seed, đủ 5.000 ảnh validation. Có 1.130 GT zero-shot trong 722 ảnh. KG thật hơn hai đối chứng ở NG@100 qua ba seed, chưa cải thiện nhất quán ở NG@50/GC. Kiểm evaluator upstream 100 ca tổng hợp + 32 ảnh thật, tái lập metric toàn tập cũ và kiểm số liệu từng ảnh.

## Chạy demo trong khoảng 3 phút

```bash
myenv/bin/jupyter lab notebooks/level1_reference_demo.ipynb
```

Chọn kernel `Python (semantic-graph myenv)`, chạy từng ô bằng Shift+Enter.

- 0:00–0:45: giải thích ảnh, hộp và nhãn object được cho sẵn; 517D gồm hình học và CLIP.
- 0:45–1:30: xem hai hộp, phân biệt input với predicate GT dùng để chấm.
- 1:30–2:30: forward CPU, đọc predicate `on`, hạng 64, giải thích cutoff 50/100.
- 2:30–3:00: đổi `CASE_ID = 'hit01'` rồi chạy lại; phân biệt recall một ảnh với mR toàn tập.

Để giải thích nhiều quan hệ cùng cặp, đổi `CASE_ID = 'miss05'`: GT cặp girl → hair là has, has, with, with. GC chỉ giữ with; NG có with hạng 5 và has hạng 7, tăng recall ảnh từ 4/6 lên 6/6. Cả miss03 và miss05 đã chạy notebook CPU thành công.

Checkpoint và dữ liệu đã có trên máy này; hướng dẫn dựng máy khác ở [TEAM_SETUP_LEVEL1.md](TEAM_SETUP_LEVEL1.md). Chưa kiểm chứng một máy teammate độc lập, chưa quay video vấn đáp và chưa có xác nhận người thật cho 10 case.

## Phần nhóm sẽ tổng hợp sau

- Xác nhận và hoàn thiện 5 đúng/5 sai; không tự động coi nhận xét AI là review của nhóm.
- Báo cáo chính 12–18 trang, slide, phân công và khai báo mã nguồn mở/AI theo đề.
- Chạy thử máy trình bày, video dự phòng ≤5 phút; hỏi giảng viên về phạm vi notebook Mức 1 nếu cần.
- Thông tin nhóm/thành viên chưa điền theo yêu cầu hiện tại.

Chưa cần thêm mô hình để hoàn thành yêu cầu cốt lõi Mức 1. Mọi bảng mới là validation của checkpoint được chọn bằng metric legacy; không dùng như test độc lập và không chọn một seed tốt nhất để đại diện kết quả.

## Kiểm tra trước khi chuyển sang Mức 2

[Biên bản pre-commit](../reports/level1/precommit-2026-11-11.json) ghi lần kiểm lại 239 tests, 108 report GC/NG, long-tail, zero-shot và notebook CPU. Demo reference đã tách ô thiết lập đường dẫn khỏi ô import; kết quả hai case vẫn khớp lần kiểm trước. Notebook walkthrough cũ được ghi rõ là legacy. Các hash notebook trong biên bản lịch sử mô tả phiên bản tại thời điểm đó; hash phiên bản hiện tại ở biên bản pre-commit.
