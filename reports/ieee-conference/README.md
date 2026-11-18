# Báo cáo đồ án theo mẫu IEEE Conference

Bản dàn trang ngày 18/11/2026 kế thừa nguồn ảnh tĩnh ở
`latex_report_source/`, thêm bảng test Custom GQA và phần mở rộng PVSG đã
nghiệm thu. Dùng lớp chuẩn `IEEEtran` 1.8b ở chế độ `conference,a4paper`.
Chưa tìm thấy file mẫu Seminar riêng của lớp trong repository hay các ZIP báo cáo.
Mục 8 của `Project01_Graph_Editing.pdf` yêu cầu khoảng 12–18 trang; thông báo
ngày 18/11 cho phép đặt phụ lục sau tài liệu tham khảo và loại phần đó khỏi giới hạn.

## Biên dịch

Ở thư mục chứa `main.tex`, dùng TeX Live với IEEEtran (gói
`texlive-publishers` trên Ubuntu), XeLaTeX, Biber, biblatex-ieee, PGFPlots,
adjustbox, xurl và phông TeX Gyre Termes.

```sh
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

Trên Overleaf, nhập ZIP nguồn, đặt tài liệu chính là `main.tex` và chọn
XeLaTeX. Khi build trong repo, LaTeX tìm hai ảnh VG150 trong
`latex_report_source/figures/` và timeline trong `reports/pvsg-trd-m5-v1/`.
Trong ZIP, các ảnh được đóng gói cùng nguồn để có thể build độc lập.

## Số liệu và phần cần nhóm xác nhận

- Phần ảnh tĩnh kế thừa nội dung và số liệu từ bản báo cáo kỹ thuật. Danh mục bằng chứng và các việc cần hoàn thiện của bản đó nằm ở `latex_report_source/SOURCE_MAP.md` và `latex_report_source/TODO.md`; hai tệp cũng đi kèm ZIP.
- Bảng Custom GQA test lấy từ `reports/level2/gqa-test-v1/summary.json`, kiểm với `completion-audit.json`: 27 checkpoint, 10.696 ảnh test của đồ án. Các bảng validation trong cùng mục giữ đúng nhãn split.
- Bảng PVSG chính lấy từ `reports/pvsg-trd-m5-v1/final.md`. Phụ lục dùng `report.md` và `case_timelines.png` cùng thư mục. Ba seed mỗi mô hình, 17 lượt gốc và 7 lượt kiểm tra hội tụ muộn. Các quy tắc giải mã và chấm nằm ở `experiments/pvsg_trd_v1/protocol.json`.
- Phân công từng người, các công cụ từng người sử dụng, nguồn/quyền sử dụng ảnh và các minh chứng kiểm tay còn đánh dấu trong báo cáo cần nhóm xác nhận trước khi nộp.
