# Báo cáo kỹ thuật — bản viết lại theo kết quả

Build tại thư mục này:

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

Cần XeLaTeX, Biber, TikZ/PGFPlots. Dùng Latin Modern Roman mặc định của XeLaTeX
theo template gốc, không dùng Noto Serif. PDF đầu ra: main.pdf.

## Tổ chức

- Text/1.tex: scorecard duy nhất, quy ước đọc số.
- Text/2.tex: Nhiệm vụ 1 VG150, 54 kết quả test, loss/fusion, long-tail/zero-shot test, case validation.
- Text/3.tex: Nhiệm vụ 2A custom GQA, PE-Net/AlignG.
- Text/4.tex: Nhiệm vụ 2B MagicBrush và EMNLP thay thế.
- Text/5.tex: Nhiệm vụ 3, E/P/G, correlation/error bar, coverage, failure cases.
- Text/6.tex: đối chiếu đề, hạn chế và kết luận.
- SOURCE_MAP.md: artifact nguồn, split và provenance.
- TODO.md: danh sách phần còn thiếu ở Nhiệm vụ 2–3.
- figures/: hai contact sheet VG150; biểu đồ còn lại viết trực tiếp bằng PGFPlots.

Đây là bản làm việc có TODO, chưa phải tuyên bố hoàn tất mọi yêu cầu nộp bài.
Không sửa notebook, model code, checkpoint hoặc manifest thí nghiệm.
