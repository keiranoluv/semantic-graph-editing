# Mã nguồn và kết quả thực nghiệm gốc

Bản gốc được giữ nguyên tại commit [`47dea69a`](https://github.com/keiranoluv/semantic-graph-editing/tree/47dea69aaff481ce2121e3c7dde42018442d2e87). Snapshot gồm mã huấn luyện, công cụ pilot/audit/review, cấu hình, kết quả theo lượt, notebook dựng hình và nguồn LaTeX.

Mã đang sử dụng nằm trong [submission/](../submission/README.md). Các bảng kết quả chính nằm trong [results/](../results/README.md).

Để mở bản gốc trong một thư mục riêng từ bản clone có đầy đủ lịch sử:

```bash
git worktree add --detach ../semantic-graph-editing-research 47dea69aaff481ce2121e3c7dde42018442d2e87
```

Các launcher gốc có kiểm tra hash và đường dẫn đã khóa. Khi tái lập lượt cũ, dùng hướng dẫn và cấu hình tại snapshot đó cùng bộ dữ liệu/checkpoint ngoài Git tương ứng. Snapshot mã nguồn không chứa các artifact lớn trong data/ và runs/.

[Thông tin xác minh snapshot](snapshot.json) ghi commit, số tệp và SHA-256 của bản ZIP lưu cục bộ. Trong workspace phát triển, các tệp lịch sử vẫn giữ nguyên tại đường dẫn cũ và được Git bỏ qua.
