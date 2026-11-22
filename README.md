# Chỉnh sửa ảnh theo yêu cầu bằng đồ thị cảnh

Đồ án nghiên cứu dự đoán quan hệ giữa các đối tượng, suy luận thao tác chỉnh sửa đồ thị từ câu lệnh và đánh giá ảnh sau chỉnh sửa bằng mức đáp ứng yêu cầu cùng mức bảo toàn cảnh.

## Bắt đầu

- [Notebook tổng hợp đã lưu output](submission/notebooks/project_submission_demo.ipynb): 5 Hit/5 Miss trên VG150 và GQA, ví dụ MagicBrush/He và sáu cặp VN100.
- [Demo trực tiếp](submission/notebooks/graph_editing_live_demo.ipynb): GNN suy luận ba yêu cầu chỉnh sửa trên CPU.
- [Cài môi trường và chạy lại](docs/REPRODUCE.md): chọn artifact chair, live hoặc overview theo phạm vi demo.
- [Dataset VN100](https://huggingface.co/datasets/ndttoan/vn100-graph-editing): 100 ảnh nguồn, 200 câu lệnh và ảnh chỉnh sửa. [Data card](submission/assets/datasets/vn100/README.md).

Hai notebook có sẵn output để xem ngay. Chạy lại demo trực tiếp cần bộ live 412 MiB; chạy lại notebook tổng hợp cần bộ overview 22,05 GiB.

## Thành viên

| Mã học viên | Họ và tên |
|---|---|
| 25C15002 | Hồ Thanh Bình |
| 25C15026 | Dương Thuỷ Tiên |
| 25C15039 | Bùi Mạnh Hùng |
| 25C15046 | Nguyễn Đăng Khoa |
| 25C15062 | Nguyễn Đăng Thới Toàn |

## Mã nguồn theo nhiệm vụ

[submission/](submission/README.md) là bộ mã chính, có thể chép riêng để nộp. Mỗi task có README mô tả tiền xử lý, split, train, inference và đánh giá.

| Phần | Mã nguồn và cách chạy | Kết quả |
|---|---|---|
| 1 · VG150 | [MLP, GAT và tri thức nền](submission/task1_vg150/README.md) | [Baseline](results/vg150/baselines.json), [fusion](results/vg150/knowledge_fusion.json) |
| 2A · GQA | [PE-PLPR, AlignG và baseline](submission/task2_gqa/README.md) | [So sánh mô hình](results/gqa/models.json) |
| 2B · MagicBrush | [GNN dự đoán thao tác theo câu lệnh](submission/task2_graph_editing/magicbrush/README.md) | [Test 59 cặp](results/magicbrush/test.json) |
| 2B mở rộng · He và cộng sự | [MLP/GNN trên đồ thị truy vấn](submission/task2_graph_editing/he2020/README.md) | [Test 530 mẫu](results/he2020/test.json) |
| 3 · VN100 | [Qwen, REACT++ và điểm nhất quán](submission/task3_vn100/README.md) | [Đánh giá 200 yêu cầu](results/vn100/consistency.json) |
| Video · PVSG | [Dự đoán quan hệ theo thời gian](submission/extension_pvsg/README.md) | [Test 62 video](results/pvsg/test.json) |

## Cấu trúc

```text
submission/       Mã nguồn, môi trường, split và hai notebook demo
docs/             Hướng dẫn chạy lại
results/          Kết quả chính và provenance
tests/           Kiểm thử thao tác đồ thị và metric
environments/     Công cụ phát triển và kiểm tra
archived/         Thông tin snapshot và cách khôi phục thực nghiệm gốc
notebooks/        Notebook VG150 bổ sung từ phiên bản gốc
```

[Snapshot gốc](archived/README.md) giữ mã nguồn, cấu hình và kết quả lịch sử để đối chiếu các thực nghiệm đã khóa. Dataset lớn, cache và checkpoint được bàn giao riêng theo [manifest artifact](submission/assets/demo-assets.json).
