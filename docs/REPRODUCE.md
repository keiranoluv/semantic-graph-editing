# Chạy lại demo và thực nghiệm

## Xem kết quả

Mở [notebook tổng hợp](../submission/notebooks/project_submission_demo.ipynb) hoặc [notebook demo trực tiếp](../submission/notebooks/graph_editing_live_demo.ipynb). Output đã được lưu; bước xem chỉ cần notebook.

## Chạy demo trên CPU

Dùng Linux và Python 3.12. Từ gốc repo:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1 torchvision==0.24.1
.venv/bin/python -m pip install -r submission/environments/demo-cpu.txt
.venv/bin/python -m ipykernel install --prefix="$PWD/.venv" --name=gnn-cpu --display-name="Graph Editing (CPU)"
cd submission
../.venv/bin/python -m common.restore_assets chair --from-dir /duong/dan/artifacts
../.venv/bin/python -m task2_graph_editing.magicbrush.demo
```

Ca xóa ghế dùng GNN prompt seed 123. Hai dòng cuối kiểm tra tập thao tác và đồ thị đích đều phải in True.

## Chọn artifact cho notebook

| Bộ | Dùng để chạy lại | Dung lượng |
|---|---|---|
| chair | Demo xóa ghế bằng lệnh trên | 357 MiB |
| live | graph_editing_live_demo.ipynb: thuyền buồm, chó và ghế | 412 MiB |
| overview | project_submission_demo.ipynb: VG150, GQA, MagicBrush, He và VN100 | 22,05 GiB |

Bộ overview có kho ảnh GQA khoảng 20,32 GiB. Chạy cả hai notebook cần khôi phục live và overview; phần chung được dùng lại, tổng khoảng 22,08 GiB. [Manifest artifact](../submission/assets/demo-assets.json) có kích thước và SHA-256 của từng tệp. Thư mục artifacts cần giữ cấu trúc data/ và runs/ của bộ đã bàn giao.

Sau khi cài môi trường bên trên, chạy từ submission/:

```bash
../.venv/bin/python -m common.restore_assets live --from-dir /duong/dan/artifacts
../.venv/bin/jupyter lab notebooks/graph_editing_live_demo.ipynb
```

Chọn kernel Graph Editing (CPU). Với notebook tổng hợp, thay live bằng overview và mở notebooks/project_submission_demo.ipynb. Bỏ --from-dir để chỉ kiểm tra các tệp đã khôi phục.

## Huấn luyện và đánh giá

Các lệnh dưới đây đều được mô tả theo môi trường, batch size, seed, output và phạm vi hàng đợi trong README của từng task:

- [VG150](../submission/task1_vg150/README.md): train.py chọn một điều kiện và một seed từ danh mục cấu hình đã dùng.
- [GQA](../submission/task2_gqa/README.md): train.py chạy tuần tự PE-PLPR và AlignG; training/baselines.py chạy một baseline.
- [MagicBrush](../submission/task2_graph_editing/magicbrush/README.md): train.py chạy hai chính sách nhãn và ba seed.
- [He và cộng sự](../submission/task2_graph_editing/he2020/README.md): train.py có preflight, smoke và hàng đợi MLP/GNN.
- [VN100](../submission/task3_vn100/README.md): generate.py tạo ảnh, extract_graphs.py trích đồ thị; dùng mô hình pretrained.
- [PVSG](../submission/extension_pvsg/README.md): train.py chạy một model/seed, evaluate.py chấm checkpoint.

Huấn luyện chính dùng W&B online. Chạy lần lượt một mô hình GPU; mỗi README nói rõ lệnh nào tự chuyển sang lượt kế tiếp. Khi chạy lại bằng mã đã refactor, dùng output mới và giữ nguyên artifact lịch sử. Các bộ artifact notebook chỉ chứa đầu vào cho demo, cần bổ sung dữ liệu đầy đủ khi train hoặc chấm toàn tập.

[README bản mã nguồn](../submission/README.md) mô tả split và phòng tránh rò rỉ. [Kết quả đã lưu](../results/README.md) có provenance; [snapshot gốc](../archived/README.md) dành cho việc đối chiếu launcher cũ.

## Kiểm tra mã

Sau khi cài môi trường demo CPU, cài thêm môi trường kiểm tra từ environments/dev.txt và chạy ở gốc repo:

```bash
.venv/bin/python -m pip install -r environments/dev.txt
.venv/bin/python -m pytest
.venv/bin/pre-commit run
```

Các kiểm thử xác nhận thao tác đồ thị, chia nhóm dữ liệu và metric PredCls trên dữ liệu nhỏ, không cần checkpoint hay dữ liệu huấn luyện.
