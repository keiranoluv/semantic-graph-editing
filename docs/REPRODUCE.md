# Tái lập các nhiệm vụ

## Mở kết quả đã lưu

Sau khi clone, mở [notebook tổng hợp](../notebooks/project_submission_demo.ipynb) trong JupyterLab hoặc VS Code. Output đã nhúng gồm 5 Hit và 5 Miss trên từng bộ VG150/GQA, hai ví dụ He, ca MagicBrush và sáu cặp VN100. Bước xem output chỉ cần tệp notebook. Để chạy lại các ô suy luận cần thêm môi trường và artifact bên dưới.

<a id="chay-demo"></a>
## Chạy demo

Các lệnh dưới đây chạy từ thư mục gốc của bản clone trên Linux, Python 3.12. Máy đã có workspace có thể dùng `myenv/bin/python`. Để kiểm tra độc lập, tạo môi trường CPU mới:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1 torchvision==0.24.1
.venv/bin/python -m pip install -r submission/requirements-cpu.txt
.venv/bin/python -m pip check
.venv/bin/python -m ipykernel install --prefix="$PWD/.venv" --name=gnn-cpu --display-name="Graph Editing (CPU)"
```

### Chọn bộ artifact

| Tên | Dùng cho | Tệp ngoài Git | Dung lượng |
|---|---|---:|---:|
| `chair` | Chạy nhanh ca xóa ghế của MagicBrush, kèm hai ảnh minh họa | 7 | 357 MiB |
| `live` | Chạy đủ ba ca trong notebook end-to-end | 21 | 412 MiB |
| `overview` | Chạy từ đầu toàn bộ notebook tổng hợp, gồm checkpoint VG150/GQA và kho ảnh GQA | 51 | 22,05 GiB |

Các bộ có tệp dùng chung. [Manifest](../submission/demo-assets.json) ghi đường dẫn khôi phục, bộ sử dụng, kích thước theo byte và SHA-256 của từng tệp. Artifact ở workspace thực nghiệm của nhóm, trong các cây `data/` và `runs/`. Dataset công khai VN100 cung cấp ảnh, lệnh, nhãn và điểm để tra cứu. Notebook còn đọc checkpoint, cache riêng, scene graph và receipt của lượt thực nghiệm.

**Nếu clone trên máy đang giữ workspace**, trỏ `--from-dir` tới gốc workspace cũ. Chạy trong bản clone mới:

```bash
.venv/bin/python scripts/submission_assets.py chair --from-dir /workspace/workspace/semantic-graph-editing
.venv/bin/python -m scripts.demo_magicbrush_chair
```

Hai dòng cuối cần in:

```text
Khớp toàn bộ thao tác: True
Khớp đồ thị đích: True
```

**Nếu chạy trên máy khác**, chuyển kèm artifact cùng với mã nguồn. Đối với ca nhanh, nhóm đã chuẩn bị `runs/submission-demo/chair-assets.tar` ở workspace gốc. SHA-256 của bản bàn giao này là `32b3407d50d3941f252e9e571fd87d78c69c048426cbf009f2ac9f418e2c9e18`. Chép tar sang máy mới, giải nén vào một thư mục trống và dùng thư mục đó làm `--from-dir`. Với bộ lớn hơn, có thể xuất đúng cây tệp cần chuyển từ workspace gốc:

```bash
python3.12 scripts/submission_assets.py overview --from-dir . --to-dir /tmp/graph-editing-assets
```

Chuyển thư mục vừa xuất sang máy đích, giữ cấu trúc tương đối. Trong bản clone mới chạy:

```bash
.venv/bin/python scripts/submission_assets.py overview --from-dir /tmp/graph-editing-assets
.venv/bin/jupyter lab notebooks/project_submission_demo.ipynb
```

Chọn kernel **Graph Editing (CPU)** và **Restart Kernel and Run All Cells**. Với demo vấn đáp, thay `overview` bằng `live` và mở `notebooks/graph_editing_live_demo.ipynb`. Để kiểm tra artifact đã có, chạy `scripts/submission_assets.py` với tên bộ và bỏ tùy chọn `--from-dir`. Công cụ báo đường dẫn đầu tiên còn thiếu hoặc sai hash. Các tệp đã khớp được dùng lại khi tiếp tục.

### Đối chiếu kết quả

- **VG150 và GQA.** Mỗi bảng có 5 hàng, Hit bên trái và Miss bên phải. GC top-50 xét cả nhãn quan hệ lẫn thứ hạng. Ví dụ VG150 ảnh 89 có hai bộ ba đúng nhãn ở hạng 1 và 64, chỉ bộ ba đầu là Hit.
- **MagicBrush và He.** Ca `magicbrush:190052:2` nạp GNN nhánh prompt, seed 123 và dự đoán trên CPU. Hai điều kiện trên phải cùng khớp. Mục He nạp thao tác GNN đã lưu rồi dựng đồ thị dự đoán.
- **VN100.** Sáu cặp ảnh lấy từ hai nguồn. Ở ca thêm chó, bộ chấm tính lại E = 1 và G ≈ 0,829. Hai ca có phép ghép mơ hồ hiển thị G không xác định.

## Dữ liệu, split và cách đối chiếu từng nhiệm vụ

| Phần | Split và phòng tránh rò rỉ | Điểm vào tái lập và đầu ra |
|---|---|---|
| VG150 | `data/raw/vg150/splits.json` cố định ID ảnh: 57.723 train, 5.000 validation và 26.446 test. Giữ chính sách lọc hộp và hash dữ liệu trong checkpoint. Demo 10 chú thích từ validation. | [Chuẩn bị dữ liệu và mã tạo split](datasets/VG150.md) · [Chấm lại 27 checkpoint đã chọn](experiments/level1_reference_test27_v1.md) · [Kết quả](../reports/level1/reference-test27-v1) |
| GQA | `data/processed/level2/gqa-v1/manifest.json` chia phần train chính thức theo ảnh thành 69.942 train và 5.000 validation. Dùng 10.696 ảnh validation chính thức làm test của đồ án. Từ vựng lấy từ train. | [Hai mô hình trong demo](../experiments/aligng_fast_v1) · [Danh sách checkpoint](../reports/level2/gqa-test-v1/checkpoint-inventory.csv) · [27 kết quả test tổng hợp](../reports/level2/gqa-test-v1/summary.json) |
| MagicBrush | [Manifest cùng split](../experiments/magicbrush_prompt_intent_v1/data/manifest.json) cố định 240/46/59 cặp train/val/test. Tách nhóm ảnh COCO và hash pixel của cả hai vai ảnh. Hai nhánh dùng cùng ID và checkpoint chọn trên validation. | [Lệnh chuẩn bị, huấn luyện và đánh giá](../experiments/magicbrush_prompt_intent_v1/README.md) · [Checkpoint theo seed](../reports/level2/magicbrush-prompt-intent-test-v1/selection.json) |
| He và cộng sự | Protocol crowdsourced dùng `trainval-groups.json` và `test-receipt.json` trong `data/processed/level2/emnlp2020-task-b-v2/`. Kiểm tra nhóm nguồn, fit từ vựng trên train. Có 530 mẫu test sau lọc. Chưa chứng minh split độc lập theo ảnh, và test từng được dùng để đánh giá v3. | [Cấu hình, lệnh CPU/GPU và checkpoint v4](LEVEL2_EMNLP2020_TASK_B_WANDB_V4.md) · [Đánh giá test và đầu ra](LEVEL2_EMNLP2020_TASK_B_V4_TEST_V1.md) |
| VN100 | [Manifest 200 yêu cầu](../configs/level3/qwen200-cohort-v1.json). Một tập đánh giá phát triển `dev`, hai yêu cầu độc lập dùng cùng một ảnh nguồn. Nếu phân chia cho thí nghiệm sau, nhóm theo `image_id`. | [Sinh ảnh và trích graph](LEVEL3_PIPELINE.md) · [Chấm graph, baseline ảnh, nhãn tay](LEVEL3_EVALUATION.md) · [Data card](LEVEL3_DATA_CARD.md) |
| PVSG | [Split theo video](../experiments/pvsg_trd_v1/manifests/pvsg_author_release_338_62.lock.json) có 338 video train và 62 test. Chia train thành 287 fit và 51 validation, giữ các cửa sổ của một video cùng tập. Bốn nhóm ID nguồn Ego4D xuất hiện ở cả hai phía của split phát hành. | [Evaluator M5](../experiments/pvsg_trd_v1/m5) · [Kết quả, checkpoint đã chọn và ví dụ](../reports/pvsg-trd-m5-v1/final.md) |

Các bộ artifact CPU bên trên chỉ chứa đầu vào được dùng bởi demo. Khi tái lập cả lượt huấn luyện hoặc đánh giá toàn tập, dùng môi trường, dữ liệu đầy đủ, checkpoint và launcher trong hàng tương ứng. Launcher thí nghiệm có thể yêu cầu workspace cũ để kiểm tra đường dẫn và hash đã khóa. Các script GPU được người chạy khởi động riêng sau khi hoàn tất bước kiểm tra CPU.

### Lệnh đối chiếu trên workspace có đầy đủ artifact thực nghiệm

| Phần | Lệnh từ gốc repo | Kết quả cần xem |
|---|---|---|
| VG150 | `myenv/bin/python -m tools.level1_reference_test27_v1 --dry-run` | Kiểm tra kế hoạch 27 checkpoint gốc trước khi chấm. |
| GQA | `myenv/bin/python -m tools.gqa_test_v1.runner cpu-smoke --output /tmp/gqa-cpu-smoke` | Kiểm checkpoint và forward trên CPU theo danh mục test. |
| MagicBrush | `myenv/bin/python -m experiments.magicbrush_prompt_intent_v1.evaluate` | Chấm sáu checkpoint validation trên cùng 59 mẫu test. Kết quả ở `reports/level2/magicbrush-prompt-intent-test-v1/`. |
| He và cộng sự | `myenv/bin/python -m tools.eval_emnlp2020_task_b_wandb_v4_test_v1 --stage summarize` | Tổng hợp lượt đánh giá v4 đã hoàn tất. |
| VN100 | `CUDA_VISIBLE_DEVICES='' bash scripts/run_level3_eval.sh graph` | Điểm đồ thị của 200 yêu cầu. |
| PVSG | `bash experiments/pvsg_trd_v1/m5/launch.sh check` | Đối chiếu manifest và checkpoint chọn từ validation bằng môi trường riêng của PVSG. |

### Phạm vi đã chạy thử

Đã kiểm tra commit `c55cfe01` từ bản clone GitHub trên Linux, bằng môi trường Python 3.12 mới cài theo hướng dẫn trên. Các artifact được chép qua manifest, và `archived/` được đưa ra ngoài bản clone trước khi chạy. Ca xóa ghế trả về hai giá trị `True`. Cả hai notebook chạy hết trong kernel CPU mới sau khi xóa output cũ trong bộ nhớ. Phép thử này kiểm tra đường chạy demo; các kết quả đánh giá toàn tập trong bảng trên thuộc những lượt thực nghiệm đã lưu.
