# Chỉnh sửa ảnh theo yêu cầu bằng đồ thị cảnh

Bài làm gồm ba nhiệm vụ: dự đoán quan hệ giữa các đối tượng, dự đoán thao tác chỉnh sửa từ câu lệnh và đánh giá ảnh chỉnh sửa bằng đồ thị cảnh. Phần mở rộng PVSG xét quan hệ giữa các đối tượng theo thời gian.

[Mở notebook tổng hợp đã lưu kết quả](notebooks/project_submission_demo.ipynb) · [Demo trực tiếp ba ca chỉnh sửa](notebooks/graph_editing_live_demo.ipynb) · [Dữ liệu VN100 và data card](https://huggingface.co/datasets/ndttoan/vn100-graph-editing)

## Mã nguồn theo nhiệm vụ

| Nhiệm vụ | Tiền xử lý và đồ thị | Huấn luyện và đánh giá |
|---|---|---|
| **1 · VG150** | [Tạo split](code/scripts/build_vg150_manifest.py), [đọc dữ liệu](code/src/datasets/vg150.py), [dựng đồ thị](code/src/graph), [đặc trưng CLIP](code/scripts/cache_vg150_regions.py) | [MLP](code/scripts/train_mlp.py), [GAT](code/scripts/train_gat.py), [GAT với tri thức nền](code/scripts/train_gat_kg.py), [chấm PredCls](code/scripts/evaluate_level1.py) |
| **2A · GQA** | [Chia tập theo ảnh và xây từ vựng trên train](code/tools/prepare_level2_gqa.py) | [PE-PLPR và AlignG](code/experiments/aligng_fast_v1), [đánh giá test](code/tools/gqa_test_v1/runner.py) |
| **2B · MagicBrush** | [Bộ cặp và split cố định](code/experiments/magicbrush_prompt_intent_v1/data), [quy tắc thao tác trên đồ thị](code/src/editing/operations.py) | [Huấn luyện GNN](code/experiments/magicbrush_prompt_intent_v1/run.py), [đánh giá thao tác](code/experiments/magicbrush_prompt_intent_v1/evaluate.py), [kết quả kiểm tay](code/experiments/magicbrush_prompt_intent_v1/review20-result.json) |
| **2B mở rộng · He và cộng sự** | [Chuẩn hóa cặp đồ thị và câu lệnh](code/tools/emnlp2020_graph_modification_v1) | [Huấn luyện MLP/GNN](code/tools/emnlp2020_task_b_wandb_v4), [đánh giá trên 530 mẫu test](code/tools/eval_emnlp2020_task_b_wandb_v4_test_v1.py) |
| **3 · VN100** | [Manifest 200 yêu cầu](code/configs/level3/qwen200-cohort-v1.json), [Qwen chỉnh ảnh](code/tools/level3_qwen_unified), [REACT++ trích đồ thị](code/tools/level3_qwen_sgg) | [So khớp đồ thị và tính điểm](code/tools/level3_evaluation). Notebook minh họa sáu ca có ảnh, đồ thị và nhãn tay. |
| **Video · PVSG** | [Split theo video](code/experiments/pvsg_trd_v1/manifests/pvsg_author_release_338_62.lock.json), [đặc trưng vùng và track](code/experiments/pvsg_trd_v1/m2) | [Mô hình và huấn luyện](code/experiments/pvsg_trd_v1/m3), [đánh giá quan hệ theo thời gian](code/experiments/pvsg_trd_v1/m5) |

## Dữ liệu và cách đọc kết quả

- **VG150/GQA.** Demo chạy checkpoint trên validation và đối chiếu 5 Hit, 5 Miss của mỗi bộ. VG150 dùng split cố định theo ảnh. GQA chia train chính thức thành 69.942 ảnh train và 5.000 ảnh validation; 10.696 ảnh validation chính thức dùng làm test của đồ án.
- **MagicBrush.** Mỗi mẫu có đồ thị nguồn, câu lệnh và thao tác đích. Nhãn thao tác được lập với hỗ trợ AI từ cặp trước/sau, giữ thay đổi mà câu lệnh yêu cầu và các thao tác cấu trúc liên quan. Đồ thị đích dùng để chấm được tạo bằng cách áp nhãn thao tác lên đồ thị nguồn. Dữ liệu dùng 240/46/59 cặp train/validation/test, tách theo nhóm COCO và hash pixel hai ảnh. Có 20/20 mẫu đạt trong đợt [kiểm tra tay trên 345 cặp](code/experiments/magicbrush_prompt_intent_v1/review20.json).
- **He và cộng sự.** Tập test đã lọc gồm 530 mẫu. Quy trình kiểm tra các nhóm nguồn đã dành cho train/validation; tính độc lập theo ảnh gốc chưa được xác minh. Notebook dùng dự đoán đã lưu để dựng lại đồ thị đích.
- **VN100.** Split phát triển gồm 100 ảnh Việt Nam và 200 yêu cầu. Hai chỉnh sửa của cùng ảnh có chung `image_id`. [Data card, nguồn ảnh và giấy phép](code/datasets/vn100/README.md) · [Ảnh nguồn, câu lệnh và ảnh Qwen](https://huggingface.co/datasets/ndttoan/vn100-graph-editing/viewer/image-pairs/dev) · [Nhãn và điểm E/P/G](https://huggingface.co/datasets/ndttoan/vn100-graph-editing/viewer/default/dev).
- **PVSG.** Chia 338 video train thành 287 fit và 51 validation; 62 video test theo split phát hành. Các cửa sổ của cùng video ở cùng tập. Bốn nhóm ID nguồn Ego4D xuất hiện hai phía của split phát hành.

## Chạy lại một ca trên CPU

Notebook tổng hợp có thể mở trực tiếp để xem output. Để suy luận lại, cài môi trường trên Linux với Python 3.12. Chạy các lệnh sau từ thư mục chứa README này:

```bash
python3.12 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1 torchvision==0.24.1
.venv/bin/pip install -r environments/demo-cpu.txt
cd code
../.venv/bin/python scripts/submission_assets.py chair --from-dir /duong/dan/workspace-thuc-nghiem
../.venv/bin/python -m scripts.demo_magicbrush_chair
```

Thay đường dẫn bằng workspace có `data/`, `runs/` hoặc thư mục đã giải nén gói artifact tương ứng. Với gói `chair-assets.tar` của nhóm, dùng `tar -xf chair-assets.tar -C /duong/dan/thu-muc-trong` rồi truyền thư mục đó cho `--from-dir`. Ca `remove the chairs` cần 7 tệp (357 MiB), gồm trọng số CLIP và GNN seed 123. Hai dòng cuối kiểm tra tập thao tác và đồ thị đích đều in `True`. [Danh mục tệp, dung lượng và SHA-256](code/submission/demo-assets.json)

Để chạy notebook, thay `chair` bằng `live` (412 MiB) hoặc `overview` (22,05 GiB). Từ `code/`, mở `../.venv/bin/jupyter lab ../notebooks` và dùng kernel của môi trường vừa cài. Mỗi tệp trong [environments/](environments) dành cho một pipeline. Các lượt đầy đủ cần dữ liệu, cache và checkpoint tương ứng.

## Chạy lại huấn luyện và đánh giá

Ví dụ với GAT trên VG150, chạy từ `code/` trong môi trường `graph-training.txt`, cài thêm `protobuf==5.29.5` cho W&B và đăng nhập W&B. Đặt dữ liệu VG150 đã chia tập vào `data/raw/vg150/` và cache CLIP vào `data/cache/vg150-clip-vit-b32-v1/`. Lệnh đầu huấn luyện một model, seed 42, batch 32 đồ thị, tích lũy 16 batch mỗi bước cập nhật. Lệnh thứ hai dùng checkpoint tốt nhất của lượt đầu để chấm validation trên CPU.

```bash
python -m scripts.train_gat --device cuda --wandb-mode online --seed 42 --batch-size 32 --accumulation-steps 16 --visual-cache data/cache/vg150-clip-vit-b32-v1 --output-dir runs/vg150 --run-name gat-seed42
python -m scripts.evaluate_level1 --checkpoint runs/vg150/gat-seed42/run_0001/best.pt --split val --protocol reference-v1 --device cpu --output runs/vg150/gat-seed42/run_0001/val.json
```

Với nhiệm vụ 2B, [launcher MagicBrush](code/scripts/run_magicbrush_prompt_intent_v1_wandb.sh) chạy các cấu hình đã khóa lần lượt. Khi có đủ sáu checkpoint, chạy `python -m experiments.magicbrush_prompt_intent_v1.evaluate` từ `code/` để chấm 59 cặp test. Các launcher cần kiểm tra phiên bản mã nguồn qua Git phải chạy từ bản clone.
