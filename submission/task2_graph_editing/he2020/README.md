# Mở rộng nhiệm vụ 2B · He và cộng sự

Chạy từ gốc submission/, dùng environments/graph-training.txt. Dữ liệu đã chuẩn hóa cùng CLIP cache nằm ở data/processed/level2/emnlp2020-task-b-v2/.

## Huấn luyện

    python -m task2_graph_editing.he2020.train --config configs/level2/emnlp2020-task-b-wandb-v4.json --stage preflight
    python -m task2_graph_editing.he2020.train --config configs/level2/emnlp2020-task-b-wandb-v4.json --stage train

Runner chạy tuần tự MLP rồi GNN, mỗi model có seeds 42, 43, 44. Batch 128, hidden 96, 2 lớp, lr 0.001, weight decay 0.01, tối đa 20 epoch, tối thiểu 6, patience 3, min_delta 0.001, gradient clip 5. W&B online. Output ở runs/level2/emnlp2020-task-b-wandb-v4/{model}-seed{seed}/. Runner tự chuyển lượt tiếp theo khi hoàn tất.

## Suy luận và đánh giá

    python -m task2_graph_editing.he2020.predict --checkpoint runs/level2/emnlp2020-task-b-wandb-v4/gnn-seed42/best.pt --model gnn --input ./input.json
    python -m task2_graph_editing.he2020.evaluate --checkpoint runs/level2/emnlp2020-task-b-wandb-v4/gnn-seed42/best.pt --model gnn --split val --max-examples 5

input.json chứa source_graph theo schema common/editing/operations.py và instruction. Suy luận chạy CPU, trả thao tác và đồ thị đích. Evaluator chấm một checkpoint trên validation hoặc test, đọc các cache đã tạo. Notebook phát lại một ca đúng và một ca sai đã lưu.

## Chuẩn bị dữ liệu

    python -m task2_graph_editing.he2020.preprocess --archive data/raw/emnlp2020-graph-modification-v1/scene_graph_modification_data.tgz --output data/processed/level2/emnlp2020-task-b-v2

Thêm --test để chuyển đổi tập test sau khi đã khóa train/validation. Bộ đọc pickle trong preprocessing/ chạy trong tiến trình cô lập. dataset.py chuẩn hóa và lọc trùng theo nhóm nguồn; model.py định nghĩa MLP/GNN và từ vựng thao tác; features.py tạo CLIP cache. training.py chứa vòng tối ưu và W&B, runtime.py chuẩn bị batch, dự đoán và tính metric dùng chung với trainer. Encoder và codec dùng chung với MagicBrush nằm ở common/editing/. Tính độc lập theo ảnh gốc chưa được xác minh.
