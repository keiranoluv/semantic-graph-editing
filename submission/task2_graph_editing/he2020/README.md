# Mở rộng nhiệm vụ 2B · He và cộng sự

Chạy từ gốc submission/, dùng environments/graph-training.txt và protobuf==5.29.5. Dữ liệu đã chuẩn hóa cùng CLIP cache nằm ở data/processed/level2/emnlp2020-task-b-v2/.

## Huấn luyện

    python -m task2_graph_editing.he2020.training --config configs/level2/emnlp2020-task-b-wandb-v4.json --stage preflight
    python -m task2_graph_editing.he2020.training --config configs/level2/emnlp2020-task-b-wandb-v4.json --stage train

Runner chạy tuần tự MLP rồi GNN, mỗi model có seeds 42, 43, 44. Batch 128, hidden 96, 2 lớp, lr 0.001, weight decay 0.01, tối đa 20 epoch, tối thiểu 6, patience 3, min_delta 0.001, gradient clip 5. W&B online. Output ở runs/level2/emnlp2020-task-b-wandb-v4/{model}-seed{seed}/. Runner tự chuyển lượt tiếp theo khi hoàn tất.

## Suy luận và đánh giá

    python -m task2_graph_editing.he2020.evaluate --config configs/level2/emnlp2020-task-b-wandb-v4.json --stage eval --device cpu

Evaluator đọc sáu checkpoint đã chọn và chấm 530 mẫu test sau lọc, với đối chiếu nguồn và cache. Notebook tổng hợp phát lại một dự đoán đúng và một dự đoán sai rồi áp thao tác lên đồ thị nguồn.

preprocessing/ đọc và chuẩn hóa bộ gốc, data_processing/ chuẩn bị train/validation, models/ cài MLP/GNN, training/ huấn luyện với W&B. Tính độc lập của split theo ảnh gốc chưa được xác minh.
