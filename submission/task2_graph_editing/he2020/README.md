# Mở rộng nhiệm vụ 2B · He và cộng sự

preprocessing/ đọc và chuẩn hóa bộ crowdsourced gốc. data_processing/ chuẩn bị train/validation. models/ cài MLP/GNN, training/ huấn luyện và evaluate.py chấm test.

Notebook tổng hợp phát lại một dự đoán đúng và một dự đoán sai, kèm thao tác áp trên đồ thị nguồn.

Cấu hình ở configs/level2/emnlp2020-task-b-wandb-v4.json. Môi trường graph-training cần protobuf==5.29.5. Launcher commands/run_emnlp2020_task_b_wandb_v4.sh nhận EMNLP_PYTHON, chạy tuần tự MLP/GNN × ba seeds theo cấu hình, W&B online. Output nằm dưới runs/level2/. Cần dữ liệu và cache đã chuẩn bị.
