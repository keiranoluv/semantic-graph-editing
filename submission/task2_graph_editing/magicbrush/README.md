# Nhiệm vụ 2B · MagicBrush

Đầu vào là đồ thị nguồn và câu lệnh. GNN dự đoán thao tác nút/cạnh, sau đó áp thao tác để dựng đồ thị đích.

- data/ và dataset.py chứa bộ cặp và split.
- structured_model/model.py cài GNN. training.py tối ưu và decoding.py giải mã.
- run.py huấn luyện và evaluate.py chấm 59 cặp test.
- predict.py tải checkpoint và suy luận. demo.py chạy ca xóa ghế.
- pair_generation/, annotation/ và preprocess.py chuẩn bị dữ liệu. sequence_model/ cung cấp biểu diễn đầu vào và mã hóa thao tác dùng chung với GNN. examples/run.py dựng mẫu từ nhãn.

Sau khi khôi phục bộ chair theo README, chạy từ thư mục gốc bản nộp:

    python -m task2_graph_editing.magicbrush.demo

Huấn luyện dùng graph-training và launcher commands/run_magicbrush_prompt_intent_v1_wandb.sh. MAGICBRUSH_PYTHON trỏ tới Python môi trường. Runner chạy tuần tự hai chính sách nhãn × seeds 42, 123, 3407, W&B online. Batch và ngân sách theo configs/level2/magicbrush-manual-experiment-v3.json. Output ở runs/level2/magicbrush-prompt-intent-v1/.

evaluate.py đối chiếu xuất xứ trong protocol để chấm checkpoint lịch sử hoặc checkpoint từ mã đã refactor. Train bằng mã đã refactor tạo hash mới.
