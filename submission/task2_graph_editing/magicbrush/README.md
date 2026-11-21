# Nhiệm vụ 2B · MagicBrush

Đầu vào là đồ thị nguồn và câu lệnh. GNN dự đoán thao tác nút/cạnh, sau đó áp thao tác để dựng đồ thị đích. Chạy lệnh từ gốc submission/, dùng environments/graph-training.txt và protobuf==5.29.5.

## Huấn luyện

    MAGICBRUSH_PYTHON=/duong/dan/venv/bin/python bash task2_graph_editing/magicbrush/commands/run_magicbrush_prompt_intent_v1_wandb.sh

Launcher chạy tuần tự GNN với hai chính sách graph_diff/prompt × seeds 42, 123, 3407, W&B online. Batch train 16, validation 32, lr 0.0003, weight decay 0.01, tối đa 200 epoch, tối thiểu 30 epoch, patience 30, gradient clip 5. Mô hình hidden 192, 2 lớp, dropout và giới hạn slot theo configs/level2/magicbrush-manual-experiment-v3.json.

run.py đọc cấu hình cha này cùng data/manifest.json, rồi dùng bộ cặp prompt 240/46/59 train/validation/test. Output ở runs/level2/magicbrush-prompt-intent-v1/{graph_diff,prompt}/seed-{42,123,3407}/. Hàng đợi tự chuyển sang lượt kế tiếp khi lượt hiện tại hoàn tất.

Cần CLIP weights và các cache train/validation/test theo đường dẫn trong config. Bước kiểm tra CPU là:

    python -m task2_graph_editing.magicbrush.run --cpu-check

## Suy luận và chấm điểm

    python -m common.restore_assets chair --from-dir /duong/dan/artifacts
    python -m task2_graph_editing.magicbrush.demo
    python -m task2_graph_editing.magicbrush.evaluate

Demo dùng GNN prompt seed 123 trên CPU. Evaluator cần đủ sáu checkpoint và test cache, đối chiếu protocol để nhận diện lượt gốc hoặc lượt từ bản refactor.

dataset.py đọc split, examples.py dựng mẫu từ nhãn, structured_model/model.py cài GNN, structured_model/training.py tối ưu, predict.py suy luận. sequence_model/ cung cấp biểu diễn đầu vào và codec dùng chung. pair_generation/, annotation/ và preprocess.py phục vụ tạo cặp và kiểm tra dữ liệu.
