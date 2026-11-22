# Nhiệm vụ 2B · MagicBrush

Đầu vào là đồ thị nguồn và câu lệnh. GNN dự đoán thao tác nút/cạnh, sau đó áp thao tác để dựng đồ thị đích. Chạy lệnh từ gốc submission/, dùng environments/graph-training.txt.

## Huấn luyện

    python -m task2_graph_editing.magicbrush.run

Launcher chạy tuần tự GNN với hai chính sách graph_diff/prompt × seeds 42, 123, 3407, W&B online. Batch train 16, validation 32, lr 0.0003, weight decay 0.01, tối đa 200 epoch, tối thiểu 30 epoch, patience 30, gradient clip 5. Mô hình hidden 192, 2 lớp, dropout và giới hạn slot theo configs/level2/magicbrush-manual-experiment-v3.json.

run.py đọc cấu hình cha này cùng data/manifest.json, rồi dùng bộ cặp prompt 240/46/59 train/validation/test. Output ở runs/level2/magicbrush-prompt-intent-v1/{graph_diff,prompt}/seed-{42,123,3407}/. Hàng đợi tự chuyển sang lượt kế tiếp khi lượt hiện tại hoàn tất.

Cần CLIP weights và các cache train/validation/test theo đường dẫn trong config. Bước kiểm tra CPU là:

    python -m task2_graph_editing.magicbrush.run --cpu-check

## Suy luận và chấm điểm

    python -m common.restore_assets chair --from-dir /duong/dan/artifacts
    python -m task2_graph_editing.magicbrush.demo
    python -m task2_graph_editing.magicbrush.evaluate --seed 123 --policy prompt --split val --max-examples 5

Demo dùng GNN prompt seed 123 trên CPU. Evaluator đọc một checkpoint và chấm split đã chọn. Có thể truyền --checkpoint để chấm lượt train mới. Danh mục checkpoint demo nằm trong assets/magicbrush_checkpoints.json.

dataset.py đọc split, examples.py dựng mẫu từ nhãn, features.py đọc đặc trưng văn bản. structured_model/model.py cài GNN, structured_model/training.py tối ưu và predict.py suy luận. sequence_model/ chứa biểu diễn đầu vào và codec dùng chung.

preprocess.py chuẩn bị nguồn MagicBrush và chia nhóm ảnh. data/ chứa bộ cặp đồ thị có nhãn thao tác và split cố định. review20.json, review20-result.json ghi nhận mẫu kiểm tay và kết quả.
