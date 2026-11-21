# Nhiệm vụ 2A · GQA

Chạy từ gốc submission/. Cài environments/gqa-cu128.txt, đăng nhập W&B và chuẩn bị data/processed/level2/gqa-v1/manifest.json, cache data/cache/gqa-full-clip-v1/ cùng GloVe data/processed/level2/gqa-glove-v1/.

## Huấn luyện PE-PLPR và AlignG

    python -m task2_gqa.training.runner --stage check --loader-workers 4
    python -m task2_gqa.training.runner --stage train-all --loader-workers 4

Runner dùng configs/level2/aligng-gqa-fast-v1-bs512.json. Batch train và validation đều 512, lr 0.001, weight decay 0.01, tối đa 200 epoch, patience 15. Sáu lượt chạy tuần tự, mỗi seed 42, 123, 3407 lần lượt train PE-PLPR + AlignG rồi PE-PLPR. W&B online, output ở runs/level2/aligng-gqa-fast-v1/full/. Runner tiếp tục hàng đợi khi lượt hiện tại hoàn tất.

training/model.py cài mô hình, training/engine.py cài vòng tối ưu, recipe/ chuẩn bị dữ liệu và validation, models/ chứa khối AlignG. dataset.py chứa loader RAM mà lượt full sử dụng.

## Baseline MLP, GAT và PE-Net

    python -m task2_gqa.train_baselines --stage train --config configs/level2/gqa-full-v3-bs512.json --model gat --seed 42

Lệnh trên chạy một model/seed, output theo trường output của config. Đổi --model thành mlp, pe-ce, pe-pl hoặc pe-plpr. Cấu hình cố định batch 512 và seeds 42, 123, 3407. evaluation.py cung cấp cách dựng baseline và tính metric. knowledge.py dựng GAT + KG để đọc checkpoint của phần đối chứng tri thức.

## Suy luận và đánh giá

    python -m task2_gqa.gqa_test_v1.runner cpu-smoke --output /tmp/gqa-cpu-smoke

Notebook tổng hợp suy luận checkpoint trên CPU, hiển thị 5 Hit và 5 Miss. Bộ chấm đọc danh mục checkpoint đã chọn trên validation. prepare_level2_gqa.py chuẩn hóa đồ thị và tách tập theo ID ảnh.
