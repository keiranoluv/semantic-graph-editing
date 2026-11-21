# Nhiệm vụ 2A · GQA

prepare_level2_gqa.py chuẩn hóa và chia tập theo ảnh. training/model.py cài PE-PLPR và AlignG. training/runner.py điều phối train. recipe/ chuẩn bị dữ liệu và chấm validation. models/ chứa thành phần AlignG dùng chung.

Notebook tổng hợp suy luận checkpoint trên CPU, trình bày 5 Hit và 5 Miss. Với đủ checkpoint và cache có thể kiểm tra evaluator bằng:

    python -m task2_gqa.gqa_test_v1.runner cpu-smoke --output /tmp/gqa-cpu-smoke

Dùng environments/gqa-cu128.txt và configs/level2/aligng-gqa-fast-v1-bs512.json. Runner có bước --stage check trên CPU và --stage train-all cho sáu lượt tuần tự, hai mô hình × seeds 42, 123, 3407, batch 512, W&B online. Output ở runs/level2/aligng-gqa-fast-v1/full/. Cần dữ liệu, CLIP cache và GloVe.
