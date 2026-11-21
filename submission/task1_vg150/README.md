# Nhiệm vụ 1 · VG150

preprocess.py tạo split, cache_vg150_regions.py trích CLIP. Các lớp mô hình và cách chấm nằm trong common/models/ và common/evaluation/.

Với môi trường graph-training, thêm protobuf==5.29.5. Dữ liệu ở data/raw/vg150/, cache ở data/cache/vg150-clip-vit-b32-v1/. Chấm hai ảnh validation trên CPU từ thư mục gốc bản nộp.

    python -m task1_vg150.evaluate --checkpoint runs/level1-clip-v1/clip-gat-seed42-v1/run_0001/best.pt --split val --protocol reference-v1 --device cpu --max-images 2 --output runs/vg150/demo-val2.json

Để train, đăng nhập W&B. Lệnh sau chạy một GAT, seed 42, micro-batch 32, tích lũy 16 batch mỗi bước cập nhật và dừng sau model này. Output ở runs/vg150/gat-seed42/run_0001/.

    python -m task1_vg150.train --device cuda --wandb-mode online --seed 42 --batch-size 32 --accumulation-steps 16 --visual-cache data/cache/vg150-clip-vit-b32-v1 --output-dir runs/vg150 --run-name gat-seed42
