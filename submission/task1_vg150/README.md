# Nhiệm vụ 1 · VG150

Chạy các lệnh từ thư mục gốc submission/ với môi trường environments/graph-training.txt, thêm protobuf==5.29.5 và đăng nhập W&B.

## Cấu hình đã dùng

Metadata checkpoint CLIP ghi batch train 512 ảnh, accumulation 1 và batch validation 512. Các lượt dùng seeds 42, 123, 3407, tối đa 200 epoch, patience 15, lr 0.001, weight decay 0.01, negative ratio 3, 57.723 ảnh train và 5.000 validation. Các mô hình có hidden dimension 256 và object embedding 32. GAT có 2 lớp, 4 heads, dropout 0.1.

[Danh mục 60 lượt](../assets/training_recipes.json) ghi tham số, đường dẫn báo cáo nguồn và SHA-256 báo cáo cho các điều kiện chính, loss/fusion và hai hoán vị KG. train_report.py đọc trực tiếp danh mục để tạo lệnh cho runner tương ứng.

## Huấn luyện

Xem chính xác lệnh trước khi chạy, rồi bỏ --dry-run để huấn luyện:

    python -m task1_vg150.train_report --condition gat --seed 42 --output-dir runs/retrain-vg150 --dry-run
    python -m task1_vg150.train_report --condition gat --seed 42 --output-dir runs/retrain-vg150

Mỗi lệnh chạy một điều kiện và một seed, W&B online. MLP/GAT lưu checkpoint trong thư mục con run_0001, các biến thể KG lưu trực tiếp trong thư mục điều kiện. Muốn chạy seed tiếp theo, đổi --seed và chạy sau khi lượt trước kết thúc.

Lệnh GAT được launcher tạo tương ứng với:

    python -m task1_vg150.train --root data/raw/vg150 --visual-cache data/cache/vg150-clip-vit-b32-v1 --device cuda --wandb-mode online --seed 42 --epochs 200 --patience 15 --min-delta 0 --batch-size 512 --accumulation-steps 1 --val-batch-size 512 --num-workers 0 --max-train-images 57723 --max-val-images 5000 --hidden-dim 256 --embedding-dim 32 --num-heads 4 --num-layers 2 --dropout 0.1 --lr 0.001 --weight-decay 0.01 --negative-ratio 3 --box-policy clip --output-dir runs/retrain-vg150 --run-name gat-seed42

Đổi --condition thành mlp_semantic, wordnet_kg_on, wordnet_edges_off, kg_off, conceptnet_transe_kg_on, conceptnet_transe_edges_off, conceptnet_rotate_kg_on hoặc conceptnet_complex_kg_on để chọn các điều kiện chính. --help liệt kê cả điều kiện loss/fusion và các hoán vị có trong danh mục. MLP chạy bằng train_mlp.py với CLIP. train_gat_kg.py, train_pair_kg.py và train_kg_fusion.py cài các biến thể còn lại.

Dữ liệu và split ở data/raw/vg150/. Cache CLIP ở data/cache/vg150-clip-vit-b32-v1/. Các điều kiện KG cần bộ đặc trưng có manifest khớp với đường dẫn và hash trong danh mục. Các mô hình KG được tạo bằng train_transe.py, train_rotate_v2.py, train_complex.py và các script export tương ứng.

## Suy luận và đánh giá

    python -m task1_vg150.evaluate --checkpoint runs/retrain-vg150/gat-seed42/run_0001/best.pt --visual-cache data/cache/vg150-clip-vit-b32-v1 --split val --protocol reference-v1 --device cpu --batch-size 128 --output runs/retrain-vg150/gat-seed42/val.json

Evaluator dựng mô hình theo config trong checkpoint. Để chấm held-out test sau khi cố định cấu hình, dùng --split test --allow-test. Notebook tổng hợp trình bày 5 Hit và 5 Miss trên validation. preprocess.py tạo split, data.py đọc dữ liệu, models.py dựng MLP, còn common/models/ chứa các lớp mô hình.
