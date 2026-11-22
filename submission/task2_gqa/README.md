# Nhiệm vụ 2A · GQA

Chạy từ gốc submission/. Cài environments/gqa-cu128.txt, đăng nhập W&B và chuẩn bị data/processed/level2/gqa-v1/manifest.json, cache data/cache/gqa-full-clip-v1/ cùng GloVe data/processed/level2/gqa-glove-v1/.

## Huấn luyện PE-PLPR và AlignG

    python -m task2_gqa.training.runner --stage check --loader-workers 4
    python -m task2_gqa.training.runner --stage train-all --loader-workers 4

Runner dùng configs/level2/aligng-gqa-fast-v1-bs512.json. Batch train và validation đều 512, lr 0.001, weight decay 0.01, tối đa 200 epoch, patience 15. Sáu lượt chạy tuần tự, mỗi seed 42, 123, 3407 lần lượt train PE-PLPR + AlignG rồi PE-PLPR. W&B online, output ở runs/level2/aligng-gqa-fast-v1/full/. Runner tiếp tục hàng đợi khi lượt hiện tại hoàn tất.

training/model.py cài mô hình, training/engine.py cài vòng tối ưu, configuration.py chuẩn bị cấu hình và dữ liệu; validation.py tính metric dùng trong train, models/ chứa khối AlignG. dataset.py chứa loader RAM mà lượt full sử dụng.

## Baseline MLP, GAT và PE-Net

    python -m task2_gqa.train_baselines --stage train --config configs/level2/gqa-full-v3-bs512.json --model gat --seed 42

Lệnh trên chạy một model/seed, output theo trường output của config. Đổi --model thành mlp, pe-ce, pe-pl hoặc pe-plpr. Cấu hình cố định batch 512 và seeds 42, 123, 3407. validation.py cung cấp cách dựng baseline và tính metric. knowledge.py dựng GAT + KG để đọc checkpoint của phần đối chứng tri thức.

## Suy luận và đánh giá

    python -m task2_gqa.evaluate --checkpoint runs/level2/aligng-gqa-fast-v1/full/pe-plpr-aligng/42/best.pt --model pe-plpr-aligng --backend fast --split val --max-images 2 --device cpu

Lệnh đọc một checkpoint và chấm trên cache đã có. predict.py dựng mô hình từ checkpoint. validation.py dùng chung cách tính metric cho train và đánh giá. Notebook tổng hợp hiển thị 5 Hit và 5 Miss, dùng danh mục hai checkpoint trong assets/gqa_checkpoints.json.

## Chuẩn bị lại dữ liệu

Các bước này tạo đầu vào cho train. Với bộ đầu vào đã khóa, dùng trực tiếp manifest, cache và GloVe ở các đường dẫn bên trên.

    python -m task2_gqa.preprocessing.graphs --root data/raw/gqa --seed 42 --val-size 5000
    python -m task2_gqa.preprocessing.glove --help

| Mã chuẩn bị | Đầu ra |
|---|---|
| preprocessing/graphs.py | Split theo ảnh, từ vựng train và đồ thị chuẩn hóa. |
| preprocessing/glove.py | Vector GloVe cho nhãn đối tượng và quan hệ. |
| preprocessing/wordnet_mapping.py | Mapping nhãn và đồ thị WordNet. |
| preprocessing/knowledge.py | Đặc trưng KG từ checkpoint TransE và mapping đã duyệt, dùng helper kiểm tra/export trong export_wordnet.py. |

Các quyết định duyệt và manifest nằm trong dữ liệu thực nghiệm.

knowledge_features.py chỉ đọc và kiểm tra tensor KG đã xuất khi chấm checkpoint GAT + KG. Đường train/inference dùng bộ đọc này độc lập với các CLI chuẩn bị dữ liệu.
