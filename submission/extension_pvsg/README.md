# Mở rộng video · PVSG

Chạy từ gốc submission/ với môi trường PVSG, tương ứng runtime đã ghi trong baselines/runtime.json. Cài môi trường từ environments/pvsg.txt. Video, geometry và CLIP cache nằm ở data/external/pvsg_m2_cache_v4/ và data/external/pvsg_clip_cache_v1/.

## Huấn luyện

train.py chạy một model và một seed. Các baseline gồm mlp, static_gat, st_gat. Nhóm ablation gồm pvsg_conv_k5_l3, mlp_conv_k3_l2, st_no_spatial, st_no_temporal, st_no_edges.

    python -m extension_pvsg.train --model st_gat --seed 17 --output-dir runs/retrain-pvsg/baselines --check
    python -m extension_pvsg.train --model st_gat --seed 17 --output-dir runs/retrain-pvsg/baselines

Cấu hình của kết quả chính dùng seeds 17, 42, 2026, physical/effective batch 32 windows, window 64 frame, hidden 128, 2 blocks, 4 heads, tối đa 150 epoch, tối thiểu 10, patience 15, lr 0.001, weight decay 0.0001, gradient clip 1 và BF16. W&B online. Mỗi lệnh kết thúc sau model/seed đã chọn. Checkpoint lưu dưới output-dir/model/seed-{seed}/.

Các đối chứng và ablation dùng cùng batch và ngân sách:

    python -m extension_pvsg.train --model pvsg_conv_k5_l3 --seed 17 --output-dir runs/retrain-pvsg/ablations

Đối chứng hội tụ muộn dùng tối thiểu 50 epoch cho bảy cặp model/seed trong cases.py:

    python -m extension_pvsg.train --model st_gat --seed 17 --minimum-epochs 50 --output-dir runs/retrain-pvsg/min50

train.py đọc baselines/configs/baselines_v1.json hoặc ablations/configs/ablations_v1.json và áp ngân sách 150 epoch, patience 15 theo các lượt cuối. Các mô hình dùng chung vòng tối ưu trong baselines/runner.py và metric trong baselines/validation.py. Loader kiểm tra hash geometry và đặc trưng của video khi sử dụng cache.

## Đánh giá

    python -m extension_pvsg.evaluate --checkpoint runs/retrain-pvsg/baselines/st_gat/seed-17/best.pt --split validation --device cpu

Evaluator dựng mô hình theo config trong checkpoint và dùng lại validation của vòng train. Evaluator tự chọn model và cách tạo batch từ config đã lưu. Chấm test bằng --split test --allow-test sau khi chọn checkpoint trên validation.

## Tiền xử lý

preprocess.py chuẩn hóa nhãn và cố định split. preprocessing/build_geometry.py tạo geometry từ mask và video. preprocessing/features.py trích CLIP theo vùng track. Các bước chuẩn bị có --help với đầu vào và output tương ứng.

baselines/models.py và ablations/models.py cài mô hình. manifests/ giữ split theo video. Các metadata trong provenance/ được giữ theo các hash dữ liệu mà trainer kiểm tra.
