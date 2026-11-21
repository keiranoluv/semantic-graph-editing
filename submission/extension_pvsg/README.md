# Mở rộng video · PVSG

Chạy từ gốc submission/ với môi trường PVSG, tương ứng runtime đã ghi trong m3/runtime.json. Video, geometry và CLIP cache nằm ở data/external/pvsg_m2_cache_v4/ và data/external/pvsg_clip_cache_v1/.

## Huấn luyện

train.py chạy một model và một seed. Các mô hình M3 gồm mlp, static_gat, st_gat. M4 gồm pvsg_conv_k5_l3, mlp_conv_k3_l2, st_no_spatial, st_no_temporal, st_no_edges.

    python -m extension_pvsg.train --model st_gat --seed 17 --output-dir runs/retrain-pvsg/m3 --check
    python -m extension_pvsg.train --model st_gat --seed 17 --output-dir runs/retrain-pvsg/m3

Cấu hình của kết quả chính dùng seeds 17, 42, 2026, physical/effective batch 32 windows, window 64 frame, hidden 128, 2 blocks, 4 heads, tối đa 150 epoch, tối thiểu 10, patience 15, lr 0.001, weight decay 0.0001, gradient clip 1 và BF16. W&B online. Mỗi lệnh kết thúc sau model/seed đã chọn. Checkpoint lưu dưới output-dir/model/seed-{seed}/.

Các đối chứng M4 dùng cùng batch và ngân sách:

    python -m extension_pvsg.train --model pvsg_conv_k5_l3 --seed 17 --output-dir runs/retrain-pvsg/m4

Đối chứng hội tụ muộn dùng tối thiểu 50 epoch cho bảy cặp model/seed trong cases.py:

    python -m extension_pvsg.train --model st_gat --seed 17 --minimum-epochs 50 --output-dir runs/retrain-pvsg/min50

train.py đọc m3/configs/baselines_v1.json hoặc m4/configs/ablations_v1.json và áp ngân sách 150 epoch, patience 15 theo các lượt cuối. Vòng tối ưu nằm trong m3/runner.py và m4/runner.py. Loader kiểm tra hash geometry và đặc trưng của video khi sử dụng cache.

## Đánh giá

    python -m extension_pvsg.m5.run check
    python -m extension_pvsg.m5.run evaluate

m5/run.py chấm danh mục checkpoint lịch sử đã chọn trên validation, gồm bảng chính và bảy lượt min50. Cần các artifact và metadata tương ứng tại đường dẫn đã khóa trong evaluator. Với lượt train mới, chọn checkpoint từ validation và lập danh mục đánh giá riêng theo cùng evaluator.

m2/ tạo đặc trưng vùng và track. m3/models.py, m4/models.py cài mô hình. manifests/ khóa split theo video. provenance/ lưu nguồn dữ liệu và metadata thực nghiệm gốc.
