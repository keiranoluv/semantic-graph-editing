# Ablation M4 trên GT tracks

Chạy từ thư mục gốc repo. Launcher gọi lại m4_150_bs32 để dùng đúng cache, split 287/51/62, sampler, loss, decoder, evaluator và cấu hình train đã dùng cho Conv k5 × 3. Batch vật lý và hiệu dụng đều là 32 cửa sổ, tối đa 150 epoch, patience 15. Chọn checkpoint theo mR@50 trên validation ở tIoU 0,5, dùng R@50 khi hòa.

| Model | Đối chiếu theo cùng seed |
|---|---|
| st_no_temporal | So với ST-GAT M3 để đo ảnh hưởng của cạnh thời gian. |
| st_no_spatial | So với ST-GAT M3 để đo ảnh hưởng của cạnh không gian. |
| st_no_edges | Giữ self-loops và toàn bộ layers của ST-GAT. So với ba cấu hình cạnh còn lại. |
| mlp_conv_k3_l2 | Conv kernel 3 × 2 có ngữ cảnh tối đa 5 frame, cùng độ rộng với ST-GAT. So thêm với MLP và Conv k5 × 3 (13 frame). Số tham số và cách tổng hợp đặc trưng khác nhau. |

Ba ablation cạnh giữ nguyên các cặp cần dự đoán, nhãn và số tham số của ST-GAT. Báo mR@50, R@50 và epoch được chọn cho từng seed, kể cả seed dừng sớm với recall bằng 0. Tổng hợp mean và sample std sau đủ ba seed.

```bash
bash extension_pvsg/m4_ablations/launch.sh plan
bash extension_pvsg/m4_ablations/launch.sh train
```

Lệnh train chờ lượt M4 đang giữ GPU, sau đó chạy tuần tự theo thứ tự bảng, mỗi model với seeds 17, 42, 2026. Mỗi model cần qua profile batch 32 và thử học trên cửa sổ fit trước khi train. Profile hợp lệ đã có được dùng lại. Probe dọn checkpoint thử. Train dùng WandB online, giữ best.pt và last.pt ở reports/pvsg-trd-m4-150-bs32-v1/<model>/seed-<seed>/. Khi lỗi queue dừng. Chạy lại cùng lệnh để xác minh các lượt đã xong và tiếp tục từ checkpoint.

Dùng `launch.sh train 17` để chạy bốn model với riêng seed 17, hoặc `launch.sh profile` để chỉ đo cả bốn model rồi dừng.
