# AlignG fast v1: kiểm tra tối ưu GQA batch 512

Implementation commit: `734ef130a8c409c323a93b50b22c4fd8cf7f80a7`.
Experiment: `ea05085cd3eea85f951deb42e51e71eda2a9322516b8f5bb57042c9a8427deda`.
GPU: NVIDIA GeForce RTX 5090. Float32, TF32/AMP tắt.

135 test CPU PASS; pre-commit hooks PASS. Kiểm tra bao gồm output/embedding/gradient C=51/311, mask và ảnh độc lập, sampling/RNG, checkpoint/dropout, exact resume trong fast v1, queue 6 lượt và W&B bắt buộc. C=51 là tensor test, không phải thực nghiệm VG150.

Mỗi case: 512 ảnh thực, 1 warmup + 2 step đo; bản cũ/mới chạy tuần tự trên cùng GPU và initialization. Train dùng AdamW tạm; không ghi checkpoint train, không W&B, không dùng test split.

| Model | Case | Cũ (s/batch) | Mới (s/batch) | Tăng tốc | Peak allocated (GiB) |
|---|---|---:|---:|---:|---:|
| pe-plpr-aligng | train_max_candidates | 5.9089 | 0.4582 | 12.90× | 8.949 |
| pe-plpr | train_max_candidates | 0.3355 | 0.1650 | 2.03× | 7.688 |
| pe-plpr-aligng | train_max_sampled | 5.8741 | 0.4538 | 12.94× | 8.977 |
| pe-plpr | train_max_sampled | 0.3350 | 0.1674 | 2.00× | 7.793 |
| pe-plpr-aligng | val_max_candidates | 1.2562 | 0.1914 | 6.56× | 2.577 |
| pe-plpr | val_max_candidates | 0.3470 | 0.1769 | 1.96× | 2.602 |

AlignG validation: max abs logit difference `8.58306884765625e-06`; baseline `0.0`. Giữ nguyên công thức, parameter schema, sampler và loss; vector hóa làm đổi thứ tự dropout, nên không khẳng định hai quá trình train cho cùng checkpoint.

Giá trị peak allocated tối đa của AlignG tăng từ khoảng 6,21 lên 8,98 GiB; peak reserved mới tối đa 10,16 GiB. Đổi lại bỏ overhead checkpoint attention nhỏ và hàng trăm vòng encoder. Batch 512 còn dư bộ nhớ trên GPU 32 GiB đã đo.

Thời gian gồm collate CPU, transfer, forward, loss, backward/optimizer (train). Không tính chuẩn bị graph một lần lúc nạp RAM, trace hash của trainer, checkpoint I/O, W&B hoặc ranking metric CPU. Khoảng 137 batch/epoch × 0,454–0,458 s ≈ 62–63 s cho phần đo tương ứng; đây không phải thời gian epoch end-to-end và không phải cam kết tốc độ trên phần cứng khác.

Queue cũ được dừng bằng SIGTERM theo xác nhận người dùng; checkpoint `aligng-gqa-bs512-v1/full/pe-plpr-aligng/42/last.pt` đã lưu epoch 2, next_index 39936, total_batches 215, best_epoch 1, đầy đủ optimizer/sampler/CPU-CUDA RNG. Không tự chạy lại train chính. Source identity cũ vẫn khớp experiment đã lưu. Mọi file tracked có trước tác vụ giữ nguyên byte (kể cả thay đổi cá nhân AGENTS.md và notebook).

Launcher mới: `bash scripts/run_aligng_gqa_fast.sh`. Train mới bắt đầu seed 42 ở epoch 1; không tự nạp checkpoint cũ. Queue AlignG/PE-PLPR cho seed 42, 123, 3407, train/val batch 512, online W&B, một model mỗi thời điểm. Output `runs/level2/aligng-gqa-fast-v1/full/`. Có thể resume các lượt fast v1 bằng cùng lệnh.
