# M3 — MLP, Static GAT, ST-GAT

**Trạng thái: code và kiểm tra CPU sẵn sàng; chờ M2 extraction hoàn tất để đo GPU và chốt physical batch.** Chưa train chính, chưa có điểm held-out PVSG. M0/M1/M2 và `src/` giữ nguyên. M3 mở rộng relation prediction cho GML, không thay thế nhiệm vụ graph editing/semantic consistency trong `Project01_Graph_Editing.pdf`; không có KG trong nhánh video theo lựa chọn của người dùng.

## Cấu hình ban đầu và cách chốt tài nguyên

Máy hiện tại: **RTX 5090, 32.607 MiB VRAM (~31,84 GiB)**, RAM khoảng 60 GiB. Đây là dung lượng phần cứng, chưa phải VRAM M3 đã đo. Không chạy GPU probe khi extractor M2 còn hoạt động hoặc chưa có summary đủ 400 video.

| Thành phần | Cấu hình / cách quyết định |
|---|---|
| Encoder thị giác | CLIP ViT-B/32 đóng băng, dùng cache M2; không đưa encoder vào optimizer |
| Model | MLP 191.801 params; Static GAT 192.313; ST-GAT 226.873 |
| Input / window | 512 chiều visual + 15 numeric/time/availability + embedding class 64; 64 frame/window |
| Hidden / GAT | Hidden 128, 4 heads, 2 blocks, dropout 0,1; spatial và temporal weights riêng |
| Physical batch | Probe riêng từng model trong **1, 2, 4, 8, 16, 32 windows**; chưa có giá trị đã duyệt |
| Effective batch | **32 windows** cho cả ba model; accumulation = 32 / physical batch |
| Pair head | Chunk 4.096 ordered pair-frame queries, đủ mọi cặp, không sampling âm |
| Precision | CUDA BF16; optimizer/weights FP32; CPU checks FP32 |
| Bộ nhớ | Peak **reserved** không quá min(80% total VRAM, 90% free VRAM trước probe) |
| Optimizer | AdamW, LR 1e-3, weight decay 1e-4, clip gradient norm 1,0 |
| Epoch | Tối đa **50**, tối thiểu **10** trước early stop; patience **10** validation epochs |
| LR schedule | ReduceLROnPlateau theo validation BCE; patience 3, factor 0,5, min LR 1e-5 |
| Checkpoint selection | Internal-val mR@50 tại tIoU=0,5; tie-break R@50, rồi epoch sớm hơn |
| Loader | 4 CPU workers, prefetch 2, pin memory, spawn, persistent workers; LRU 2 video/worker |

**Batch M3 tính bằng cửa sổ video**, không phải số crop CLIP của M2. Số node và query trong một window thay đổi theo visibility. 38 object × 64 frame có 2.432 nodes và 89.984 queries; pair head/graph activations quyết định phần lớn VRAM dù model chỉ khoảng 0,2 triệu parameters.

GPU probe dùng hai dạng input: (1) envelope tổng hợp gồm 38 object hiện diện ở mọi frame, chặt hơn các window đã audit; (2) windows thật từ fit ở các quantile mật độ. Số 38 lấy từ audit đầu vào M2, không lấy test relation labels. Các batch được đo forward/backward/AdamW sau một warmup, ba bước đo. Chọn batch nhỏ nhất đạt ít nhất 95% throughput tốt nhất trong các batch đủ khoảng trống bộ nhớ. Không cố chiếm đầy VRAM khi throughput không tăng.

Timing probe là **resident-tensor throughput**, chưa bao gồm DataLoader/đọc disk. Main epochs ghi riêng `loader_wait_seconds`, windows/s và total epoch time để nhận ra CPU/I/O bottleneck; không suy ra thời gian train chỉ từ phần trăm GPU utilization. Cấu hình chưa được gọi là tối ưu cho tới khi có số đo thực. Nếu cần đổi LR, effective batch, loss hoặc epoch budget sau pilot, tạo experiment/config version mới và profile lại; không trộn với runs đã bắt đầu.

## Kiến trúc và supervision

Ba model có chung node projection, pair features và head 57 sigmoid channels. MLP chỉ cập nhật từng node; Static GAT truyền spatial context trong frame; ST-GAT xen hai spatial và hai temporal GAT layers. Temporal edges nối cùng track ở frame liền nhau, hai chiều, có signed delta-time, không nối qua gap. Mọi model cùng nhận position/time input; track ID chỉ là khóa tra cứu, không phải feature số.

Reuse M1 `Graph`, batching/evaluator/interval conventions và M2 geometry/feature artifacts. Window topology được vector hóa trong M3, có test đối chiếu từng node/cạnh với M1. Sparse targets có test đối chiếu `rasterize_adapted_targets`: bỏ negative supervision ở những key đã quarantine thời gian nhưng giữ positives hợp lệ. Inference không đọc GT relations để dựng graph/chọn query. Sparse streaming decoder được đối chiếu chính xác với M1 về overlap mean logits, visibility, nhiều intervals và rank.

Loss v1 là **unweighted masked BCE**: lấy mean trên các labels hợp lệ trong từng window, rồi mean trên các windows của một effective batch. Cách này tránh window đông object tự có trọng số lớn hơn. Toàn bộ candidate queries vẫn được xét. Pair head backward theo chunk, tích lũy gradient vào node embeddings trung gian rồi backprop encoder đúng một lần; tests đối chiếu gradient với head không chunk và accumulation không đều.

Mỗi epoch sample **4 windows/fit video**, start uniform, chỉ phụ thuộc seed/epoch/video length; không ưu tiên window bằng nhãn GT. Có 287 × 4 = **1.148 windows/epoch**, tương ứng 36 optimizer steps ở effective batch 32 (group cuối 28 windows). Các video ngắn có thể lặp window. Epoch ở đây là một lượt sampler này, không phải duyệt toàn bộ frames của fit. Internal validation chạy phủ toàn bộ 51 video: **545 windows**, stride 32, ghép toàn timeline rồi mới rank/evaluate. 62 held-out video không được đánh giá hoặc dùng để chọn hyperparameters.

## Kiểm tra hội tụ

Thống kê **fit-only** có 395.011 nhãn dương trên 693.263.184 nhãn được giám sát (57 kênh × ordered pair-frame queries sau ignore mask), khoảng **0,057%**. Đây không phải tỷ lệ video/pair có quan hệ. [Fit statistics](provenance/fit_statistics.json).

Loss giảm chưa đủ để nói model học được relation. Code ghi riêng positive/negative BCE, frame precision/recall/F1, tỷ lệ active predictions, gradient norm, LR, validation BCE và R/mR/tIoU. `all_negative_prediction_warning` chỉ ra epoch không xuất relation nào. Early stopping dựa trên metric chính, không dùng train loss; LR scheduler dùng validation BCE và được ghi riêng.

CPU đã kiểm tra overfit fixture bằng **kiến trúc hidden 128 thực**, 200 bước, LR 1e-3, tắt dropout/weight decay trong diagnostic. Cả ba model đạt frame F1=1 trên fixture rất nhỏ. Cấu hình nhỏ hidden 16 thử trước đó từng giảm total BCE nhưng đoán toàn âm; kết quả đó là lý do giữ tiêu chí positive BCE/recall trong gate, không hạ tiêu chí để che collapse. [CPU overfit evidence](provenance/cpu_overfit.json).

[CPU window thật](provenance/cpu_real_window.json) kiểm tra alignment và gradient trên shard M2 đã hoàn tất: cả ba model forward/backward hữu hạn; không cập nhật optimizer để coi là training thật, không khởi tạo CUDA. Runner cũng được kiểm tra interruption/resume/completed-skip trên fixture CPU với tracking giả lập, không tạo online run. Các con số loss/F1 của lượt này là model chưa học, không phải benchmark.

GPU probe còn phải overfit **một window fit PVSG thật dài 16 frames**, bounded query count, 200 bước/model; tắt dropout/weight decay và ghi rõ diagnostic khác train chính. Gate đòi loss giảm ít nhất 30%, positive BCE giảm và positive recall >0. Đây chỉ là kiểm tra khả năng học/memorization, không chứng minh generalization hoặc bảo đảm hội tụ trên full dataset. Nếu gate không đạt, launcher train chính từ chối chạy; điều tra loss imbalance, gradient, labels và LR trước khi sửa recipe có version. Không tự tăng class weights hoặc chọn threshold theo test.

## Các lệnh chạy, từ repo root

Environment dùng `data/external/pvsg_m3_env/bin/python`, là overlay riêng cho M3: đọc PyTorch/PyG/NumPy từ các môi trường đã có, pin riêng **protobuf 5.29.5 và setuptools 78.1.0** (cần distutils shim cho WandB trên Python 3.12). WandB 0.18.7 trong môi trường kế thừa yêu cầu protobuf <6; bản protobuf 7 ở môi trường cũ gây ImportError. Không cài/sửa package trong environment M2 đang chạy. [Requirements](requirements.txt) và [runtime lock](runtime.json). Không sửa environment/source cũ.

Kiểm tra CPU trong lúc M2 chạy:

```bash
data/external/pvsg_m3_env/bin/python -m experiments.pvsg_trd_v1.m3.preflight --cpu-only
data/external/pvsg_m3_env/bin/python -m pytest -q experiments/pvsg_trd_v1/m3
```

M2 extraction đang do người dùng chạy; nếu cần resume, lệnh cũ vẫn là:

```bash
bash experiments/pvsg_trd_v1/m2/launch_extract.sh
```

**Sau M2, chỉ probe; không tự train chính:**

```bash
bash experiments/pvsg_trd_v1/m3/launch_profile.sh
```

Scope: GPU mặc định 0 (`PVSG_GPU` để đổi), ba model tuần tự; seed 17; quét tối đa sáu batch/model; tối đa 4 optimizer steps/case (1 warmup + 3 đo), hai case/batch; overfit nhỏ 200 bước/model. Test không cần WandB. Output `reports/pvsg-trd-m3-v1/gpu_profile.json`; lưu peak allocated/reserved, throughput, batch và accumulation đề xuất, kết quả overfit. **Mọi `.pt` do probe tạo được xóa trong `finally`**, kể cả khi lỗi; pretrained CLIP và checkpoints của thực nghiệm khác giữ nguyên. Probe lưu checksum kết quả checkpoint roundtrip để truy vết. Không có queue train tiếp sau probe.

Main preflight chỉ đạt khi có đủ 400 feature shards, hash đúng, M3 code đã commit, runtime phù hợp và GPU profile trạng thái `passed` khớp code/config/features. Trước mỗi main run, VRAM đang trống phải cao hơn peak reserved đã đo ít nhất 10%. Profile có sẵn được giữ nguyên; muốn đo lại dùng `launch_profile.sh --output <profile_moi.json>`, sau đó truyền profile đó vào launcher train.

**Train chính chỉ sau khi đã xem/chốt kết quả profile:**

```bash
bash experiments/pvsg_trd_v1/m3/launch_train.sh
```

Scope: **9 runs tự chạy tuần tự** — MLP seeds 17,42,2026 → Static GAT seeds 17,42,2026 → ST-GAT seeds 17,42,2026. Một model/run mỗi thời điểm; lỗi thì queue dừng. Physical batch lấy riêng từng model từ profile, effective batch 32, window 64; epoch/LR theo bảng trên. GPU mặc định 0. Output `reports/pvsg-trd-m3-v1/<model>/seed-<seed>/`. WandB **online bắt buộc**, project `pvsg-trd-v1`; lỗi khởi tạo logging thì không train. Không truyền API key qua code/chat; dùng login local của WandB nếu môi trường chưa đăng nhập.

Lượt riêng nếu chưa muốn chạy cả queue:

```bash
PVSG_GPU=0 CUDA_VISIBLE_DEVICES=0 data/external/pvsg_m3_env/bin/python -m experiments.pvsg_trd_v1.m3.runner --model mlp --seed 17
```

Profile khác: `bash experiments/pvsg_trd_v1/m3/launch_train.sh reports/pvsg-trd-m3-v1/gpu_profile_v2.json`. Việc thay profile/code/config khi một run đã tồn tại sẽ bị từ chối; không âm thầm resume sang cấu hình mới.

Mỗi run lưu `last.pt` để resume và `best.pt` theo metric chọn checkpoint; **đây là weights train chính nên được giữ**, khác weights của probe phải xóa. Checkpoint có optimizer/scheduler, epoch, RNG và identity; resume từ epoch đã ghi đầy đủ, không từ giữa epoch. Torch GPU scatter có thể không bitwise deterministic; ghi giới hạn này và chạy ba seeds. Queue chạy lại sẽ xác minh/bỏ qua run đã hoàn tất và resume run đang dở. Sau đủ chín runs, `validation_summary.json` ghi mean/sample std của điểm **internal-val**, chưa phải kết quả test M5.

## Nghiệm thu hiện tại

CPU regression bao gồm graph/targets equivalence với M1/M2, batch isolation, track-ID invariance, temporal message passing, chunk/accumulation gradients, sparse decoder, full validation coverage, multiprocessing sampler, checkpoint/RNG resume, early stopping và overfit positives. [Verification](provenance/verification.json) lưu số tests và phạm vi. Test `.pt` đã được xóa; chưa có `.pt` train chính.

**Chưa đo:** physical batch tốt nhất, VRAM thực của từng model, GPU throughput, thời gian epoch đầy đủ, hội tụ trên PVSG fit/validation. Các phần này chờ M2 hoàn tất; không dùng phép overfit fixture để tuyên bố model sẽ thắng baseline hoặc đạt điểm paper.
