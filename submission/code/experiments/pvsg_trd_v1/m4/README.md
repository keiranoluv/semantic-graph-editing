# M4 — Đối chứng CVPR 2023 và ablation cạnh

**Đã chuẩn bị code/CPU; chưa chạy GPU profile hoặc train M4.** M4 nằm riêng, dùng lại M2 cache và phần toán/evaluator của M3 đã khóa. Không thay `src/`, M0–M3, split hoặc target policy. Không có KG. Đây là phần thực nghiệm TRD bổ sung của GML, chưa bao phủ chỉnh sửa video/tính nhất quán của toàn bộ đề bài.

## Ưu tiên paper và phạm vi tái lập

Chọn **Yang et al., Panoptic Video Scene Graph Generation, CVPR 2023** vì paper giới thiệu chính PVSG và có temporal relation baseline đơn giản. [CVF paper, §5.2](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_Panoptic_Video_Scene_Graph_Generation_CVPR_2023_paper.pdf) mô tả Conv1D kernel 5, ba layers. [Code đã pin](https://github.com/LilyDaytoy/OpenPVSG/blob/b2850b980c272368215a754299986cb904d55820/models/relation_head/convolution.py) có default `num_layers=1`, cùng `span_head` theo frame và `pred_head` temporal max pooling. [PDF trên trang tác giả](https://jingkangyang.com/PVSG/static/pdfs/pvsg.pdf) là bản cập nhật; phần Conv chỉ nêu kernel 5. M4 chọn tường minh **kernel 5 × 3 layers theo CVF**, không suy diễn default của code thành cấu hình paper.

Tên báo cáo: **PVSG-inspired temporal Conv — GT-track/CLIP adaptation**. Không gọi kết quả M4 là tái lập benchmark end-to-end hoặc so số trực tiếp với bảng paper. Những thay đổi có chủ đích:

- Dùng GT masks/labels/track IDs và CLIP đóng băng thay predicted IPS/VPS query tubes.
- Dùng node MLP/pair projection của M3, rồi Conv trên pair sequence; head 57 kênh theo frame, masked BCE và decoder M1 chung. Không triển khai nguyên hai heads/pair selection của release.
- Giữ split tác giả **338/62**, nội bộ **287 fit/51 validation/62 test**; giữ policy `pvsg-gml-valid-support-v1`. CVF ghi 360/40 nên không thể gọi đây là đúng protocol CVF.
- Chỉ relation prediction; không train segmentation/tracking. Nhánh tái lập gốc `IPS+T → Conv` cần artifacts và môi trường riêng, chưa nằm trong launcher này.

[Sources/provenance](provenance/sources.json) khóa commit upstream và phạm vi chuyển thể; [license upstream](OPENPVSG_LICENSE.txt) giữ điều khoản S-Lab cho mục đích phi thương mại. Code Conv được triển khai trong module này; không kéo runtime OpenMMLab cũ vào environment M3.

## Các model đã giữ lại

| CLI model | Cấu hình | Parameters | Vai trò |
|---|---|---:|---|
| `pvsg_conv_k5_l3` | MLP pair features → Conv1D k5 × 3 → 57 logits | 437.945 | Ưu tiên pilot seed 17, sau đó ba seeds nếu đưa vào bảng chính |
| `mlp_conv_k3_l2` | Cùng cách dựng pair, Conv1D k3 × 2 | 290.361 | Tùy chọn; receptive field 5 frame, cùng bán kính thời gian với ST-GAT M3 |
| `st_no_spatial` | ST-GAT bỏ spatial message edges | 226.873 | Tùy chọn; giữ temporal edges và self-loops |
| `st_no_temporal` | ST-GAT bỏ temporal message edges | 226.873 | Tùy chọn; giữ spatial edges và self-loops |
| `st_no_edges` | ST-GAT bỏ cả hai loại message edges | 226.873 | Tùy chọn; chỉ self-loops, cùng số layers/parameters |

Full ST-GAT đối chứng lấy từ **M3** (226.873 params). Các ablation chỉ thay adjacency khi encode; giữ mọi module, query pairs, inputs, labels/ignore masks, loss và decoder. `st_no_temporal` không đồng nghĩa Static GAT M3: nó còn temporal block hoạt động qua self-loops, giúp kiểm tra topology trên cùng kiến trúc. Không bỏ các lựa chọn spatial/temporal khỏi code và không tự xếp chúng vào queue.

Conv ưu tiên có receptive field tối đa **13 frame**; control k3 × 2 có **5 frame**. Window 64 frame không có nghĩa mọi node/pair nhìn đủ 64 frame. Các Conv có nhiều parameters hơn GAT, vì vậy bảng kết quả phải ghi parameters/RF; đây không phải đối chứng khớp hoàn toàn dung lượng model.

## Input/output và xử lý thời gian

Input giống M3: visual CLIP 512 chiều, numeric/time/availability 15 chiều, class embedding 64; hidden 128. MLP tạo node embeddings rồi descriptor `[h_s, h_o, h_s*h_o, h_s-h_o, pair_geometry]`, chiếu về 128 và ReLU. Conv xử lý theo từng **window + directed track pair + đoạn cùng xuất hiện liên tục**, sau đó dropout và linear 57 chiều.

Collate xây sequence chỉ từ IDs/frames/visibility, không dùng GT relation để chọn pair hoặc cắt interval. Pair đảo chiều có sequence riêng; hai windows cùng video vẫn tách biệt. Gap visibility ngắt sequence, không nén gap rồi xem hai frame xa nhau là liền kề. RGB thiếu nhưng mask còn hợp lệ giữ quy tắc availability của M2/M3.

Các đoạn được gom theo chiều dài lũy thừa hai để giảm padding. Padding bằng zero và được mask lại **sau từng Conv layer**, tránh bias tại vùng padding quay lại frame thật. Không cắt temporal context theo loss chunk. Conv encoder trả embeddings của toàn bộ pair-frame rows; head/loss vẫn chunk 4.096 rows và backprop encoder một lần qua engine M3. Conv activations vẫn tăng theo tổng queries, vì vậy **phải profile riêng**, không dùng batch đo cho GAT.

Output: 57 logits/query/frame. Ghép mean logits ở overlap, sigmoid threshold 0,5, gom các đoạn active, rank tuple theo M1. Validation phủ toàn bộ 51 video; đánh giá R/mR@20/50/100 tại true tIoU 0,3/0,5/0,7. 62 test video không được đánh giá trong training. Không đổi threshold/loss theo từng model.

## Cấu hình train và kiểm tra hội tụ

Các hyperparameters chung kế thừa M3 và được kiểm tra không lệch trong `common.config()`: window 64, stride val 32, bốn windows/fit video/epoch, effective batch **32 windows**, AdamW LR 1e-3/weight decay 1e-4, BF16, clip norm 1, dropout 0,1. Tối đa **50 epochs**, early stopping sau tối thiểu 10 epochs, patience 10 theo validation mR@50 tại tIoU 0,5 (tie-break R@50, rồi epoch sớm nhất). LR scheduler theo validation BCE, patience 3/factor 0,5.

Physical batch chưa chốt: profile từng model ở **1/2/4/8/16/32 windows**, accumulation = 32 / physical batch. Giữ peak reserved trong min(80% tổng VRAM, 90% VRAM đang trống); chọn batch nhỏ nhất đạt ít nhất 95% throughput tốt nhất trong những batch đạt giới hạn. Probe gồm envelope 38 objects × 64 frames và windows thật từ fit. Timing không bao gồm CPU loader; main run ghi cả loader wait và thời gian epoch.

Loss dương/âm, frame P/R/F1, active fraction, gradient norm và cảnh báo dự đoán toàn âm đều được log. Do lớp dương rất thưa, train BCE giảm chưa đủ kết luận hội tụ. CPU overfit hidden 128 trên fixture 4 frames, 200 steps, tắt dropout/weight decay, đã học được positives ở cả năm model; [kết quả](provenance/cpu_overfit.json). Đây chỉ là kiểm tra khả năng ghi nhớ, không chứng minh generalization. GPU gate còn yêu cầu tiny-overfit trên **fit window thật** trước train chính.

## Lệnh chạy và phạm vi

Dùng lại environment **`data/external/pvsg_m3_env`**, không sửa environment M2 đang extraction. Chạy kiểm tra CPU trong lúc chờ được:

```bash
CUDA_VISIBLE_DEVICES='' data/external/pvsg_m3_env/bin/python -m experiments.pvsg_trd_v1.m4.preflight --cpu-only
CUDA_VISIBLE_DEVICES='' data/external/pvsg_m3_env/bin/python -m pytest -q experiments/pvsg_trd_v1/m4
```

Sau M2 hoàn tất, lúc GPU rảnh, đo **một model ưu tiên**, seed 17:

```bash
bash experiments/pvsg_trd_v1/m4/launch_profile.sh --model pvsg_conv_k5_l3
```

Phạm vi: seed 17, quét batch nêu trên; mỗi case một warmup + ba bước đo forward/backward/AdamW; overfit nhỏ 200 bước. Output **`reports/pvsg-trd-m4-v1/profiles/pvsg_conv_k5_l3.json`**. Test không dùng WandB. Probe xóa `.pt` của chính nó trong `finally`; không đụng pretrained CLIP hoặc main weights. **Không tự train tiếp.** Đã có profile thì từ chối ghi đè; dùng `--output <profile_moi.json>` nếu cần đo lại.

Sau khi xem VRAM, throughput và overfit trong profile, người dùng chạy **một main run**:

```bash
bash experiments/pvsg_trd_v1/m4/launch_train.sh --model pvsg_conv_k5_l3 --seed 17
```

Phạm vi: model `pvsg_conv_k5_l3`, seed **17**, physical BS lấy từ profile đã đạt, effective BS **32**; epochs/early stopping như trên. Output **`reports/pvsg-trd-m4-v1/pvsg_conv_k5_l3/seed-17/`**. WandB **online bắt buộc**, project `pvsg-trd-v1`, group `pvsg-gt-trd-m4-v1`. GPU mặc định 0, đổi bằng `CUDA_VISIBLE_DEVICES`. **Một lệnh một run; không tự chạy seed/model kế tiếp.** M4 dùng chung GPU mutex với M3, tránh hai main runs chạy đồng thời. Chỉ chạy khi queue M3 đã xong hoặc được người dùng dừng.

Mỗi run giữ `best.pt`, `last.pt`, history, validation reports và completion identity. Đây là checkpoint train chính, được giữ. Resume cùng lệnh từ epoch đã ghi đầy đủ; code/config/cache/profile khác bị từ chối. CUDA scatter/cuDNN có thể không bitwise deterministic; cần ba seeds nếu đưa vào bảng so sánh. Profile custom phải truyền lại `--profile <path>` khi train.

Muốn chạy ablation, thay `--model` bằng một tên trong bảng và **profile riêng trước**. Không có queue tự chạy năm models. Sau pilot, chạy thủ công seeds 42 và 2026 với cùng code/config nếu đưa model vào bảng chính; không chọn riêng seed tốt nhất. Tổng hợp internal validation:

```bash
data/external/pvsg_m3_env/bin/python -m experiments.pvsg_trd_v1.m4.summarize --model pvsg_conv_k5_l3
```

Summary ghi seeds còn thiếu; chỉ có một seed thì không báo sample std. Đây là điểm chọn checkpoint trên internal-val, chưa phải held-out test M5.

## Nghiệm thu và phần còn lại

[Verification](provenance/verification.json) ghi CPU tests, kiểm tra real cached window, kiểm tra bảo toàn M0–M3 và cleanup weights. Tests bao gồm topology/parameter controls, candidate/target invariance, temporal receptive field, gap/reverse-pair/window isolation, chunk/accumulation gradient equivalence, empty queries, multiprocessing collate, validation không đưa target vào model, profile identity/headroom, mandatory online logging và checkpoint resume.

M4 **sẵn sàng về code/CPU**, chưa có batch GPU được chọn, thời gian epoch hoặc benchmark. Bước tiếp theo: hoàn tất M2 → chạy M3 theo kế hoạch → profile/pilot Conv M4 → quyết định chạy thêm hai seeds/ablation → khóa config để đánh giá test ở M5. Không đổi loss/decoder để cứu riêng một model trong cùng phiên bản; nếu phải chỉnh sau pilot, tạo version mới và chạy các đối chứng tương ứng.
