# PE-Net adapted — triển khai và GQA pilot v1

**Đã port prototype head và PL/PR, chạy ba biến thể trên dữ liệu thật, kiểm
checkpoint và benchmark graph lớn.** Đây là pilot 512 train / 128 validation,
seed 42, 5 epoch; chưa phải thực nghiệm chính ba seed hoặc tái lập điểm paper.
PE-PL đạt GC mR@50 cao nhất trong ba biến thể ở lượt này. Thêm PR chưa cải thiện
chỉ số chọn checkpoint; không bỏ hoặc sửa kết quả bất lợi này.

## 1. Phương pháp và thay đổi được công bố

Nguồn: [PE-Net, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Zheng_Prototype-Based_Embedding_Network_for_Scene_Graph_Generation_CVPR_2023_paper.html),
[code pin `9c9f507`](https://github.com/VL-Group/PENET/tree/9c9f50777c66647799cb7a17dfb855aa98aecd1e).
Giữ GloVe object/predicate embeddings có học, phép chiếu subject/object riêng,
gated visual residual, fusion `ReLU(s+o)−(s−o)²`, trừ gated pair visual,
residual/norm và shared projection. Logits là cosine với prototype, temperature
có học. Notice MIT được giữ tại [PENET-MIT.txt](../../licenses/PENET-MIT.txt).

Thay đổi đã chọn trong [quyết định phương pháp](../../docs/LEVEL2_METHOD_DECISION.md):

- Input là cache CLIP 512D + geometry 5D đóng băng, GT box/object labels theo PredCls.
  Không đưa GT relation vào forward. Pair visual là MLP trên concat có thứ tự hai
  object, **không phải union-box crop**. Không dùng checkpoint train VG150.
- Semantic width 256, projection 512; **2.467.109 tham số học được** mỗi biến thể.
  Số lớp động: GQA 311 gồm background; test có cả 51 và 311 lớp.
- Target đa predicate là soft distribution đều. PL loại tất cả predicate đúng
  khỏi tập hard negative; dòng GT trùng không tăng trọng số train.
- Dùng code tham chiếu cho PR cosine `L21/C²`, detach nhánh prototype đích và
  trung bình hinge từng prototype; `k1=10, k2=1, gamma1=1, gamma2=7`, bốn hệ số 1.
  Loại diagonal PR theo ID, normalize có epsilon. Squared distance dùng norm/matmul,
  chỉ clamp phần âm do sai số; tránh tensor `[pairs,classes,dimension]`.
- Giữ evaluator GC/NG chung; không relation NMS riêng. Không gọi bản này là
  reproduction nguyên gốc. GloVe là prior ngôn ngữ, chưa phải GAT+KG.

## 2. GloVe và provenance

[Stanford GloVe](https://nlp.stanford.edu/projects/glove/) 6B 300D từ
[mirror chính thức đã pin revision](https://huggingface.co/stanfordnlp/glove/tree/1db2080b2d94def6e5b0386a523102f9d8849e9d).
ZIP SHA-256 `6471382cdd837544bf3ac72497a38715e845897d265b2b424b4761832009c837`;
text SHA-256 `a12599d41e3589c7160be27fffe5b0080eccd0f0c75f46666c59f90188093c40`.
File thực tế có 400.001 dòng vector hợp lệ; không ép số làm tròn 400K thành
điều kiện từ chối dữ liệu. Mapping chỉ đọc vocabulary từ train dự án đã khóa.

| Nhãn foreground | Exact | Fallback có từ GloVe | Hoàn toàn OOV |
|---|---:|---:|---:|
| 1.699 object | 1.343 | 353: từ dài nhất | 3 |
| 310 predicate | 89 | 221: mean các từ đã biết | 0 |

Ba object OOV: `dish drainer`, `dry-erase board`, `plier`. Chúng và ID 0 dùng
normal N(0,1), seed 42 kết hợp SHA-256 theo loại/chuỗi nhãn; ổn định khi thêm lớp,
khác RNG theo vị trí dòng của upstream. Không chia 0 với phrase toàn OOV.
Chi tiết từng nhãn: [penet-glove-v1.json](penet-glove-v1.json).

## 3. Thiết kế đối chứng và kết quả

Dùng nguyên [cohort](gqa-pilot-v1-cohort.json), cache, sampler và evaluator của
[pilot MLP/GAT](GQA_PILOT.md). Train có 120/310 predicate, validation 80/310;
121/128 ảnh validation có GT. mR vẫn chia toàn bộ **310** lớp, lớp vắng tính 0.
GC dùng một foreground predicate mỗi pair; NG cho nhiều predicate. Softmax gồm
background, matcher IoU inclusive 0,5 và giữ GT trùng theo protocol đã khóa.

AdamW lr 0,001, weight decay 0,01; batch 8; dropout 0,2; gradient clip 5.
Train dùng mọi positive pair và tối đa 3 background/positive, ảnh không GT lấy
tối đa 32 background. Batch không có pair bỏ optimizer step, không update riêng PR.
Các biến thể có **cùng initial-state hash và cùng sampling trace từng epoch**;
[kiểm đối chứng](penet-pilot-v1/matched-controls.json). Chọn checkpoint bằng
validation GC mR@50, giữ epoch sớm khi hòa; không dùng NG/test để chọn.

Tất cả số trong bảng là **phần trăm**; mỗi dòng dùng checkpoint đã chọn của nó.

| Biến thể | Epoch | GC R@50 | GC mR@50 | GC R@100 | GC mR@100 | NG R@50 | NG mR@50 | NG R@100 | NG mR@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PE-CE | 5 | 23.031 | 0.215 | 32.380 | 0.311 | 31.676 | 0.297 | 48.782 | 0.519 |
| PE-PL | 3 | 20.083 | 0.479 | 28.145 | 0.554 | 26.133 | 0.510 | 38.515 | 0.653 |
| PE-PLPR | 5 | 19.430 | 0.441 | 27.648 | 0.539 | 27.427 | 0.514 | 40.073 | 0.794 |

PL tăng GC mR@50 so với PE-CE nhưng giảm R@50; PLPR cũng có đánh đổi này.
PR chưa tăng GC mR@50 so với PL trong lượt chạy này, dù NG mR@100 cao hơn.
Chỉ có một seed và 5 epoch, không có ước lượng độ biến thiên hoặc bằng chứng hội tụ.
MLP/GAT lịch sử có GC R@50 30,99%/29,87%, mR@50 0,340%/0,315% trên cùng pilot.
PE có kiến trúc và GloVe prior khác; chỉ CE/PL/PLPR là đối chứng trực tiếp cho loss.
Không kết luận PE vượt baseline toàn diện hoặc PR vô ích từ pilot này.

Loss PLPR trung bình epoch 1 tổng khoảng 8,04; epoch 5 còn 0,921, gồm CE 0,806,
PL Euclidean 0,0594, PR cosine 0,0140 và PR Euclidean 0,0411. Loss và gradient
hữu hạn trong cả 15 model-epoch. Margin/scale theo code gốc không tự động tối ưu
cho dimension nhỏ và 311 lớp; chưa tune hoặc đổi hệ số sau khi xem kết quả.
History lưu mọi thành phần và gradient norm trước clip, không chỉ tổng loss.

## 4. Bộ nhớ và thời gian

RTX 5090, torch 2.7.0+cu128, float32, CLIP đã cache. Pilot train + validation mỗi
epoch + final GC/NG mất khoảng 3,68/3,69/3,80 giây cho CE/PL/PLPR, **không gồm**
tải dữ liệu, GloVe, trích CLIP hoặc initial validation. Peak allocated lần lượt
0,464/0,486/0,486 GiB. Không ngoại suy các số này thành thời gian end-to-end full GQA.

Benchmark dùng 64 graph train lớn nhất đã pin từ audit trước, không chọn theo
metric. Mỗi case bắt đầu từ checkpoint PLPR, optimizer mới; 2 warmup + 3 lần đo,
có CUDA synchronize. Update là tạm thời, không ghi lại checkpoint. Thời gian
gồm collate CPU, chuyển tensor và bước train/forward; không gồm I/O/CLIP.
Forward mọi pair chưa gồm ranking/matcher của evaluator.

| Batch graph lớn | Train peak GiB | Train giây/batch | Full-pair forward peak GiB | Forward giây/batch |
|---|---:|---:|---:|---:|
| 1 | 0.304 | 0.0143 | 0.272 | 0.0118 |
| 8 | 1.864 | 0.1177 | 0.997 | 0.1250 |
| 32 | 5.132 | 0.2828 | 2.387 | 0.2907 |
| 64 | 8.967 | 0.3810 | 3.780 | 0.3925 |

Graph lớn nhất có 126 node / 15.750 pair. Cả tám case pass, không OOM; file
checkpoint không đổi. Có thể bắt đầu kiểm launcher chính với batch train 64,
validation 32 trên **cấu hình adapter/máy này**. Chưa kiểm hội tụ ở batch lớn,
accumulation hoặc throughput full loader; không coi batch lớn là cùng recipe
tối ưu hóa với pilot batch 8. [Số đo chi tiết](penet-pilot-v1/stress.json).

## 5. Kiểm chứng và chạy lại

**296 tests pass**, gồm 2 test mapping và 14 test PE mới. Loss values và gradient
single-label 51 lớp khớp bản broadcast/sort tham chiếu float64; thêm multi-positive,
background, empty, lớp cuối, CPU/CUDA, hướng pair và batch/single equivalence.
Đây là parity của loss và test adapter, không phải parity toàn bộ detector gốc.
Full suite ban đầu gặp CuBLAS workspace do test cũ bật global deterministic mode;
fixture PE đã đặt workspace như launcher, sau đó full suite pass.

Runner reload checkpoint và tái chấm đúng metric đã chọn cho cả ba biến thể.
Demo dùng checkpoint PLPR trên validation index 0 cố định, không chọn ảnh theo
điểm: [prediction JSON](penet-pilot-v1/demo.json). Cờ khớp GT là kiểm tự động,
**không tính thành human review**; demo dự đoán relation, chưa chỉnh graph theo câu.

```bash
myenv/bin/python -m pytest -q
myenv/bin/python -m tools.demo_penet_pilot
myenv/bin/python -m tools.train_penet_pilot \
  --output-dir checkpoints/level2/penet-pilot-v2 \
  --report-dir reports/level2/penet-pilot-v2
myenv/bin/python -m tools.benchmark_penet_pilot \
  --output reports/level2/penet-stress-v2.json
```

Launcher từ chối ghi đè run/report đã tồn tại. GloVe mapping ở
`data/processed/level2/gqa-glove-v1/`, checkpoint ở
`checkpoints/level2/penet-pilot-v1/{ce,pl,plpr}/best.pt`; binary nằm ngoài Git.
[Config](penet-pilot-v1/config.json) pin hash source/cohort/cache/GloVe;
HEAD lúc chạy `106a7ea`, source launcher được khóa bằng SHA-256 trước commit.
Mapping commit `f8b6b78`, model/loss `106a7ea`, runner/demo/stress `75cdd34`.
[Biên bản kiểm chứng](penet-pilot-v1/verification.json) khóa các artifact.
Quyết định phương pháp cũ giữ nguyên như bản lịch sử; trạng thái mới ở report này
và [LEVEL2_PROGRESS.md](../../docs/LEVEL2_PROGRESS.md).

## 6. Bước tiếp theo

Chuẩn bị cache và launcher full GQA có checkpoint/resume, khóa recipe chung rồi
chạy MLP/GAT, PE-CE/PL/PLPR và GAT+KG với seeds 42/123/3407. Tách việc chuẩn bị
GQA KG khỏi GloVe. Với VG150, chạy các đối chứng cùng supervision đa nhãn mới,
không coi checkpoint Level 1 single-label là matched control.

Chưa dùng model trên project test; chưa có main study, 300 cặp MagicBrush,
human review 20 mẫu hoặc model thao tác đồ thị. Việc port/pilot PE hoàn thành
nhưng hạng mục thực nghiệm phương pháp 2023+ trong toàn bộ bài vẫn chưa hoàn tất.
