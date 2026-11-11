# Chạy full GQA — 5 model × 3 seed

**Bộ chạy đã sẵn sàng; chưa chạy thực nghiệm full.** Phạm vi đã chốt là MLP,
GAT, PE-CE, PE-PL, PE-PLPR × seeds 42/123/3407 = **15 lượt** trên toàn bộ
69.942 train / 5.000 validation. GAT+KG và vòng VG150 theo protocol mới để đợt sau.
Đây là bản PE-Net adapted đã công bố, không tái lập trực tiếp điểm paper.

## Chạy ngay

Từ thư mục repository:

```bash
scripts/run_gqa_full.sh
```

Lệnh này lần lượt kiểm đầu vào → tạo cache train/val → train 15 lượt tuần tự
trên một GPU → tổng hợp kết quả. Mặc định là `--stage all`, dùng
[gqa-full-v1.json](../configs/level2/gqa-full-v1.json). Script đổi về thư mục repo,
dùng `myenv/bin/python`, in PID và tự ghi log riêng trong `logs/gqa-full-*.log`.
Có thể chỉ định interpreter khác qua `GQA_PYTHON`, nhưng resume cần cùng môi trường.

Muốn chạy nền qua phiên SSH:

```bash
mkdir -p logs
nohup scripts/run_gqa_full.sh > logs/gqa-full-launch.log 2>&1 &
GQA_RUN_PID=$!
echo "$GQA_RUN_PID"
tail -f logs/gqa-full-launch.log
```

`Ctrl+C` khi đang `tail` chỉ dừng xem log. Để dừng tiến trình nền vừa khởi tạo
trong cùng shell, dùng `kill -TERM "$GQA_RUN_PID"`; ở shell khác dùng đúng PID
được in trong log của lượt đang chạy. Launcher dùng `exec`, nên PID này là Python.
Chạy lại chính lệnh ban đầu để tiếp tục.

## Cấu hình đã chuẩn bị

| Thành phần | Giá trị |
|---|---|
| Ngân sách | 20 epoch cho mỗi model/seed, không scheduler hoặc early stopping |
| Train / validation batch | 64 / 32 ảnh |
| Optimizer | AdamW, lr 0,001; weight decay 0,01; clip gradient norm 5 |
| Precision | Float32, không AMP hoặc accumulation; deterministic algorithms |
| Checkpoint | Mỗi 100 batch, cuối epoch, khi nhận SIGINT/SIGTERM ở điểm an toàn |
| Cache | 256 ảnh/shard, crop CLIP batch 64; float16 đã normalize |
| Đầu vào | CLIP ViT-B/32 đóng băng + geometry 5D + GT object labels/boxes |
| Taxonomy | 1.699 object và 310 predicate, thêm ID 0 |
| Supervision | Soft CE đều trên các predicate đúng, duplicate không tăng trọng số train |
| Sampling | Mọi positive pair + tối đa 3 background/positive; ảnh không GT tối đa 32 background |
| Chọn checkpoint | Validation GC mR@50 trên đủ 310 lớp; epoch sớm nếu hòa |
| Báo cáo | GC/NG R/mR@50/100, per-predicate support, mean ± sample std ba seed |

20 epoch là **recipe khởi điểm cố định**, chưa có bằng chứng hội tụ trên full GQA.
Batch 64 đã qua stress test graph lớn cho MLP/GAT và PE; điều này chứng minh khả
năng chạy trong bộ nhớ trên RTX 5090, không chứng minh tối ưu chất lượng.
Các model dùng cùng epoch budget, thứ tự ảnh và sampling theo seed; các biến thể
PE còn có cùng initial-state hash. Kiến trúc/số tham số/prior khác nhau nên không
coi MLP/GAT và PE là cùng FLOPs hoặc cùng đầu vào ngữ nghĩa.

## Tách từng giai đoạn

```bash
# Chỉ kiểm đầu vào và in kế hoạch; không train/cache full.
scripts/run_gqa_full.sh --stage preflight

# Tạo cache hoặc tiếp tục cache dở.
scripts/run_gqa_full.sh --stage cache

# Dùng cache hoàn tất; train/resume cả 15 lượt.
scripts/run_gqa_full.sh --stage train

# Chạy/resume một lượt, vẫn dùng cùng experiment/config.
scripts/run_gqa_full.sh --stage train --model pe-plpr --seed 42

# Tổng hợp khi đủ 15 result.json hợp lệ; thiếu lượt sẽ báo lỗi.
scripts/run_gqa_full.sh --stage summary

# Kiểm toàn pipeline trên 32 train / 16 val thật, 2 epoch × 15 lượt.
# Tự dùng đường dẫn -smoke riêng; có kiểm ngắt/resume PE-PLPR.
scripts/run_gqa_full.sh --stage smoke
```

`--model` có thể là `mlp`, `gat`, `pe-ce`, `pe-pl`, `pe-plpr`; `--seed` là
42, 123, 3407. Bỏ một filter để chạy toàn bộ chiều tương ứng. Chỉ dùng filters
với `train`/`all`. Sau các lượt chạy riêng, gọi `--stage summary`.

Có thêm `--stop-after-batches 1` cho `train`/`all` để kiểm resume có chủ đích.
Lượt đó thoát code 130 sau khi lưu; bỏ tùy chọn này khi chạy tiếp. Nó không đổi
epoch budget hay biến lượt chưa xong thành kết quả hoàn tất.

## Resume và tính toàn vẹn

Cache dùng shard HDF5 tạm, chỉ rename và công bố marker sau khi hoàn tất. Khi
chạy lại, shard hoàn tất được kiểm SHA-256; shard tạm bị làm lại. Nếu ngắt sau
rename nhưng trước marker, tool kiểm lại record/JPEG rồi hoàn tất marker.
Không sửa hoặc tự bỏ qua shard đã công bố mà sai hash. Identity khóa manifest,
annotation, ZIP ảnh, CLIP weights, code trích feature và danh sách ảnh.
Kiểm hash ZIP 22 GB có thể làm phần kiểm đầu vào cache mất thêm thời gian.

Loader chỉ dựng graph cho batch cần dùng, giữ tối đa 8 file HDF5 mở; không tải
toàn bộ tensor N(N−1) của full train vào RAM. Record, JPEG hash và thứ tự object
được giữ trong cache. Không thêm union crop cho PE-Net.

`last.pt` lưu model hiện tại, optimizer, CPU/CUDA RNG, sampler RNG, shuffle order,
vị trí batch, loss accumulator, history và bản model tốt nhất. Ghi qua file tạm
rồi rename, nên lỗi giữa lúc ghi giữ được checkpoint trước. Nếu kill cứng hoặc
mất máy, resume từ checkpoint gần nhất, có thể phải làm lại tối đa 100 batch;
không cộng các batch đó hai lần vào kết quả. Không cam kết bảo vệ khỏi hỏng ổ đĩa.

SIGINT/SIGTERM được xử lý tại biên an toàn. Nếu đang validation, chờ validation
xong rồi lưu trạng thái epoch. Không bấm ngắt lặp hoặc sửa checkpoint bằng tay.
Một lock chặn hai tiến trình ghi cùng experiment/cache. Recipe, source hashes,
cache/GloVe hashes, torch version và GPU được khóa; đổi chúng thì tool từ chối
resume vào cùng experiment. Giữ nguyên code và môi trường trong lúc chạy ma trận.

## Kết quả nằm ở đâu?

```text
data/cache/gqa-full-clip-v1/
  identity.json, manifest.json, train-*.h5, val-*.h5, *.h5.json
runs/level2/gqa-full-v1/
  requested-config.json, preflight.json, experiment.json, summary.json
  <model>/<seed>/
    identity.json, last.pt, history.json, best.pt, result.json
```

`history.json` cập nhật sau từng epoch; `last.pt` là nguồn resume. `best.pt` và
`result.json` được công bố sau đủ epoch và reload/tái chấm GC/NG. Lượt đã xong
được kiểm identity/hash rồi bỏ qua khi chạy lại. Các file lớn nằm ngoài Git.
`summary.json` dùng **fraction** (nhân 100 để ra %), std mẫu `ddof=1`; kiểm cùng
sampling và cùng khởi tạo PE trước khi tổng hợp. Các báo cáo mỗi lượt giữ metric
của toàn bộ 310 predicate, kể cả lớp vắng GT tính 0 trong mR.

Nếu muốn đổi epoch/batch/lr, copy JSON sang phiên bản mới và đổi `output` sang
thư mục mới trước khi chạy. Có thể giữ `cache` nếu không đổi định nghĩa feature,
source, device/torch hoặc tham số cache. Không tăng epoch ngay trong experiment
đã chạy rồi trộn với những seed đã hoàn tất recipe cũ.

## Đã kiểm chứng và phần chưa chạy

**310 tests pass**, gồm 14 ca mới: cache recovery/corruption, lock, no-pair epoch,
ghi checkpoint lỗi và resume chính xác cho 5 model trên CPU/CUDA. Smoke thật
32/16 ảnh đã chạy đủ 15 lượt; PE-PLPR chạy ngắt/resume khớp history, metric và
trọng số cuối với chạy liền. Các vector của 48 ảnh smoke khớp chính xác cache
pilot đã khóa. Có kiểm SIGTERM thật qua shell launcher và chạy tiếp thành công.
[Biên bản và hash](../reports/level2/gqa-full-readiness-v1/verification.json).

Chưa trích full cache hoặc train 15 lượt full trong bước bàn giao. Chưa có điểm
full hoặc dự toán thời gian end-to-end đáng tin cậy; log của epoch đầu sẽ cho
thông tin thực tế. Script này chỉ đọc/train/chọn checkpoint trên train/validation,
**chưa chấm project test**. Chốt lựa chọn sau ma trận validation rồi chuẩn bị
lượt test riêng, không dùng test để tune. Nó cũng chưa thay thế GAT+KG, VG150
protocol mới hoặc các phần MagicBrush/model chỉnh sửa đồ thị của Level 2.
