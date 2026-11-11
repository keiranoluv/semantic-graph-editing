# Full GQA v2 — W&B và early stopping

**Cấu hình hiện tại: tối đa 200 epoch, patience 15, min_delta 0**, theo lựa chọn
của người dùng. Giữ ma trận 5 model × seeds 42/123/3407 và protocol GQA đã khóa.
Đây là trần epoch chung; số epoch/optimizer step thực tế có thể khác giữa các lượt.
Chưa chạy full cache hoặc thực nghiệm full.

Project online: [keiranoluv-hcmus/semantic-graph-editing](https://forge.coreweave.com/wandb/keiranoluv-hcmus/semantic-graph-editing/workspace).
W&B SDK trong môi trường là **0.18.7**; đã kiểm online bằng credentials sẵn có
trên máy. Không đưa API key vào source/config/report.

## Chạy

```bash
# Kiểm config/assets, không train.
scripts/run_gqa_full.sh --stage preflight

# Tự cache, train/resume 15 lượt và tổng hợp; mặc định dùng config v2.
scripts/run_gqa_full.sh

# Hoặc một lượt sau khi cache hoàn tất.
scripts/run_gqa_full.sh --stage train --model pe-plpr --seed 42

# Smoke 32 train/16 val, 2 epoch × 15 lượt; tự đổi W&B thành offline.
scripts/run_gqa_full.sh --stage smoke
```

Config: [gqa-full-v2.json](../configs/level2/gqa-full-v2.json).
Kết quả mới ở `runs/level2/gqa-full-v2/`; cache vẫn dùng
`data/cache/gqa-full-clip-v1/` vì không đổi feature/source cache.
Chạy nền, log, lock và signal dừng giữ cách dùng của
[hướng dẫn v1](LEVEL2_GQA_FULL_RUN.md), nhưng v2 thay epoch budget, checkpoint
export và tracking như mô tả ở đây. Nếu chuyển sang máy chưa đăng nhập:

```bash
myenv/bin/wandb login
```

Chỉ chạy đăng nhập tại terminal của bạn; không dán API key vào chat hoặc JSON.
Nếu chủ động muốn offline, copy config, đổi `wandb.mode` thành `offline` và đổi
`output` sang thư mục riêng. `disabled` tắt tracking hoàn toàn. Mode/entity/project
được khóa trong experiment, không đổi giữa lượt đang chạy.

## Vì sao bỏ giới hạn cố định 20 epoch?

20 epoch v1 là ngân sách khởi điểm chung để chuẩn bị launcher, **không được suy ra
từ đường hội tụ full GQA**. Với batch 64, full train có khoảng 1.093 batch/epoch,
nên không thể đồng nhất 20 epoch full với 20 epoch pilot. Chưa có bằng chứng để
kết luận 20 chắc chắn đủ hoặc quá ít. V2 dùng trần 200 và early stopping để các
lượt có thêm thời gian cải thiện; vẫn phải xem learning curve sau khi chạy.
Không thêm scheduler hoặc thay lr/weight decay trong lượt tích hợp này.

## Early stopping và tiêu chí best

- Theo dõi **validation GC mR@50**, luôn lấy mean trên đủ 310 predicate.
  Lớp vắng GT có recall 0; không chọn checkpoint theo test hoặc NG.
- `best.pt` luôn giữ **giá trị GC mR@50 cao nhất thực tế** đã thấy; bằng điểm giữ
  epoch sớm hơn. Đây là tiêu chí chọn model, không phải train loss hay R@50.
- Counter patience tăng sau mỗi validation không cải thiện quá `min_delta` so
  với mốc stopping tốt nhất; cải thiện thì reset về 0. Với `min_delta=0`, hòa
  cũng tính là không cải thiện. Dừng khi đủ **15 validation liên tiếp** như vậy,
  hoặc khi chạm 200 epoch.
- Nếu sau này đặt min_delta > 0, một tăng nhỏ vẫn được lưu best nếu đạt max mới,
  nhưng chưa reset patience. Không dùng min_delta để bỏ mất checkpoint tốt nhất.
- Lưu `bad_epochs`, mốc stopping và cờ đã early-stop trong `last.pt`; resume
  tiếp tục counter cũ. Lượt đã dừng không bị train thêm chỉ vì chạy lại lệnh.

mR cân bằng trọng số giữa predicate, phù hợp kiểm tra long-tail hơn việc chỉ chọn
theo R bị các quan hệ phổ biến chi phối. Pilot chưa chứng minh phương pháp nào
vượt trội trên full GQA; tiêu chí này được giữ từ protocol đã chốt.

## Best và last checkpoint

| File | Thời điểm lưu | Dùng để |
|---|---|---|
| `last.pt` | Mỗi 100 batch, cuối epoch, khi ngắt có xử lý | Resume model + optimizer + RNG + shuffle/offset + history + early stopping |
| `best.pt` | Ngay sau validation có best mới | Reload/evaluate/inference model tốt nhất; không dùng làm nguồn resume training |

`last.pt` còn chứa bản trọng số best. Mỗi epoch commit `last.pt` trước, rồi xuất
`best.pt` qua file tạm + rename. Nếu bị ngắt giữa hai thao tác, resume dựng lại
best từ last. Trước validation đầu tiên chưa có best hợp lệ. `result.json` chỉ
được công bố khi lượt kết thúc và reload best chấm lại khớp metric lựa chọn.
W&B bị lỗi logging sau validation không làm mất checkpoint epoch đã commit.

## W&B log gì và resume thế nào?

Mỗi model/seed có một run online. Group dùng fingerprint experiment; tên và tags
chỉ rõ `full` hay `smoke`, model và seed. Các metric dùng **fraction**, nhân 100
để đọc thành phần trăm.

- `train/*`: loss CE/PL/PR/total trung bình epoch, số pair/step, gradient norm.
- `val/*`: loss mọi pair, GC R/mR@50/100; trục X là `epoch`.
- `best/*`, `early_stopping/bad_epochs`: best score/epoch và counter patience.
- Summary khi xong: GC/NG metric của selected checkpoint, selected epoch,
  epochs completed, stopping reason, hash và đường dẫn checkpoint local.
- Config lưu hyperparameters, nguồn/hash, taxonomy và protocol. SDK giữ các
  system metrics mặc định. Tích hợp hiện log metric/metadata; checkpoint binary
  nằm local, không tự upload artifact model hoặc ảnh/dataset.

Chỉ log epoch **sau khi checkpoint đã lưu bền vững**. Online dùng ID ổn định và
`resume="allow"`; đọc step của W&B để replay những epoch đã commit nhưng chưa
log, không phát lại epoch server đã có. Sidecar `wandb-identity.json` và
`wandb-latest.json` trong từng `<model>/<seed>/` giữ ID, URL, trạng thái và thư mục
SDK. Lỗi W&B được báo ra; chạy lại cùng lệnh khôi phục từ checkpoint và replay log.
Code tracking bảo toàn CPU/CUDA/Python/NumPy RNG để không đổi quá trình học.
[Resume theo tài liệu chính thức](https://docs.coreweave.com/products/wandb/runs/resuming),
[trục epoch tùy chỉnh](https://docs.coreweave.com/products/wandb/track/log/customize-logging-axes).

**Offline với SDK 0.18.7:** code SDK đang cài bỏ qua resume offline. Do đó mỗi
invocation offline tạo một snapshot có physical ID riêng, replay toàn bộ history
đã commit và giữ chung logical ID. Sau khi lượt hoàn tất, chỉ sync **snapshot
mới nhất** được trỏ bởi `wandb-latest.json` (thư mục cha của `files_dir`):

```bash
myenv/bin/wandb sync <thu-muc-offline-run-moi-nhat>
```

Không gọi các snapshot offline khác nhau là một run đã resume trên cloud. Lệnh
smoke mặc định offline để kiểm integration mà không tạo 15 run thử trên project.
Một lượt online riêng đã được tạo có nhãn `integration-smoke` để kiểm API thực.

## Tổng hợp và khả năng tái lập

`summary.json` vẫn báo mean±std mẫu ba seed. Thêm `training_lengths` gồm số epoch
đã chạy, lý do dừng và selected epoch từng lượt. Sampling chỉ được đối chiếu trên
các epoch chung: một run dừng ở epoch 30 phải khớp 30 epoch đầu của run dài hơn.
Không tuyên bố equal compute khi dùng early stopping. Khởi tạo ba biến thể PE
vẫn phải khớp theo seed trước khi tổng hợp.

Recipe v1 và báo cáo readiness v1 được giữ như lịch sử ở commit `0c1cf20`.
Source/recipe đã đổi nên **không resume run v1 bằng code v2**; nếu cần tái lập v1
phải dùng checkout riêng của commit đó. Full v1 chưa bắt đầu trên máy khi thực
hiện thay đổi này. Không sửa config/epoch/entity trong một run đã khởi tạo.

Kiểm chứng: **318 tests pass**; test patience/ties/min_delta, early-stop resume,
best trước khi kết thúc, khôi phục best bị thiếu, logging lỗi sau checkpoint và
summary có run dài ngắn khác nhau. Smoke thật đủ 15 lượt offline; kiểm online
ngắt sau epoch 1 rồi resume cùng ID, server có đúng epochs `[1, 2]`, metric khớp
checkpoint local. [Biên bản v2](../reports/level2/gqa-full-readiness-v2/verification.json).
