# Test evaluation v4, extension v1

Người dùng cho phép assistant dùng GPU để eval official test sau khi đã hoàn
thành sáu lượt train chính. Mã eval tách ở
`tools/eval_emnlp2020_task_b_wandb_v4_test_v1.py`; không sửa source/config/manifest
đã khóa của v4 hoặc các phiên bản cũ. Code và CPU checks phải commit trước GPU.

Đánh giá **MLP và GNN, mỗi model seeds 42/43/44**, lấy `best.pt` đã khóa theo
validation loss; batch eval **128**, max decode 192 field tokens. Cùng 530 mẫu
test sạch của v3, official test gốc 1.000 mẫu; dùng đúng split/filter/feature
cache/vocabulary đã khóa. Test này đã từng được đánh giá ở v3. V4 không được
mô tả là test mới chưa xem; arguments v4 được chọn từ history validation.

Trước inference, audit toàn source/config, input hashes, freeze đủ sáu lượt,
completion receipts, checkpoint tensors/metadata, early stopping và W&B
selected summaries. Kiểm test không trùng source groups train/val đã reserve,
GT reconstruction và cache source-only. Không fit vocabulary hoặc tune lại
model bằng test. Không có optimizer/training hoặc nhánh khởi chạy lượt train.

Eval thực hiện **GPU forward mới** trên từng checkpoint v4; số đo không chép từ
v3 dù tham số đã được xác nhận giống nhau. Hai baseline no-op và nearest training
program được tính lại trên CPU bằng đúng protocol cũ. Chạy lần lượt một model
mỗi thời điểm, xong đủ sáu lượt thì dừng, không tự chạy experiment khác.

```bash
CUDA_VISIBLE_DEVICES=0 myenv/bin/python -m tools.eval_emnlp2020_task_b_wandb_v4_test_v1 --stage eval
```

Output riêng: `runs/level2/emnlp2020-task-b-wandb-v4-test-v1/`. Mỗi model/seed có
predictions/cases/result và completion receipt; root có evaluation identity,
summary và baselines. Không thêm/sửa file trong run train v4. Checkpoint/result
đã khóa không bị thay bởi eval; mode `test_used=false` trong training freeze là
trạng thái **tại lúc khóa train**, còn test evidence nằm trong output eval này.
Không init W&B hoặc sửa summary W&B của train; log online train giữ nguyên.

```bash
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m tools.eval_emnlp2020_task_b_wandb_v4_test_v1 --stage summarize
```

Lệnh summarize chỉ CPU, không inference, ghi thống kê 3 seed/P/R/F1 sáu loại,
exact/valid rates và case IDs đã lược bỏ raw vào
`reports/level2/emnlp2020-task-b-wandb-v4-test-v1/`. Full instruction/graphs/
operations/checkpoints chỉ giữ cục bộ. Đối chiếu v3 được ghi sau khi thực sự
chạy eval, không dùng để lựa chọn hoặc sửa prediction.

Completed eval được kiểm hash và reuse khi chạy lại; code/input/device đổi phải
đổi output version. Test dữ liệu nhỏ trên CPU dùng weights synthetic để kiểm
luồng sáu checkpoint, freeze barrier, no optimizer/no W&B, reuse và tamper;
không tính là kết quả dữ liệu thật:

```bash
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m pytest -q tests/editing/test_emnlp2020_task_b_wandb_v4_test_eval_v1.py
```
