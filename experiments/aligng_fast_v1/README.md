# AlignG fast v1 — custom GQA, Frozen-CLIP PredCls

Phiên bản thực thi riêng của `aligng_adapted_v1`, trainer/evaluator sao chép từ
`aligng_gqa_v1`, queue từ `aligng_gqa_bs512_v1`. Không sửa `src/`, manifest,
config, source hay checkpoint của các thực nghiệm cũ. Vẫn GQA 310 predicate
foreground, 1.699 object foreground; chưa triển khai VG150 trong phiên bản này.

## Ba thay đổi

1. Encoder: chiếu node/word một lần cho batch; encode toàn bộ context theo các
   khối 16.384 cặp, thay cho vòng lặp encoder trên từng ảnh.
2. CPU: kiểm tra graph, tính query-to-context lookup và lưu supervision sparse
   một lần khi nạp RAM. Forward AlignG không còn `.item()`, `.any()`, `.all()`,
   sort/searchsorted để kiểm tra từng ảnh trên GPU. Vẫn kiểm tra loss/gradient
   hữu hạn ở trainer. Graph đã chuẩn bị phải được xem là bất biến.
3. Attention: bucket theo số context cặp, tối đa 32 ảnh/nhóm, giới hạn dự kiến
   16 triệu phần tử attention mỗi nhóm (một ảnh lớn có thể vượt ngưỡng này).
   Mask loại padding khỏi softmax, mỗi ảnh chỉ nhìn context của chính nó.
   Mọi prototype update chạy trước reverse attention, không phụ thuộc nhãn
   hay số query được lấy mẫu. Checkpoint encoder ở khối lớn; bỏ checkpoint
   các attention query khối 64. Không chia/truncate trục softmax hay lấy mẫu context.

Giữ nguyên parameters, công thức PE/GRU, global classifier, PL/PR, sampling,
metric GC/NG và early stopping. Float32; không AMP/TF32. PE-PLPR không feedback
gọi đúng forward baseline cũ. CPU tests đối chiếu output/embedding/gradient,
mask, ảnh rỗng, thứ tự cặp, ảnh độc lập, RNG sampler, checkpoint và resume.

Vector hóa thay đổi thứ tự rút dropout và sai số cộng số thực: đây là **thực
nghiệm mới**, không tự động tiếp tục checkpoint của phiên bản cũ. Resume trong
fast v1 giữ nguyên RNG, optimizer, order/index và kiểm tra identity. Không suy
diễn chất lượng cuối cùng từ kiểm tra tương đương khi tắt dropout.

## Kiểm tra và chạy

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 myenv/bin/python -m pytest -q tests/aligng_fast_v1 tests/aligng_adapted_v1 tests/aligng_gqa_v1 tests/aligng_gqa_bs512_v1
```

Sau khi code đã commit và GPU rảnh, benchmark giới hạn trên 3 batch thực 512 ảnh
đã audit: max-context train, max-sampled train, max-context validation. Chạy bản
cũ rồi bản mới tuần tự; mỗi case warmup 1 + đo 2 step. Train step dùng optimizer
tạm, không lưu checkpoint train, không W&B. Val so sánh logits với bản cũ.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 myenv/bin/python -u -m experiments.aligng_fast_v1.probe
```

Report: `runs/level2/aligng-gqa-fast-v1/capacity.json`, khóa theo experiment hash.
Thời gian đo gồm collate CPU, transfer và compute; không tính chuẩn bị graph một
lần khi nạp RAM. Val benchmark không bao gồm ranking/metric CPU; không phải thời
gian cả epoch. Source/config thay đổi cần version mới và đo lại.

Lệnh train chính, do người dùng chạy sau khi benchmark PASS:

```bash
bash scripts/run_aligng_gqa_fast.sh
```

Queue tự chạy 6 lượt tuần tự: AlignG42 → PE-PLPR42 → AlignG123 → PE-PLPR123 →
AlignG3407 → PE-PLPR3407. Batch train/val/effective đều 512, tối đa 200 epoch,
patience 15. Bắt buộc W&B online; lỗi tracker không fallback offline. Một model
tại một thời điểm, dùng chung khóa GPU với launcher cũ. Preflight sẽ từ chối nếu
lượt cũ đang chạy; không tự ngắt hoặc khởi động lại lượt cũ.

Output `runs/level2/aligng-gqa-fast-v1/full/{model}/{seed}/`; tmux `aligng-gqa-fast`;
log `logs/aligng-gqa-fast.log`. Lặp lại launcher resume lượt fast còn dở, xác minh
và bỏ qua lượt đã hoàn tất; lỗi/signal dừng toàn queue.

```bash
tail --follow=name --retry --sleep-interval=2 logs/aligng-gqa-fast.log
tmux attach -t aligng-gqa-fast
```

Ctrl-C trong tmux yêu cầu dừng sau batch và lưu `last.pt`; không dùng SIGKILL.
