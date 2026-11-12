# GQA + WordNet: mapping và pilot

Mốc 2026-11-12: đã triển khai phần mở rộng, tạo mapping/embedding riêng GQA,
chạy pilot ba đối chứng và kiểm exact resume. Chưa chạy GAT+KG full ba seed;
chưa có human review mới của mapping GQA.

- [Báo cáo mapping, review CSV và KG](../reports/level2/gqa-wordnet-v1/README.md).
- [Kết quả pilot và giới hạn diễn giải](../reports/level2/gqa-wordnet-pilot-v1/README.md).

## Tái tạo artifact

Chạy từ thư mục gốc repo bằng `myenv/bin/python`; cần WordNet 3.0 có sẵn trong
NLTK, artifact review VG150 của Toàn và cache GQA full đã xác minh. Không tải
corpus hoặc đọc annotation test ngầm. Lệnh chuẩn bị có thể chạy lại để kiểm
artifact trùng nội dung; checkpoint/feature output cần thư mục mới khi train lại.

```bash
myenv/bin/python -m tools.prepare_gqa_wordnet

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 myenv/bin/python -u -m scripts.train_transe \
  --kg-dir data/knowledge/gqa-wordnet-v1/kg --mode draft \
  --output-dir runs/level2/gqa-wordnet-transe-draft-v1 \
  --split-policy transductive --dim 64 --epochs 100 --patience 15 \
  --batch-size 512 --lr 0.001 --seed 42 --device cuda

myenv/bin/python -m tools.export_gqa_wordnet
```

`train_transe` dùng mode draft vì quyết định review nguồn VG150 không phải
review trong ngữ cảnh GQA. Export xác minh checkpoint/KG/mapping/vocabulary,
tái tạo split triple và che mọi entity chưa xuất hiện ở train.

## Chạy pilot tuần tự

```bash
tmux new-session -s gqa-wordnet-pilot-v1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 myenv/bin/python -u \
  -m tools.run_gqa_kg_pilot > logs/gqa-wordnet-pilot-v1.log 2>&1
```

Nếu run đã hoàn tất, lệnh xác minh rồi bỏ qua từng kết quả. Nếu bị ngắt, nó
resume `last.pt` với cùng identity. Dùng `--output` và `--report` mới khi đổi
nguồn/protocol. Script hiện chỉ hỗ trợ pilot cố định; không có chế độ tự khởi
động main training. W&B tắt cho pilot; recipe full cũ vẫn giữ thiết lập online.

Pilot dùng AdamW lr 0,001, weight decay 0,01, clip 5; kiến trúc GAT 4 head,
2 lớp, dropout 0,1; CE với soft target trên các predicate GT phân biệt.
Topology là toàn bộ cặp có hướng không self-edge, GAT thêm self-loop nội bộ.
Tri thức chỉ bổ sung vector node, không đưa predicate GT vào message edges.

```bash
myenv/bin/python -m tools.audit_gqa_wordnet
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 myenv/bin/python -m pytest -q \
  tests/knowledge tests/models/test_gat.py tests/models/test_gat_kg.py \
  tests/training/test_gqa_full.py tests/training/test_gqa_prepared_loader.py \
  tests/training/test_gqa_kg_extension.py \
  tests/evaluation/test_predcls_protocol.py tests/evaluation/test_reference_predcls.py
```

Audit kiểm file nguồn đã khóa, hash của 15 result/best cũ, buffer KG theo ID
và toàn bộ state của bằng chứng resume. Kiểm notebook dựa trên SHA256 của
bản bàn giao; nếu người dùng tiếp tục chỉnh notebook, audit báo khác hash để
kiểm tra chứ không ghi đè.

## Trước thực nghiệm chính

1. Kiểm tay mapping bằng ảnh train; điền `review_sample.csv`, lưu reviewer,
   quyết định và ghi chú. Mẫu 30 nhãn gồm nhãn đã liên kết và nhãn chưa chọn;
   đây không phải mẫu 20 cặp MagicBrush được yêu cầu riêng trong đề.
2. Đo độ phủ object trên full train, ưu tiên các nhãn đa nghĩa có tần suất cao.
   Giữ provenance của mọi quyết định bổ sung và báo rõ những lớp nhận vector 0.
3. Khóa phiên bản mapping/KG/feature và launcher full mới. Giữ cùng split,
   evaluator, batch 512, AdamW và early stopping 200 epoch / patience 15 của GAT.
4. Chạy GAT+KG các seed 42/123/3407 tuần tự trong tmux, output riêng. Dùng KG-off
   làm đối chứng kiến trúc nếu cần quy kết cải thiện cho tri thức.
5. So sánh validation R@50/100, mR@50/100, long-tail với GAT đã có; chốt phương
   pháp/protocol rồi mới chấm held-out test GQA.

PDF trang 2 yêu cầu liên kết khái niệm có kiểm tra tay, trang 4 yêu cầu tri thức
ngoài dữ liệu và cùng split, trang 5 yêu cầu MLP/GNN/GNN+KG cùng ít nhất ba seed.
Pilot này mới hoàn thành phần chuẩn bị và kiểm kỹ thuật cho đối chứng GQA+KG.
