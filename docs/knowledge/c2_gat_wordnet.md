# C.2 và GAT+WordNet — pipeline mức 1

Trạng thái: mapping đại diện 150/150 của Toan và KG chính đã được tạo. Toan đã
xác nhận toàn bộ lựa chọn này là chính thức; manifest ghi rõ đây là xác nhận
theo lô, không có lý do riêng cho phần lớn class. C.1/C.2/GAT+KG đã nhận đường
`reviewed` này. TransE và GAT+KG vẫn chờ người dùng train GPU trên dữ liệu đầy
đủ. Smoke CPU reviewed đi qua TransE → export 150/150 → GAT+KG PASS. Cả hai
trainer đã hỗ trợ WandB; TransE chọn `best.pt` theo validation filtered MRR,
GAT+KG chọn theo validation mR@50.

## Đường chạy tiếp theo với mapping 150 class

Chạy từ thư mục gốc repository; dùng `myenv`. Các output dir phải chưa tồn tại.
Người dùng chạy lệnh tạo KG trên CPU và các lệnh train GPU. Codex không chạy
train GPU.

```bash
myenv/bin/python -m scripts.build_wordnet_graph \
  --mapping data/knowledge/wordnet-reviewed-toan-v1/wordnet_mapping.json \
  --reviewed-overrides configs/knowledge/wordnet_overrides.reviewed-toan-v1.json \
  --require-fully-reviewed --max-hops 2 \
  --output-dir data/knowledge/wordnet-kg-toan-v1
```

Lệnh CPU tạo KG; smoke riêng đạt 150/150 class, 1.077 entities, 1.312 triples,
`ready_for_gat_kg_training=true`. Đăng nhập WandB bằng `myenv/bin/wandb login`
trên máy của người dùng (không gửi API key vào chat). Tạo manifest xác nhận
chính thức từ các file mapping/override/selection đã commit:

```bash
myenv/bin/python -m scripts.prepare_level1_official_manifest
```

Trên máy hiện tại manifest đã được tạo tại
`data/knowledge/wordnet-kg-toan-v1/review_manifest.json`; teammate tái tạo
theo đường dẫn KG của họ. Manifest khóa SHA-256 của bốn file KG và nguồn lựa
chọn; không tạo thêm lý do review chưa được Toan cung cấp. Sau đó train TransE
trên GPU:

```bash
myenv/bin/python -m scripts.train_transe \
  --kg-dir data/knowledge/wordnet-kg-toan-v1 \
  --mode reviewed --review-manifest data/knowledge/wordnet-kg-toan-v1/review_manifest.json \
  --split-policy transductive \
  --output-dir runs/transe/wordnet-toan-reviewed-seed42 \
  --dim 64 --epochs 200 --patience 20 --min-delta 0.0001 \
  --batch-size 512 --seed 42 --device cuda --wandb-mode online
```

10 epochs ở bản C.1 cũ chỉ dùng smoke; validation MRR của bản WordNet cũ còn
tăng tại epoch 10. Bản này dùng tối đa 200 epochs, dừng sau 20 epochs liên tiếp
không tăng validation filtered MRR ít nhất 0.0001; export `best.pt`, không dùng
epoch cuối một cách mặc định. Chỉ dùng validation để chọn checkpoint; test
được tính một lần sau train trên checkpoint đã chọn. WandB log train loss,
validation MRR/Hits và epoch tốt nhất. Dry-run split transductive PASS:
1.050 train, 131 validation, 131 test triples. Sau khi run hoàn tất, xuất
embedding trên CPU:

```bash
myenv/bin/python -m scripts.export_knowledge \
  --checkpoint runs/transe/wordnet-toan-reviewed-seed42/best.pt \
  --kg-dir data/knowledge/wordnet-kg-toan-v1 \
  --review-manifest data/knowledge/wordnet-kg-toan-v1/review_manifest.json \
  --mapping data/knowledge/wordnet-reviewed-toan-v1/wordnet_mapping.json \
  --output-dir data/knowledge/c2-wordnet-reviewed-v1
```

Kiểm tra output báo `150/150 mapped classes` trước khi train GAT+KG, rồi chạy
seed 42:

```bash
myenv/bin/python -m scripts.train_gat_kg \
  --kg-features data/knowledge/c2-wordnet-reviewed-v1 \
  --device cuda --seed 42 \
  --epochs 200 --patience 15 --min-delta 0 \
  --batch-size 512 --accumulation-steps 1 --val-batch-size 128 \
  --max-train-images 57723 --max-val-images 5000 \
  --hidden-dim 256 --embedding-dim 32 --num-heads 4 --num-layers 2 \
  --dropout 0.1 --lr 0.001 --weight-decay 0.01 \
  --negative-ratio 3 --box-policy clip --wandb-mode online \
  --output-dir runs/level1/wordnet-reviewed-seed42-v1
```

200 là trần, không phải số epoch bắt buộc: dừng sau 15 epochs liên tiếp không
tăng validation mR@50, giữ `best.pt`. Dùng cùng patience 15 với GAT baseline;
baseline seed 123 đạt best epoch 100 nên không đặt trần 100 cho KG run.
WandB project là `semantic-graph-editing`; runner log train/validation mỗi epoch
và lưu run ID trong `last.pt` để resume. Checkpoint lưu local, không tự upload
lên WandB. Khi resume, thêm
`--resume runs/level1/wordnet-reviewed-seed42-v1/last.pt` vào đúng lệnh gốc.

Kết quả dùng mapping chính thức 150 synset đại diện. Báo cáo phải nêu rõ xác
nhận theo lô của Toan, class đa nghĩa có thể chứa object instance không khớp
synset đại diện, và không có lý do review riêng cho phần lớn class. Đó là giới
hạn của bằng chứng, không phải trạng thái `draft` của pipeline.

## Bản draft 124 class trước đây

## C.2: mục đích và cách xuất

```bash
python -m scripts.export_knowledge \
  --checkpoint runs/transe/c1-wordnet-transductive-seed42/epoch_0010.pt \
  --kg-dir data/knowledge/wordnet-ai-draft-v1/kg \
  --mapping data/knowledge/wordnet-ai-draft-v1/mapping/wordnet_mapping.json \
  --output-dir data/knowledge/c2-wordnet-draft-v1
```

Lệnh đã chạy trên CPU ở máy project. Không chạy lại cùng output-dir vì exporter
từ chối ghi đè. Trên máy teammate cần có đúng checkpoint và artifact mapping/KG
đầu vào, hoặc dùng artifact export chung và đối chiếu hash.

Output: `class_embeddings.pt` và `manifest.json`.

- `[151,64]`: hàng 0 background, hàng 1..150 đúng class ID từ vocabulary.
- 124 hàng class được ánh xạ; 26 chưa ánh xạ dùng vector 0, mask=false.
- Không tự coi ID entity 0 là background: KG entity 0 là một entity bình thường.
- Kiểm tra SHA-256 giữa checkpoint, KG, mapping và vocabulary; đối chiếu thêm
  `class_entity_ids.json` với synset thực tế. Sai thứ tự hoặc nguồn sẽ dừng.
- Lưu review status và curation tier; `linked_manual` trong schema cũ không
  tự được hiểu là human-approved. Toàn bộ export vẫn là draft.
- Giữ nguyên vector TransE, không tự chuẩn hóa lại hay căn chỉnh với ConceptNet.
- 124/150 là coverage ánh xạ, không phải độ chính xác ngữ nghĩa.

## GAT+WordNet và ablation

Model mới kế thừa kiến trúc GAT baseline, ghép vào input projection:
geometry 5D + embedding nhãn học được 32D + WordNet 64D + valid bit 1D.
KG là buffer đóng băng, được lưu cùng checkpoint để máy khác đánh giá được
mà không cần đường dẫn export gốc.

Các file baseline MLP/GAT không bị sửa. Runner mới tái sử dụng hàm train,
negative sampling và evaluator của GAT gốc.

| Điều kiện | Switch | Mục đích |
|---|---|---|
| GAT+WordNet | mặc định | Đo ảnh hưởng của đặc trưng tri thức |
| Cùng kiến trúc, tắt KG | `--no-knowledge` | Zero vector KG và valid bit; giữ nguyên số tham số để đối chứng |
| GAT+WordNet, bỏ cạnh giữa đối tượng | `--message-edges none` | Không truyền giữa các đỉnh; vẫn giữ self loops, candidate pairs và GT |

Baseline GAT gốc và đối chứng `--no-knowledge` không có cùng input dimension:
baseline dùng 5+32, đối chứng có thêm 65 kênh bằng 0. Phải báo cáo riêng, không
gộp chúng thành cùng một mô hình. Dùng đối chứng cùng kiến trúc để phân tích
đóng góp KG; dùng baseline gốc để báo cáo tiến trình MLP → GAT → GAT+KG.

Graph baseline là fully connected nên hoán vị đỉnh không thay đổi tập cạnh;
chọn ablation **bỏ cạnh** để có đối chứng thực sự thay đổi message passing.

## Lệnh train GPU — chỉ người dùng chạy

Kích hoạt `myenv`. Trên máy project, Codex đã cài torch 2.7.0+cu128 và xác
nhận PyTorch nhận RTX 5090; máy teammate cài theo `../TEAM_SETUP_LEVEL1.md`.
Codex chưa chạy lệnh train GPU dưới đây.

Chạy một seed trước để kiểm tra tài nguyên và đầu ra của thực nghiệm dài:

```bash
python -m scripts.train_gat_kg \
  --kg-features data/knowledge/c2-wordnet-draft-v1 --allow-draft \
  --device cuda --seed 42 \
  --epochs 200 --patience 15 \
  --batch-size 512 --accumulation-steps 1 --val-batch-size 128 \
  --max-train-images 57723 --max-val-images 5000 \
  --hidden-dim 256 --embedding-dim 32 --num-heads 4 --num-layers 2 \
  --dropout 0.1 --lr 0.001 --weight-decay 0.01 \
  --negative-ratio 3 --box-policy clip --wandb-mode disabled \
  --output-dir runs/level1/draft-wordnet-seed42-v1
```

Giải thích:

- `--allow-draft`: chấp nhận chạy chẩn đoán với mapping chưa kiểm duyệt;
  không biến kết quả thành kết quả KG đã được xác minh.
- `--epochs 200 --patience 15`: tối đa 200 epochs; dừng sau 15 epochs không
  tăng validation mR@50. Đây là ngân sách mới được ghi rõ, không phải bằng chứng
  ngân sách lịch sử của baseline giống hệt. Kiểm tra đối chứng trước kết luận cuối.
- Batch 512, accumulation 1 và validation batch 128 theo config GAT baseline
  đã kiểm kê. Nếu OOM, dừng và ghi nhận để thiết kế lại ngân sách; không tự
  coi micro-batch khác + accumulation là thực nghiệm hoàn toàn tương đương.
- `--max-*-images`: sử dụng đầy đủ train/validation của split project.
- Model chọn `best.pt` bằng validation mR@50; tuyệt đối không đọc test khi train.
- `--output-dir`: thư mục run riêng, bắt buộc chưa tồn tại. Local logs thay W&B.

Output gồm config với hash dữ liệu/code/export, môi trường và pip freeze,
history, bản manifest KG, `best.pt` và `last.pt`.

Nếu bị ngắt giữa chừng, chạy lại đúng lệnh với
`--resume runs/level1/draft-wordnet-seed42-v1/last.pt`.
Resume yêu cầu cùng config, budget, hash dữ liệu và code; không dùng `best.pt`.
Checkpoint ở cuối epoch, nên phần epoch chưa hoàn thành sẽ chạy lại.

Sau khi kiểm tra seed 42, chạy seeds 123 và 3407 bằng cách đổi cả `--seed`
và `--output-dir`. Với ablation, dùng cùng các tham số và seed, thêm switch
trong bảng trên và đặt thư mục riêng. Chưa khởi chạy hàng loạt trước khi nhóm
quyết định thời điểm kiểm tra tay mapping và chốt ngân sách đối chứng.

## Đánh giá validation sau train

```bash
python -m scripts.evaluate_level1 \
  --checkpoint runs/level1/draft-wordnet-seed42-v1/best.pt \
  --device cpu --batch-size 32 \
  --output reports/level1/draft-wordnet-seed42-val.json
```

Mặc định đánh giá toàn bộ validation, không phải 100 ảnh. Có thể dùng CUDA
cho inference nếu người dùng muốn; lệnh trên dùng CPU. `--max-images` chỉ dùng
cho smoke và report ghi rõ không phải full split. Output không ghi đè file cũ.

Cùng evaluator này đọc được checkpoint `mlp`, `mlp_semantic`, `gat`, `gat_kg`.
Báo cáo có R@50/100, mR@50/100, per-predicate recall, hash dữ liệu/checkpoint,
config và trạng thái draft. Test bị khóa nếu không có `--allow-test` và chỉ
được dùng sau khi chốt protocol.

## Việc còn lại trước kết quả mức 1

1. Nhóm kiểm tra tay mapping và quyết định cách xử lý các nhãn chưa ánh xạ;
   bổ sung reviewed export sau khi có bằng chứng thực tế.
2. Hoàn thành 3 seeds và ablation với ngân sách đã ghi nhận, xem xét rerun GAT
   đối chứng nếu chưa xác minh được ngân sách lịch sử.
3. Đánh giá full validation độc lập, tổng hợp mean/sample std; chốt protocol
   rồi mới đánh giá held-out test cuối cùng.
4. Chọn và phân tích 5 đúng/5 sai từ model đã train đủ, không dùng kết quả
   smoke làm ví dụ chứng minh mô hình tốt.
5. Notebook minh họa, ảnh nguồn được tải theo ID, báo cáo, slide và demo.
