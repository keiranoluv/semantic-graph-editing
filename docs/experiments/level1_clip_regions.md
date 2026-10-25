# Mức 1 — Ma trận PredCls có vùng ảnh CLIP

Chạy mọi lệnh từ gốc repository bằng `myenv`. Đây là nhánh thực nghiệm mới;
không ghi đè checkpoint/report annotation-only trong `runs/level1/` và
`reports/level1/wordnet-reviewed-*`. Người dùng chạy các lệnh PredCls GPU.
Trên máy hiện tại, cache CLIP và TransE/C.2 v2 đã hoàn tất; các lệnh tạo chúng
ở dưới phục vụ tái lập trên máy khác. Ma trận 15 PredCls run chưa chạy.

## Thiết kế và đối chứng

- Dùng đúng `splits.json`: 57.723 train, 5.000 validation, 26.446 test.
  Chỉ train/validation được dùng để chọn checkpoint; chưa đánh giá test.
- CLIP ViT-B/32 [bản OpenAI](https://github.com/openai/CLIP) được đóng băng.
  Cắt từng GT box trên JPEG, dùng preprocessing do `clip.load` trả về,
  chuẩn hóa L2 và cache vector 512 chiều theo HDF5 box index. JPEG nào khác
  kích thước metadata sẽ scale tọa độ box riêng theo trục x/y trước khi crop;
  hình học 5 chiều vẫn tính từ GT annotation gốc.
- MLP semantic, GAT và GAT+WordNet nhận **cùng** hình học 5D, cùng vector
  CLIP 512D và cùng class ID; mỗi model học bảng embedding nhãn 32D riêng.
  GAT+WordNet ghép thêm
  TransE 64D cùng valid bit; KG-off giữ kiến trúc đó nhưng zero KG. Edges-off
  giữ input/KG nhưng chỉ còn self-loop trong message passing. Tất cả dùng
  cùng negative ratio 3, batch 512, trần 200 epoch, patience 15 và chọn
  validation mR@50. Kiến trúc khác nhau nên đây là so sánh cùng dữ liệu và
  ngân sách epoch, không phải bằng nhau về FLOPs.
- Ba seed PredCls: 42, 123, 3407. TransE seed 42 tạo một embedding cố định
  dùng chung cho ba seed downstream. Giá trị validation của ma trận mới phải
  được đánh giá độc lập trên đủ 5.000 ảnh rồi tính mean/sample std.

### Đối chiếu đề bài và giới hạn diễn giải

Ma trận có ba mô hình chính MLP phi đồ thị, GAT, GAT+WordNet và hai ablation
KG-off, edges-off; mỗi điều kiện có ba seed. Mọi mô hình dùng cùng GT boxes,
object labels, vector CLIP, split và tập cặp ứng viên. Metrics R@50/100 và
mR@50/100 được tính trên đủ validation. Điều này đáp ứng phần so sánh và ablation
của mức 1 trong PDF; phân tích 5 đúng/5 sai và đánh giá test cuối vẫn là bước
sau các run.

- Embedding nhãn 32D được học từ train thay cho GloVe/CLIP text mà PDF gợi ý;
  PDF cho phép chọn input khác nếu giải thích rõ trong báo cáo.
- GAT truyền tin trên đồ thị ứng viên có cạnh có hướng giữa mọi cặp object,
  không đưa predicate GT vào cạnh. Edges-off chỉ bỏ truyền tin giữa các object;
  kết quả đo ích lợi của ngữ cảnh đồ thị ứng viên, không đo riêng khả năng suy
  luận trên topology cạnh quan hệ GT.
- Một checkpoint TransE seed 42 dùng chung ba seed downstream. Độ lệch chuẩn
  ba seed vì thế chỉ phản ánh phần PredCls, không phản ánh biến thiên khi train
  lại KG embedding.
- Chọn `best.pt` theo validation mR@50 rồi báo cáo trên chính validation đó;
  đây là số validation để chọn mô hình. Chỉ đánh giá test một lần sau khi khóa
  thiết kế. Protocol `graph_constrained_top1_per_pair` cần ghi cùng bảng số,
  không so trực tiếp với bảng dùng protocol khác.
- GAT/GAT+KG dùng micro-batch 512 đồ thị như các run trước. Nếu GPU hết bộ
  nhớ, dừng ma trận và điều chỉnh batch/accumulation cùng bộ kiểm tra protocol
  trước khi chạy lại; không đổi riêng một điều kiện hoặc trộn report khác
  cấu hình.

## 1. Tải JPEG đầy đủ

Trước tiên chạy phần annotation trong [TEAM_SETUP_LEVEL1.md](../TEAM_SETUP_LEVEL1.md),
sau đó:

```bash
myenv/bin/python -m scripts.download_vg150_images --dry-run
myenv/bin/python -m scripts.download_vg150_images
```

Script tải có resume hai archive Visual Genome từ mirror đã dùng để review,
giải nén đúng 89.169 JPEG được `splits.json` tham chiếu, kiểm tra giải mã JPEG
và ghi `data/raw/vg150/images_manifest.json` với SHA-256 từng ảnh, SHA-256
archive và kích thước thực. Ảnh metadata sai kích thước không bị âm thầm bỏ.
Nguồn JPEG cụ thể là [images.zip](https://huggingface.co/datasets/pppop7/VisualGenome_VG_100K_1_and_2/resolve/main/images.zip)
và [images2.zip](https://huggingface.co/datasets/pppop7/VisualGenome_VG_100K_1_and_2/resolve/main/images2.zip)
trên mirror Hugging Face; trang [Visual Genome](https://visualgenome.org/api/v0/api_home)
liệt kê hai archive ảnh gốc. URL annotation nằm trong `scripts/download_vg150.sh`.
Archive và JPEG là dữ liệu local, không commit. Chạy lại cùng lệnh để xác minh
hoặc tiếp tục nếu tải dở.

## 2. Tạo cache vùng ảnh

```bash
myenv/bin/python -m scripts.cache_vg150_regions \
  --output-dir data/cache/vg150-clip-vit-b32-v1 \
  --device cpu --batch-size 4 --max-images 1
```

Lệnh trên là smoke CPU một ảnh, để lại `features.h5.partial`. Sau đó người
dùng chạy trích CLIP suy luận trên GPU, tiếp tục đúng cache đang dở:

```bash
myenv/bin/python -m scripts.cache_vg150_regions \
  --output-dir data/cache/vg150-clip-vit-b32-v1 \
  --device cuda --batch-size 64
```

Đây là **suy luận encoder**, không cập nhật trọng số. Cache hoàn tất có
`features.h5`, `manifest.json` và checkpoint CLIP trong `weights/`. Manifest
khóa hash dữ liệu, JPEG manifest, trọng số encoder, box policy và file vector.
Checkpoint OpenAI ViT-B/32 vừa tải có SHA-256
`40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`;
manifest cache phải ghi đúng hash này.
Script chỉ công bố manifest hoàn tất khi mọi ảnh của ba split đã có vector.
Nếu hết bộ nhớ GPU, giảm `--batch-size`; không đổi kiến trúc, trọng số hay
preprocessing. CPU smoke và GPU tiếp tục có cùng identity cache.

## 3. Sửa filter TransE và xuất embedding mới

Negative lúc train chỉ lọc triple train. Validation ranking chỉ lọc train+val;
test ranking sau khi chọn checkpoint mới lọc train+val+test theo filtered link
prediction thông thường. Không dùng triple KG test để tạo negative hoặc chọn
checkpoint. Đây là triple WordNet, không phải predicate VG150 test.

```bash
myenv/bin/python -m scripts.train_transe \
  --kg-dir data/knowledge/wordnet-kg-toan-v1 \
  --mode reviewed --review-manifest data/knowledge/wordnet-kg-toan-v1/review_manifest.json \
  --split-policy transductive \
  --output-dir runs/transe/wordnet-toan-reviewed-filtered-v2-seed42 \
  --dim 64 --epochs 200 --patience 20 --min-delta 0.0001 \
  --batch-size 512 --seed 42 --device cuda --wandb-mode online

myenv/bin/python -m scripts.export_knowledge \
  --checkpoint runs/transe/wordnet-toan-reviewed-filtered-v2-seed42/best.pt \
  --kg-dir data/knowledge/wordnet-kg-toan-v1 \
  --review-manifest data/knowledge/wordnet-kg-toan-v1/review_manifest.json \
  --mapping data/knowledge/wordnet-reviewed-toan-v1/wordnet_mapping.json \
  --output-dir data/knowledge/c2-wordnet-reviewed-filtered-v2
```

Lệnh đầu train GPU do người dùng chạy; lệnh export chỉ chạy CPU và phải in
`150/150 mapped classes`. Đường output mới tránh trộn với embedding trước sửa
filter. MRR TransE là KG link prediction, không phải PredCls Recall.

## 4. Ma trận 15 run và đánh giá

Khi cache và KG export đều hoàn tất, xem lệnh trước khi chạy:

```bash
myenv/bin/python -m src.level1_visual_matrix train --max-parallel 1 --dry-run
```

Người dùng chạy train GPU:

```bash
myenv/bin/python -m src.level1_visual_matrix train --max-parallel 1
```

Launcher tạo 3 seed cho mỗi điều kiện `mlp_semantic`, `gat`, `kg_on`,
`kg_off`, `edges_off`; checkpoint trong `runs/level1-clip-v1/`, log trong
`logs/level1-clip-v1/`. Mặc định chạy tuần tự để dễ phát hiện lỗi và không
vượt bộ nhớ. Output/log đã tồn tại sẽ bị từ chối, không ghi đè.
Nếu cần chạy riêng một ô sau khi xử lý lỗi, thêm `--condition gat --seed 42`
(thay tên/seed phù hợp) để launcher chỉ tạo đúng run còn thiếu.

Sau khi đủ 15 `best.pt`, người dùng chạy đánh giá GPU trên full validation:

```bash
myenv/bin/python -m src.level1_visual_matrix evaluate --device cuda --max-parallel 1 --dry-run
myenv/bin/python -m src.level1_visual_matrix evaluate --device cuda --max-parallel 1
```

Mỗi evaluator xác minh cùng hash cache và dữ liệu với checkpoint. Report nằm
ở `reports/level1/clip-v1-*-val5000.json`; summary 3 seed tại
`reports/level1/clip-v1-matched-3seed-val5000-summary.json`. Gửi lại log, 15
report và summary để đối chiếu paired delta, per-predicate và chọn phân tích
5 dự đoán đúng/5 sai. Chỉ khi khóa toàn bộ pipeline mới đánh giá test một lần.
