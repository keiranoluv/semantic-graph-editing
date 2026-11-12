# Chạy MagicBrush sau pilot v1: thứ tự, phạm vi và điều kiện dừng

Pilot v1 đã chạy 40 ảnh: 24 PASS cấu trúc, 16 ERROR (8 lỗi endpoint/schema,
7 bbox sai dạng, 1 thiếu key `bbox`). CPU compile có 7 cặp đủ hai graph,
trong đó chỉ 3 candidate tự động và 4 cần xem lại. Chưa có review hoàn thành.
Không mở rộng v1 thẳng lên full/training. Các kết quả v1 giữ nguyên.

Code v2 dùng cùng Qwen2.5-VL-3B/revision/seed và 20 ID review, prompt/schema rõ
hơn, tên node/predicate đơn giản, max output 2048 token, batch 1. Bbox có thể
đổi key `bbox_2d` hoặc bỏ đúng một chiều singleton mà không đổi tọa độ; ghi
lại mọi normalization. Không clip bbox, đoán ID endpoint hay xóa cạnh lỗi để
biến ERROR thành PASS. Chưa chạy inference v2, chưa có bằng chứng v2 tốt hơn.

Assistant chỉ chạy CPU, test và commit; **mọi lệnh GPU do người dùng chạy**.
Mỗi lệnh chạy một stage; không có tự nối tiếp full hay training. VG150 chạy
sau vì hiện ưu tiên MagicBrush. Không cần tải lại raw dataset hoặc cài package.

## 0. Chuẩn bị terminal

Tại repo root, dùng cùng terminal cho các biến dưới đây. Nếu mở terminal khác
thì khai báo lại. Các biến chỉ rút gọn đường dẫn, không tự chạy tác vụ nào.

```bash
cd /workspace/workspace/semantic-graph-editing
mkdir -p logs
MB_DATA=data/processed/level2/magicbrush-v1
MB_GRAPHS="$MB_DATA/graphs-qwen25vl3b-schema-v2-bs1"
MB_REVIEW=reviews/magicbrush-human-v2
MB_TRAIN_CONFIG=configs/level2/magicbrush-edit-model-v2.json
```

## 1. GPU: chỉ chạy lại pilot v2

Kiểm phạm vi bằng CPU trước:

```bash
myenv/bin/python -m tools.infer_magicbrush_graphs_v2 --review-only --plan
```

Sau đó người dùng chạy GPU:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 myenv/bin/python -u -m tools.infer_magicbrush_graphs_v2 --review-only --wandb-mode online > logs/magicbrush-graphs-pilot-v2.log 2>&1
```

Qwen2.5-VL-3B đóng băng, bf16, **batch 1, seed 42, 40 ảnh của cùng 20 cặp
train**. Không fine-tune/train model; không dùng câu chỉnh sửa/ảnh còn lại
trong input VLM. Cache mới ở `$MB_GRAPHS`, không ghi đè v1. W&B run name
`magicbrush-graphs-schema-v2-qwen25vl3b-bs1-pilot`. Lần này weights đã có trong
HF cache từ v1; không đổi model/revision.

Theo dõi ở terminal thứ hai nếu muốn:

```bash
tail -f logs/magicbrush-graphs-pilot-v2.log
```

Ctrl+C ở terminal `tail` chỉ dừng xem log. Đợi process inference hoàn tất rồi
kiểm tra trạng thái bằng CPU:

```bash
myenv/bin/python -m tools.magicbrush_review_gate_v2
```

Nếu còn ERROR/missing, dừng để phân tích. Không xóa prediction lỗi rồi chạy
lại cùng cache, không thay ID mẫu. Tool giữ raw responses và báo nguyên nhân;
đổi prompt/parser/model phải version mới, test và commit trước lệnh GPU mới.

## 2. CPU: căn chỉnh graph, sinh thao tác và dựng viewer

```bash
myenv/bin/python -m tools.magicbrush_graphs --graphs "$MB_GRAPHS" --output "$MB_DATA/edits-pilot-schema-v2.json"
myenv/bin/python -m tools.review_magicbrush --stage build --output "$MB_REVIEW" --edits "$MB_DATA/edits-pilot-schema-v2.json"
myenv/bin/python -m tools.review_magicbrush --stage serve --output "$MB_REVIEW"
```

Lệnh đầu chỉ dùng graph đã có: Hungarian alignment → sáu thao tác → kiểm
tái dựng target. Candidate chưa phải nhãn được con người duyệt. Các cặp khác
trong cohort vẫn pending vì chưa inference, không tự tạo graph giả.
Hai lệnh sau dựng HTML và mở server, không dùng GPU. Nếu cổng 8766 đang phục
vụ viewer cũ, dừng server đó trước; không dừng process không liên quan.

Mở/forward `http://127.0.0.1:8766`. Giữ terminal server trong lúc duyệt.

## 3. Người dùng kiểm đủ 20 cặp, rồi CPU khóa CSV thật

Với từng cặp, đánh giá graph nguồn, graph đích, alignment, thao tác và sự
phù hợp với câu yêu cầu. Nhập reviewer/notes, chọn đúng/sai/chưa rõ theo ảnh,
không ép thành đúng để đủ số lượng. Nhấn **Lưu vào reviews/** để có CSV
trong repo; chỉ bấm Xuất CSV sẽ lưu ở Downloads và chưa tạo submission phía server.

Dừng server bằng Ctrl+C rồi chạy:

```bash
myenv/bin/python -m tools.magicbrush_review_gate_v2 --select-review
```

Tool yêu cầu cả 40 graph đã qua cấu trúc, đúng 20 ID gốc, ≥20 kết luận
đúng/sai có tên và đủ năm trục, tỷ lệ đúng ≥80%. Đây là ngưỡng nội bộ khai
báo trước, không phải yêu cầu của đề hoặc bảo đảm độ tin cậy thống kê. Mẫu
uncertain không được tự đổi thành đúng. Chưa đạt thì **dừng**, sửa quy trình
nhãn train ở version mới và kiểm lại, không chuyển sang full/train.

Tool chỉ sao chép byte gốc của CSV mới nhất và packet tương ứng sang
`$MB_REVIEW/selected/`; không sinh review hoặc đổi quyết định. Hash/per-pair
evidence được kiểm lại. Review đối với graph cũ không được tự áp dụng cho graph mới.

## 4. GPU: full inference, chỉ khi bước 3 đạt

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 myenv/bin/python -u -m tools.infer_magicbrush_graphs_v2 --full --wandb-mode online > logs/magicbrush-graphs-full-v2.log 2>&1
```

Vẫn Qwen2.5-VL-3B đóng băng, batch 1, seed 42. Phạm vi **912 ảnh riêng biệt
của 602 cặp**, tái sử dụng 40 ảnh pilot cùng identity nên thường chỉ sinh
**872 ảnh mới**. Chỉ lúc này mới inference ảnh thuộc split val/test nội bộ;
prompt/policy đã khóa trên train, không dùng nội dung held-out để sửa chúng.
Official dev/test MagicBrush không được mở. W&B name kết thúc `-full`.

Launcher kiểm actual CSV/packet/prediction provenance và ngưỡng review trước
khi load GPU model, ghi `full-review-gate.json`. Không tự train sau inference.

## 5. CPU: tạo bộ nhãn đầy đủ và preflight training

```bash
myenv/bin/python -m tools.magicbrush_graphs --graphs "$MB_GRAPHS" --output "$MB_DATA/edits-full-schema-v2.json"
myenv/bin/python -m tools.train_magicbrush_edits --config "$MB_TRAIN_CONFIG" --stage plan
```

Lệnh đầu lọc lỗi, no-op, graph drift, alignment mơ hồ/ngoài schema; lệnh sau
kiểm CSV thật và bằng chứng của cùng 20 cặp, disjoint split, **≥300 cặp
train+val dùng được sau lọc**, cả hai split không rỗng. Không suy ra rằng
602 cặp ảnh thô chắc chắn cho ≥300 nhãn dùng được. `ready:false` hoặc exit 2
thì dừng và xem lý do, không bỏ qua bằng force/đổi cờ approved.

Giữ packet pilot đã chọn: cùng graph/nhãn/prediction hash có thể đối chiếu
với full export bằng per-pair evidence. Không cần build viewer đè lên packet
đã review, không cần kiểm lại nếu bằng chứng từng cặp hoàn toàn không đổi.

## 6. GPU: cache CLIP text đóng băng

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 myenv/bin/python -u -m tools.train_magicbrush_edits --config "$MB_TRAIN_CONFIG" --stage cache-text --wandb-mode online > logs/magicbrush-edit-text-v2.log 2>&1
```

Chỉ encode câu yêu cầu, nhãn/thuộc tính node và predicate **graph nguồn** của
train/val, CLIP ViT-B/32 đóng băng, **batch 128, seed 42**. Không encode target,
không train CLIP. Output `data/processed/level2/magicbrush-edit-model-v2/clip-text/`.

## 7. GPU: train trước một lượt MLP seed 42

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 myenv/bin/python -u -m tools.train_magicbrush_edits --config "$MB_TRAIN_CONFIG" --stage train --model mlp --seed 42 --wandb-mode online > logs/magicbrush-edit-mlp-seed42-v2.log 2>&1
```

Đây mới là training: **MLP encoder + GRU decoder**, CLIP text vẫn frozen,
**batch train 64, val 32, seed 42**, tối đa 100 epoch/patience10, chọn best
bằng val autoregressive micro operation F1. Output
`runs/level2/magicbrush-edit-model-v2/mlp/42/` có best/last, metric và dự đoán.
Kiểm decode failure/validity/F1 trước khi chạy các lượt còn lại; chưa có số
đo thực nghiệm để cam kết chất lượng. Nếu đổi recipe thì version/commit mới.

## 8. GPU: năm lượt còn lại, tuần tự sau khi chốt recipe

Giữ nguyên config/source/protocol, chạy lần lượt năm lệnh train với:

| Model | Seed | Log gợi ý |
|---|---:|---|
| mlp | 123 | `magicbrush-edit-mlp-seed123-v2.log` |
| mlp | 3407 | `magicbrush-edit-mlp-seed3407-v2.log` |
| gnn | 42 | `magicbrush-edit-gnn-seed42-v2.log` |
| gnn | 123 | `magicbrush-edit-gnn-seed123-v2.log` |
| gnn | 3407 | `magicbrush-edit-gnn-seed3407-v2.log` |

Dùng lệnh bước 7, thay `--model`, `--seed` và log cho từng dòng. Mỗi process
chạy một model/seed, không tự nối tiếp. Chỉ chạy khi lệnh trước kết thúc;
không chạy đồng thời với VG150. Sau đủ sáu kết quả mới tổng hợp mean/sample
SD và phân tích lỗi; test model cuối sẽ có lệnh riêng sau khi khóa lựa chọn.

Các stage GPU dùng chung lock với GQA/VG150; code chỉ bàn giao lệnh, không tự
khởi chạy. V1 source/cache/review/checkpoint/history vẫn được giữ nguyên.
