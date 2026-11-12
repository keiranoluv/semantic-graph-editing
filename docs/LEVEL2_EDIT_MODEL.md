# Bước 4 — mô hình dự đoán thao tác đồ thị v1

Đã chuẩn bị loader, hai mô hình, cấu hình, checkpoint/resume, evaluator và CLI.
**Chưa train trên MagicBrush, chưa cache embedding thật và chưa chạy GPU trong
mốc triển khai này.** Kiểm thử học chỉ dùng fixture tổng hợp trên CPU; không
phải bằng chứng chất lượng model. Nhãn ảnh thật/human review vẫn cần hoàn tất
theo [pipeline MagicBrush](LEVEL2_MAGICBRUSH.md).

Thiết kế đáp ứng đầu vào graph nguồn + embedding câu đóng băng và sáu loại
thao tác ở trang 3–5 của [đề gốc](../Project01_Graph_Editing.pdf). Không sinh
ảnh ở bước này. Không thay đổi model PredCls hoặc thực nghiệm GQA/VG150 cũ.

## Kiến trúc và đầu vào

| Thành phần | MLP encoder + decoder | GNN encoder + decoder |
|---|---|---|
| Đầu vào | Graph nguồn, câu yêu cầu | Giống MLP |
| Text encoder | CLIP ViT-B/32 đóng băng, vector chuẩn hóa 512 chiều | Giống MLP |
| Node | Nhãn + thuộc tính qua CLIP, projection và embedding vị trí ID | Giống MLP, thêm hai lớp truyền thông điệp |
| Edge | Predicate qua CLIP, hai endpoint và embedding ID | Giống MLP; message có predicate và hướng |
| Truyền thông tin qua đồ thị | Không có các lớp message passing | Tổng/degree hai hướng, MLP cập nhật + LayerNorm |
| Decoder | GRU có attention trên node/edge và câu | Cùng loại decoder |
| Output | Chuỗi hành động UTF-8, đủ sáu loại | Giống MLP |

Đây là hai **encoder** khác nhau ghép cùng kiến trúc decoder, không gọi toàn
bộ baseline là một MLP thuần. Thêm baseline không học `no-edit` trên cùng
validation để phân biệt việc thực thi hợp lệ với khả năng dự đoán chỉnh sửa.

CLIP weights dùng lại file đã có, SHA256
`40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`;
không fine-tune CLIP. [API CLIP tại revision đã pin trong requirements](https://raw.githubusercontent.com/openai/CLIP/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/README.md)
cung cấp `encode_text` và context 77 token. Cache ghi hash của text bị cắt
dài, không diễn giải là model đã đọc toàn bộ câu dài. Train không gọi CLIP,
chỉ đọc vector đã detach; không có tham số text encoder trong optimizer.

`feature_texts` chỉ nhận graph nguồn và câu; không nhận target/operations.
Không encode target labels, ảnh đích, bbox đích, hoặc graph test. Node/cạnh
nguồn dùng toàn bộ semantic payload, kể cả node cô lập; không dùng đặc trưng
pixel hay geometry trong mô hình v1 này. Nguồn graph vẫn là pseudo annotation
của VLM, không phải graph GT được con người gán đầy đủ.

## Action codec và quy tắc ID

1. Đánh số lại ID nguồn theo thứ tự ID nguồn thành `n001...`, `e001...`;
   không nhìn graph đích để xây input.
2. Serialize nhãn thành JSON list gọn. Ví dụ đổi thuộc tính:
   `[["replace_node","n001","cup",["blue"]]]`.
3. Các trường `before` được khôi phục từ **graph nguồn** theo ID model dự đoán.
   Node thêm/thay cần nhãn + thuộc tính; cạnh thêm/thay cần cả hai endpoint
   và predicate; delete cần ID. Không ngầm xóa cạnh khi xóa node.
4. Vocabulary cố định 256 byte UTF-8 + PAD/BOS/EOS, tổng 259 mã. Không fit
   dictionary từ validation/test; nhãn mới vẫn biểu diễn được. Điều này không
   bảo đảm model sẽ hiểu hay sinh đúng nhãn chưa gặp.
5. New-node IDs chuẩn hóa độc lập theo nhãn/thuộc tính và cạnh incident tới
   source ID hoặc semantic payload của node mới. New-edge IDs theo bộ ba
   semantic sau khi map node. Thực hiện giống nhau cho prediction và nhãn,
   **không tìm correspondence dựa trên target GT**. Nếu signature node mới
   vẫn trùng/mơ hồ, báo `unsupported_ambiguous_new_instance_identity` và thống
   kê coverage bị loại, thay vì tự chọn một ID chuẩn dựa vào GT.

Output không được bảo đảm đúng grammar. Greedy decode thiếu EOS, UTF-8/JSON
lỗi hoặc sai schema được đếm là decode failure; validity và exact match bằng
0. Không tự sửa output thành graph đúng, không lấy thao tác từ graph đích.
JSON không parse được không có thao tác để tính TP/FP; các thao tác GT vẫn
là FN, đồng thời báo riêng tỷ lệ failure. Với JSON parse được nhưng giao
dịch lỗi, giữ các thao tác để chấm FP/FN, không báo giao dịch hợp lệ.
Nếu chỉ canonical ID mới bị mơ hồ nhưng transaction vẫn áp dụng được, validity
cấu trúc vẫn tính đúng và lỗi identity được báo riêng; exact match bằng 0.

## Loader, lọc và điều kiện trước khi train

Config: [magicbrush-edit-model-v1.json](../configs/level2/magicbrush-edit-model-v1.json).
Loader dùng cùng manifest/split của bước 1–3 và kiểm:

- Không trùng ID, source group, session hay RGB pixel hash qua split.
- Metadata nhãn phải khớp ID/câu/split gốc và hash manifest.
- Chỉ candidate qua bộ lọc mới được dùng; thao tác phải tái dựng đúng target.
  Không sửa file nhãn thành approved. Tất cả vẫn là pseudo labels.
- Candidate bị người kiểm đánh sai/chưa đủ căn cứ bị loại khỏi cohort học.
- Source budget 20 node/60 edge; tối đa 12 thao tác. Train target dài quá
  2048 byte-token kể cả EOS bị loại và đếm rõ. Validation dài quá budget
  vẫn ở mẫu số; model phải chịu lỗi decode/truncation của mình.
- Chỉ deserialize target/actions của **train/val**. Test chỉ dùng metadata
  cho audit split; không dùng nhãn test để chọn policy, cache text hoặc train.

Điều kiện launcher: **≥300 cặp train+val dùng được sau lọc**, cả hai split
không rỗng; **≥20 review có kết luận đúng/sai**, tỷ lệ đúng tối thiểu **80%**.
80% là ngưỡng chất lượng nội bộ khai báo trước trong config, không phải yêu
cầu của đề hoặc chứng minh độ tin cậy thống kê. `uncertain` không tính là
kết luận. Nếu chưa đạt, quay lại chất lượng nhãn train, không đổi test.

CSV phải có đúng schema của viewer, tên reviewer, đủ năm trục, đúng 20 ID
đã chọn và hash bằng chứng. Chưa có CSV thì không tạo giả. Có thể pin CSV
pilot cùng packet đã archive khi chuyển sang full export **chỉ nếu từng
cặp có evidence hash không đổi**. Đây là lựa chọn đầu vào tường minh qua CLI,
không tự chuyển review sang graph đã thay đổi. Mọi thay đổi graph của mẫu
được review buộc phải được kiểm lại với packet mới.

## CLI CPU hiện dùng được

Chạy tại repo root, không cần package mới ngoài requirements hiện có:

```bash
myenv/bin/python -m tools.train_magicbrush_edits --stage plan
myenv/bin/python -m pytest -q tests/editing/test_edit_model_v1.py tests/editing/test_magicbrush_pipeline.py tests/editing/test_magicbrush_review_browser.py tests/editing/test_operations.py
```

`plan` trả exit code 2 khi chưa đủ điều kiện, không gọi CUDA, không tải weights,
không tạo training run. Hiện thiếu `edits-v1.json` và CSV kiểm tay; đây là trạng
thái chờ dữ liệu thật, không phải lỗi cần bỏ qua bằng một cờ force.

## Lệnh GPU bàn giao, chỉ chạy sau khi nhãn/review đủ điều kiện

Assistant không khởi chạy các lệnh dưới đây. Code phải được commit trước;
người dùng chạy từng lệnh khi job GQA hiện tại đã kết thúc. Các stage dùng
cùng khóa thiết bị với GQA, không có hàng đợi tự nối tiếp.

Trước tiên chọn **CSV thật** và **packet archive tương ứng** bằng hai tham số
`--reviews-csv` và `--review-packet`; dùng cùng hai đường dẫn ở mọi stage.
`selected-review.csv` trong config chỉ là placeholder, không có file duyệt sẵn.
Các dòng dưới dùng hai biến task-local do người dùng điền đường dẫn thật:

```bash
EDIT_REVIEW_CSV=reviews/magicbrush-human-v1/submissions/review-REPLACE_WITH_ACTUAL_HASH.csv
EDIT_REVIEW_PACKET=reviews/magicbrush-human-v1/packets/REPLACE_WITH_ACTUAL_PACKET_HASH.json
myenv/bin/python -m tools.train_magicbrush_edits --stage plan --reviews-csv "$EDIT_REVIEW_CSV" --review-packet "$EDIT_REVIEW_PACKET"
```

Cache text **inference đóng băng**, không training: CLIP ViT-B/32, batch 128,
seed 42, chỉ text đầu vào train/val. Output
`data/processed/level2/magicbrush-edit-model-v1/clip-text/`:

```bash
mkdir -p logs
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 myenv/bin/python -u -m tools.train_magicbrush_edits --stage cache-text --reviews-csv "$EDIT_REVIEW_CSV" --review-packet "$EDIT_REVIEW_PACKET" --wandb-mode online > logs/magicbrush-edit-text-v1.log 2>&1
```

Sau khi cache hoàn tất, train **một** model/seed, batch 64, validation batch 32:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 myenv/bin/python -u -m tools.train_magicbrush_edits --stage train --model mlp --seed 42 --reviews-csv "$EDIT_REVIEW_CSV" --review-packet "$EDIT_REVIEW_PACKET" --wandb-mode online > logs/magicbrush-edit-mlp-seed42-v1.log 2>&1
```

Lệnh chỉ train **MLP encoder + GRU**, seed 42, không tự train GNN hoặc seed kế.
Sau đó chạy riêng các tổ hợp `--model mlp|gnn` và `--seed 42|123|3407` cho
thực nghiệm chính, tuần tự một model tại một thời điểm. Chưa benchmark peak
VRAM/throughput GPU; batch 64 là cấu hình chuẩn bị, không phải số đã đo.

Optimizer AdamW lr 0.0003, weight decay 0.01, clip gradient 5, hidden 192,
byte embedding 64; tối đa 100 epoch, patience 10. Teacher forcing chỉ ở train;
validation luôn autoregressive. Chọn best theo validation micro operation F1,
bằng điểm giữ epoch sớm hơn. Không chọn bằng teacher-forcing loss hoặc test.

W&B project `keiranoluv-hcmus/semantic-graph-editing`:

- Cache run: `magicbrush-edit-clip-text-bs128`.
- Train run: `magicbrush-edit-{mlp|gnn}-seed{seed}-bs64`, tags nêu task/model/seed/batch.

Output dưới `runs/level2/magicbrush-edit-model-v1/{model}/{seed}/` có
`identity.json`, `best.pt`, `last.pt`, `result.json`, `predictions-val.json`.
Root có experiment/preflight manifest. Identity khóa config, mã nguồn mới,
CLIP implementation, weights hash, dữ liệu/review hash, ID cohort và cả
manifest/vector hash của text cache. Thay input, code, policy hoặc batch thì
dùng version output/cache/config mới; không ghi đè lịch sử.

Resume tự động ở **ranh giới epoch** từ `last.pt`, gồm optimizer/RNG/history;
nếu ngắt giữa epoch thì epoch chưa lưu được chạy lại. Test CPU đã so khớp
tensor model sau chạy liên tục và sau giả lập ngắt/resume. Chưa kiểm tính
bitwise deterministic của CUDA. Run đã hoàn tất chỉ verify hash và trả kết quả,
không train lại khi gọi cùng lệnh.

## Metric và giới hạn diễn giải

Báo P/R/F1 từng loại trong sáu thao tác, micro, macro trên loại có GT support,
exact operation set, transaction validity, exact target graph và decode failure.
`simple` được khai báo trước là **một thao tác GT canonical**, chỉ là proxy
đơn giản theo cấu trúc, không tuyên bố nhận diện độ phức tạp ngôn ngữ.
Report không biến loại vắng support thành F1=1. Trước khi báo mean/sample SD
cần đủ ba seed cho mỗi model; mốc này chưa có điểm thực nghiệm thật nào.

Các số đo tương lai chấm trên **pseudo target graph**, tách biệt với chất
lượng graph nhìn từ ảnh và chất lượng tuân thủ câu lệnh do người đánh giá.
Encoder mới chưa dùng bbox/pixel/KG; decoder byte có thể sinh JSON không hợp
lệ, và canonical new-instance policy v1 chưa hỗ trợ mọi graph có đối xứng.
Coverage/loại lỗi bị loại phải đi cùng bảng điểm, không chỉ báo score trên
những mẫu còn lại.

Bước tiếp sau khi đủ nhãn: cache text → pilot MLP seed 42 → kiểm lỗi/decoder
validity trên train/val → khóa protocol → MLP/GNN ba seed tuần tự. Giữ test
để dành sau khi chốt mô hình; demo/đánh giá test sẽ có lệnh GPU riêng.
