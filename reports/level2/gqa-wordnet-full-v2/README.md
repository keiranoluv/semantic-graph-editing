# GQA WordNet v2 — full GAT+KG ba seed

Đây là thực nghiệm **dự đoán quan hệ PredCls** trên GQA. GPU học trọng số
GAT, embedding nhãn object và head phân loại 310 predicate. Input là box/nhãn
object GT, đặc trưng CLIP từ cache, hình học và vector WordNet 64D đã học bằng
TransE. CLIP và vector WordNet đóng băng; không train detector hoặc mô hình
sinh ảnh, không đưa quan hệ GT vào topology hay KG.

Mapping/embedding v2 và [pilot](../gqa-wordnet-pilot-v2/README.md) đã hoàn tất.
Launcher được khởi chạy trong tmux `gqa-wordnet-v2`, với **một trainer mỗi thời
điểm**, thứ tự **42 → 123 → 3407**. Đây là mô tả lúc khởi chạy, không phải
tuyên bố cả ba seed đã hoàn tất. Trạng thái thực tế ở
`runs/level2/gqa-wordnet-v2-pipeline/status.json`; từng run có `history.json`,
`last.pt`, `best.pt`, và `result.json` sau khi hoàn thành.

## Recipe và W&B

- 69.942 train / 5.000 validation, cùng split/cache/evaluator với GAT cũ.
- Train/validation batch **512/512**, AdamW lr 0,001, weight decay 0,01,
  clip gradient norm 5, tối đa 200 epoch, early stopping patience 15 / delta 0.
- Best theo validation GC mR@50, bằng điểm giữ epoch sớm hơn; không scheduler.
- Tính toán float32, deterministic; 4 worker nạp RAM, 8 thread ghép tensor,
  1 numerical thread. Có kiểm digest khi chuyển dữ liệu giữa các tiến trình CPU.
- GAT+KG: 556.727 tham số; GAT gốc: 540.087. Cùng sampling được kiểm lại bằng
  trace các epoch chung với GAT. Vòng main không có KG-off ba seed, vì vậy
  chưa tách riêng lợi ích nội dung tri thức khỏi thay đổi số tham số.

W&B entity/project: `keiranoluv-hcmus/semantic-graph-editing`.
Tên run: `gqa-wordnet-v2-gat-kg-bs512-seed<seed>`; tag gồm `batch-512`,
`gqa-wordnet-v2`, `partial-human-review`, `sequential`, cùng tag seed.

- [TransE v2, đã hoàn tất](https://wandb.ai/keiranoluv-hcmus/semantic-graph-editing/runs/5s1mpswa).
- [GAT+KG seed 42](https://wandb.ai/keiranoluv-hcmus/semantic-graph-editing/runs/43b6ecc6cb4cd252efe0ad44).
- URL các seed tiếp theo sẽ nằm trong `wandb-latest.json` của từng thư mục run.

`experiment.json` đầy đủ gồm danh sách image ID được giữ ở runtime output;
[experiment-reference.json](experiment-reference.json) lưu checksum và cấu
hình gọn để theo dõi trong Git. Mapping có review mẫu 30 nhãn, độ phủ 701/1.699
lớp và 64,69% object train; không giả định mọi nhãn đã được duyệt tay.

## Kết quả và tiếp tục

Sau đủ ba seed, launcher tự xuất `summary.json` tại thư mục report này và run
output: R@50/100, mR@50/100 GC/NG, mean/sample SD, paired delta so với GAT và
long-tail. Metric lưu dưới dạng tỷ lệ; nhân 100 để thành phần trăm.

Nhóm long-tail được khóa trước train từ tần suất annotation **train**: head
62 lớp, body 124 lớp, tail 124 lớp, sắp giảm dần tần suất rồi tăng ID. Đây là
quy ước phân tích của dự án, không phải taxonomy benchmark mới.

Các lượt GPU tiếp theo do người dùng chạy sau khi code đã commit.
[Lệnh chạy/tiếp tục và cách theo dõi](../../../docs/LEVEL2_GQA_WORDNET_V2.md).
Chưa dùng held-out test GQA để chọn phương pháp hoặc báo kết quả main.
