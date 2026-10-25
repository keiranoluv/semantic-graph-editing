# CLIP + ConceptNet: review và run mức 1

Nhánh này dùng ConceptNet 5.7 độc lập với WordNet. Không có cạnh `SameAs` hay
bridge WordNet↔ConceptNet. Vùng ảnh vẫn là cache CLIP ViT-B/32 đã dùng trong
15 run trước; dữ liệu VG150, ba seed và siêu tham số PredCls giữ nguyên.

**Trạng thái hiện tại:** Toan duyệt 134 đề xuất AI v2 và trực tiếp quyết định
16 nhãn focus: `accept` ID 11, 54, 123, 143; `reject` 12 ID còn lại. Hồ sơ
`data/knowledge/conceptnet-review-toan-v3/review.csv` có **138 accept,
12 reject, 0 defer, 0 pending**. Compiler đã tạo
`data/knowledge/conceptnet-reviewed-toan-v3/` và CPU preflight TransE PASS.
`review_basis=human_mixed_direct_and_ai_batch` giữ rõ 134 quyết định phê duyệt
đề xuất AI và 16 quyết định trực tiếp. `bike` và `glass` được chấp nhận theo
quyết định của Toan nhưng URI vẫn trộn nghĩa; rationale ghi rõ rủi ro đó.
Review này chỉ xác nhận liên kết class→URI, không xác nhận từng assertion.
TransE seed 42 đã có checkpoint và C.2 CPU export nằm ở
`data/knowledge/c2-conceptnet-toan-v3/` với 138/150 class có vector. Sáu ô
PredCls mới và GPU evaluate đã hoàn tất. Xem
[bảng 21 run CLIP trên validation 5.000 ảnh](../../reports/level1/clip-all-21run-val5000-summary.md);
các lệnh ở mục 3 dùng để tái lập khi tạo version output mới.

## 1. Hồ sơ review và cách kiểm lại

Hai file nháp gốc trên máy project:

- `data/knowledge/conceptnet-review-v1/review.csv` — 150 nhãn VG150, mỗi dòng
  là một quyết định. Bản v1 giữ nguyên 150 dòng trống để đối chiếu lịch sử.
- `data/knowledge/conceptnet-review-v1/evidence_samples.csv` — 1.705 cạnh mẫu,
  gồm URI assertion, hướng, relation, concept kề, dataset, license và weight.
  Đây là mẫu cao điểm theo từng loại relation/hướng để tìm bằng chứng, không
  phải chứng nhận tất cả 56.703 assertion đều đúng.

Theo yêu cầu của Toan, Codex còn tạo bản đề xuất sau khi kiểm lại toàn bộ
cạnh của 27 nhãn ban đầu bị hoãn:
`data/knowledge/conceptnet-review-ai-v2/review_suggestions.ai-draft.csv` với **134
accept, 14 defer, 2 reject đề xuất**. File này dùng WordNet representative
mapping đã chốt chỉ để hiểu nghĩa VG và chọn bằng chứng ConceptNet; mỗi dòng
ghi `proposed_by=Codex_AI_provisional`, `requested_human_approver=Toan`, URI,
assertion và lý do. Đây là bản nháp để Toan xem nhanh, không tự điền
`reviewer=Toan` hoặc mở cổng train. Hai nhãn được đề xuất reject là `pant`
và `short`; 14 nhãn defer cùng hai nhãn reject có bản lọc 16 dòng ở
`data/knowledge/conceptnet-review-ai-v2/review_focus.ai-draft.csv` và bảng dễ đọc
ở [conceptnet-ai-review-focus.md](../../reports/level1/conceptnet-ai-review-focus.md). Nếu Toan phê duyệt bản nháp
theo lô, cần lưu rõ rằng Toan **phê duyệt đề xuất AI**, không tự gán thành
150 quyết định đã kiểm chứng độc lập.

Bản AI v1 (121/27/2) còn nguyên tại `data/knowledge/conceptnet-review-v1/`;
không sửa hoặc ghi đè hash cũ. Bản v2 có 13 quyết định `accept` mới, mỗi dòng
ghi URI, assertion nguồn và lý do sau khi xem cạnh của tất cả URI ứng viên.
`review.csv` ở hai bản nháp vẫn 150 `pending`; bản Toan v3 là bản chính thức.

Quyết định của Toan được lưu nguyên văn ở
`reviews/conceptnet-toan-v3/approval.json`; bốn URI/assertion do
Codex chọn sau chỉ đạo accept nằm ở `reviews/conceptnet-toan-v3/selection.csv`.
Hai file nhỏ này được theo dõi bằng Git; KG và review CSV trong
`data/knowledge/` bị bỏ qua và tái dựng từ snapshot nguồn. File
`direct_decisions.csv` ghi 16 quyết định trực tiếp; 134 dòng còn lại của
`review.csv` ghi tiền tố `Batch-approved AI proposal`. Manifest KG khóa hash
tất cả các file này và đã được CPU gate kiểm tra.

Đường phê duyệt toàn bộ AI v2 dưới đây là phương án lưu trữ cũ; bản v3 hiện dùng
đường kết hợp quyết định trực tiếp và duyệt AI theo lô. Đường cũ yêu cầu một tuyên bố của Toan khóa SHA-256 của cả
file 150 dòng và file focus 16 dòng. `scripts.approve_conceptnet_suggestions`
chỉ nhận tuyên bố này làm đầu vào, tạo review directory **mới** và ghi
`reviewer=Toan` với tiền tố `Batch-approved AI proposal` ở mỗi lý do.
Manifest sẽ ghi `review_basis=human_batch_approved_ai_proposal`,
`human_verified_class_count=0` và `human_batch_approved_class_count=150`.
Codex không tự tạo tuyên bố phê duyệt thay Toan.

Lọc cột `priority` theo thứ tự `ambiguous_label` (26 dòng),
`multiple_candidate_uris` (112), `standard` (12). Nhóm đầu dùng trạng thái
WordNet cũ **chỉ để ưu tiên xem**; không lấy WordNet synset làm quyết định cho
ConceptNet. Có 135 nhãn có hai URI ứng viên, thường là `/c/en/<term>` và
`/c/en/<term>/n`. Hậu tố `/n` chỉ chỉ loại từ danh từ, không chứng minh đã
đúng nghĩa cụ thể. Kiểm tra nhãn đa nghĩa như `glass`, `light`, `short`,
`stand`, `trunk`, `tie`, `train` cẩn thận.

Với **mỗi dòng**, đối chiếu URI và assertion mẫu với nghĩa vật thể trong VG150,
rồi điền:

| Cột | Cách điền |
| --- | --- |
| `decision` | `accept`, `reject` hoặc `defer` |
| `selected_uri` | Nếu `accept`, chọn **đúng một** URI có trong `conceptnet_uri_candidates`; nếu không thì để trống |
| `reviewer` | Tên người thực sự duyệt |
| `evidence_reference` | Nếu `accept`, copy một `assertion_uri` thuộc đúng `selected_uri` từ file evidence; với `reject/defer`, ghi nguồn đã xem hoặc mã lý do như `no_suitable_candidate` |
| `rationale` | Một câu giải thích nghĩa và lý do chọn/loại/hoãn |

Ví dụ cấu trúc, **không phải quyết định thật**: với nhãn `airplane`, sau khi
người review kiểm `/c/en/airplane`, có thể ghi `accept`, URI đó, tên mình,
một assertion URI từ evidence và lý do. Nếu không chắc nhãn `glass` là chất
liệu hay cốc trong tập ảnh, dùng `defer` và nêu vấn đề; không ép chọn vì cần
đủ 150 accept. Không dùng predicate GT của VG validation/test để chọn URI.

Có thể kiểm tra bản đã duyệt bất cứ lúc nào:

```bash
myenv/bin/python -m scripts.prepare_conceptnet_review check \
  --review-dir data/knowledge/conceptnet-review-toan-v3 --require-complete
```

Lệnh báo số `accept/reject/defer/pending`, và dừng nếu URI, class ID, reviewer
hoặc assertion của dòng được chấp nhận không hợp lệ. Khi hết `pending`:

```bash
myenv/bin/python -m scripts.prepare_conceptnet_review compile \
  --review-dir data/knowledge/conceptnet-review-toan-v3 \
  --output-dir data/knowledge/conceptnet-reviewed-toan-v3 --dry-run
```

Nếu cần tái dựng bản v3 từ quyết định đã lưu, chạy
những lệnh CPU sau sau khi đã dựng graph ConceptNet B.2 từ snapshot ghim hash:

```bash
myenv/bin/python -m scripts.prepare_conceptnet_review prepare \
  --review-dir data/knowledge/conceptnet-review-ai-v2
myenv/bin/python -m scripts.suggest_conceptnet_review \
  --review-dir data/knowledge/conceptnet-review-ai-v2
myenv/bin/python -m scripts.apply_conceptnet_user_review \
  --approval reviews/conceptnet-toan-v3/approval.json \
  --selection reviews/conceptnet-toan-v3/selection.csv \
  --output-dir data/knowledge/conceptnet-review-toan-v3
myenv/bin/python -m scripts.prepare_conceptnet_review compile \
  --review-dir data/knowledge/conceptnet-review-toan-v3 \
  --output-dir data/knowledge/conceptnet-reviewed-toan-v3
```

Phương án cũ nếu dùng phê duyệt AI v2 theo lô là chạy `scripts.approve_conceptnet_suggestions`
với `--source-dir data/knowledge/conceptnet-review-ai-v2` và một file approval
khóa SHA-256 của đúng bản v2. Sau đó chạy `check` và `compile` với
`--review-dir data/knowledge/conceptnet-review-approved-ai-v2`; thư mục này
chỉ được tạo sau tuyên bố phê duyệt. Dùng output KG riêng cho nhánh đó, không
trộn với bản review thủ công từng class.

Compiler đã tạo `data/knowledge/conceptnet-reviewed-toan-v3/`, khóa hash nguồn,
CSV, liên kết class ID và manifest. Các class `reject/defer` có vector 0 và
valid bit 0 sau export. Nếu chỉnh lại review, dùng **version thư mục mới**;
không sửa KG/checkpoint đã dùng cho run. Review chỉ xác nhận liên kết class→URI;
assertion ConceptNet vẫn theo bộ lọc nguồn đã công bố, không được tự nhận là
56.703 cạnh đã duyệt thủ công. KG và review CSV tái dựng nằm dưới
`data/knowledge/` và không commit; teammate dùng hai file duyệt được theo dõi
và lệnh trên để tái dựng sau khi tải snapshot.

## 2. TransE ConceptNet và C.2

Chỉ chạy sau khi review hoàn chỉnh và compiler PASS. CPU dry-run trước:

```bash
myenv/bin/python -m scripts.train_transe \
  --kg-dir data/knowledge/conceptnet-reviewed-toan-v3 \
  --mode reviewed --review-manifest data/knowledge/conceptnet-reviewed-toan-v3/review_manifest.json \
  --split-policy transductive --val-ratio 0.08 --test-ratio 0.08 \
  --eval-max-triples 512 \
  --output-dir runs/transe/conceptnet-toan-v3-seed42 \
  --dim 64 --epochs 200 --patience 20 --min-delta 0.0001 \
  --batch-size 512 --seed 42 --device cpu --wandb-mode disabled --dry-run
```

**Người dùng chạy GPU train**, cùng lệnh nhưng đổi `--device cuda`,
`--wandb-mode online` và bỏ `--dry-run`:

```bash
myenv/bin/python -m scripts.train_transe \
  --kg-dir data/knowledge/conceptnet-reviewed-toan-v3 \
  --mode reviewed --review-manifest data/knowledge/conceptnet-reviewed-toan-v3/review_manifest.json \
  --split-policy transductive --val-ratio 0.08 --test-ratio 0.08 \
  --eval-max-triples 512 \
  --output-dir runs/transe/conceptnet-toan-v3-seed42 \
  --dim 64 --epochs 200 --patience 20 --min-delta 0.0001 \
  --batch-size 512 --seed 42 --device cuda --wandb-mode online
```

Graph hiện có 46.414 entity, 56.703 unique triples. Split transductive
10%/10% mặc định **không khả thi** do không thể giữ đủ entity trong train;
8%/8% đã PASS preflight CPU trên chính graph này (47.631/4.536/4.536
triples). Validation/test MRR dùng mẫu 512 triple cố định để đủ nhanh; đây
là đánh giá link prediction của KG, không phải PredCls. Negative train chỉ lọc
triple train; validation ranking lọc train+val, test ranking lọc cả ba split.
Model chọn `best.pt` theo KG validation MRR.

Sau TransE, Codex có thể chạy export **CPU** (hoặc teammate chạy):

```bash
myenv/bin/python -m scripts.export_conceptnet \
  --checkpoint runs/transe/conceptnet-toan-v3-seed42/best.pt \
  --kg-dir data/knowledge/conceptnet-reviewed-toan-v3 \
  --review-manifest data/knowledge/conceptnet-reviewed-toan-v3/review_manifest.json \
  --output-dir data/knowledge/c2-conceptnet-toan-v3
```

## 3. PredCls: 6 run ConceptNet đã chạy và đánh giá

MLP và GAT thuần đã có cùng CLIP cache nên không cần train lại. Ba run
`kg_off` CLIP+WordNet trước cũng dùng GAT+KG 64D, cùng siêu tham số và ba seed;
trong model, `--no-knowledge` đặt **cả vector KG lẫn valid bit về 0**. Do đó
chúng là đối chứng KG-off hợp lệ cho ConceptNet, không cần train lại mặc định.
Chỉ cần train `kg_on` và `edges_off` với ConceptNet, mỗi điều kiện ba seed:
**6 run mới**. `edges_off` vẫn dùng vector ConceptNet nên phải chạy lại.
Tất cả dùng train 57.723 ảnh, validation 5.000 ảnh,
batch 512, tối đa 200 epoch, patience 15, chọn best theo validation mR@50.
Output/log riêng, không đè 15 run CLIP+WordNet. Nếu muốn kiểm tra tái lập
KG-off sau khi đổi nguồn embedding, có thể chạy thêm ba ô tùy chọn bằng
`--condition kg_off`; chúng không cần thiết cho so sánh chính.

Xem kế hoạch trên CPU:

```bash
myenv/bin/python -m src.level1_conceptnet_matrix train --max-parallel 3 --dry-run
```

**Người dùng chạy GPU train và GPU evaluate**:

```bash
myenv/bin/python -m src.level1_conceptnet_matrix train --max-parallel 3
myenv/bin/python -m src.level1_conceptnet_matrix evaluate --device cuda --max-parallel 1
```

Có thể chọn một ô, ví dụ `--condition kg_on --seed 42`, để chạy ô chưa có
output. Sau đủ 6 report mới, launcher tổng hợp 3 seed trong
`reports/level1/clip-conceptnet-v1-matched-3seed-val5000-summary.json`.
So sánh với 15 run CLIP trước, gồm ba report KG-off cũ, trên **cùng validation protocol**; ghi rõ **138/150**
class ConceptNet được chấp nhận. Chênh lệch WordNet↔ConceptNet còn phản ánh
coverage, kích thước KG và nguồn tri thức, không chỉ chất lượng từng assertion.

Snapshot tải một lần từ URL ConceptNet 5.7.0 chính thức và SHA-256 đã ghi ở
[b2_conceptnet_workflow.md](b2_conceptnet_workflow.md); teammate có lệnh tải
và dựng graph trong tài liệu đó.

Thử nghiệm embedding RotatE riêng sau 21 run CLIP được hướng dẫn tại
[rotate_conceptnet_workflow.md](rotate_conceptnet_workflow.md). Nhánh này dùng
chính ConceptNet reviewed v3 và đường dẫn output mới; không thay các kết quả
TransE hiện có.
