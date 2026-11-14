# AlignG adapted to frozen-CLIP PredCls — P0

P0 triển khai kiến trúc và kiểm chứng CPU. Không chạy GPU, W&B, huấn luyện
thực nghiệm hoặc đọc annotation held-out test. Các optimizer update trong
unit test chỉ dùng graph tổng hợp. Không có kết quả benchmark hay tuyên bố
cải thiện độ chính xác.

## Audit và phạm vi đề bài

- Bắt đầu tại `main`, HEAD `4e149ce75828f38d5bf81676ff8ac1eee7175279`, đúng
  baseline yêu cầu. Nhánh công việc: `feat/aligng-adapted-v1-p0`.
- Đã đọc `AGENTS.md`, `Project01_Graph_Editing.pdf` và toàn bộ các file audit
  được liệt kê trong yêu cầu. Mở rộng này phục vụ phương pháp mới cho nhiệm vụ
  dự đoán quan hệ; không thay thế các đối chứng MLP/GNN/GNN+KG, thực nghiệm
  hai dataset hoặc nhiệm vụ chỉnh sửa đồ thị của đề bài. Việc đổi dataset
  Task B và ghi nhận khó khăn vẫn thuộc nhánh công việc riêng.
- `src/models/penet.py`: tái sử dụng toàn bộ module/tham số PE-Net, giữ nguyên
  thứ tự khởi tạo tham số nền. Relation encoder được chép công thức vào
  namespace mới để chèn feedback trước projection; có test đối chiếu riêng.
- `src/training/penet.py`: gọi trực tiếp soft CE/PL/PR hiện hữu, không chép loss.
  `src/training/gqa_pilot.py`: dùng nguyên `soft_targets` và `sample_pairs`.
  Sampler giữ mọi positive pair, tối đa 3 negative/positive; ảnh không GT
  tối đa 32 negative. Duplicate GT không tăng trọng số, nhiều predicate
  khác nhau trên cùng pair nhận phân phối đều.
- `src/graph/builder.py`: full ordered pairs không self-loop; `edge_index`
  độc lập nhãn. `edge_label_index` giữ thứ tự prediction candidates.
  `src/datasets/batching.py` cung cấp loader ảnh có thứ tự, có thể tái sử dụng
  cho P1; không dùng PyG batching ngầm để suy ra ranh giới ảnh.
- `src/training/gqa_full.py`, `tools/run_gqa_full.py` và
  `tools/vg150_penet_v1/experiment.py`: đã kiểm tra identity, sampler RNG,
  checkpoint/resume, evaluator và W&B. Không monkeypatch trainer hay tạo
  launcher trong P0: forward mới cần context/query/node pointers tường minh.
- Hai evaluator `src/evaluation/reference_predcls.py` và
  `src/evaluation/gqa_pilot.py` giữ nguyên: IoU inclusive XYXY, duplicate GT,
  image-macro per-predicate, denominator cố định 50 hoặc 310. Test tích hợp
  kiểm query slicing của adapter qua cả GC và NG.
- Đã đọc báo cáo GQA full v3, VG150 PE-Net v1 và `docs/LEVEL2_VG150_PENET.md`.
  GQA hiện có 15 lượt validation của 5 model/3 seed; VG150 PE-Net v1 mới ở
  mức chuẩn bị. Custom GQA 310 foreground **không phải GQA200**.
- Asset cục bộ có cache CLIP, GloVe, manifest và checkpoint. Chỉ đọc metadata
  được ghi trong `provenance.json`; không load ví dụ train/val/test thật.
- GQA khóa glob toàn bộ `src/**/*.py`: 70 file `src/` và 3 runner/reference
  file trong identity GQA v3 được kiểm hash. Không thêm/sửa `src/`, config,
  manifest, checkpoint, runner hoặc báo cáo lịch sử. Source verifier cũ vẫn
  tương thích; P0 không chạy lại inference để xác minh checkpoint cũ.
- Những thay đổi có sẵn ở `AGENTS.md`, notebook cá nhân và các artifact
  MagicBrush/EMNLP/review không được sửa hoặc đưa vào commit này.

## Nguồn upstream và kiến trúc

[Paper ICML 2026](https://proceedings.mlr.press/v306/jung26a.html),
[source ghim](https://github.com/Namgyu97/AlignG-SGG.pytorch/blob/ba2dd471fb3c3bb39030660dcb7f6f105c52446c/maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py).
Commit upstream: `ba2dd471fb3c3bb39030660dcb7f6f105c52446c`.
`provenance.json` ghi SHA256 full source tải về, LICENSE và excerpt `SGGLayer`.
Fixture `tests/aligng_adapted_v1/upstream_reference.py` chứa nguyên văn class,
chỉ thay phần import để không cần detector. Test hash xác minh cả class lẫn
file fixture. Giữ notice trong `LICENSE-AlignG.txt` và `licenses/PENET-MIT.txt`.

Luồng mặc định, D=256, hai attention head:

1. Object GloVe học được + geometry 5D/CLIP frozen 512D; subject/object gates,
   fusion và ordered pair proxy như PE-Net hiện hữu.
2. Mỗi ảnh tạo `H_i` từ **tất cả** cặp context. Query relation `Q_i` lấy bằng
   chỉ mục từ `H_i`; không encode lại query với một dropout mask khác.
3. Chuẩn hóa global `P` bằng LayerNorm. `proto_q(P_norm)` attend tới
   `edge_k(H_i)`, `edge_v(H_i)`, softmax theo tất cả context của ảnh.
   Attention dropout → weighted sum → `update_norm_p` →
   `GRUCell(update, P_norm)` tạo `P_i`.
4. `edge_q(LayerNorm(Q_i))` attend tới `proto_k(P_i)`, `proto_v(P_i)`.
   Attention dropout → weighted sum. Ghép với `LayerNorm(Q_i)` rồi qua
   `edge_ffn`, một Linear 2D→D đúng upstream (không thêm activation/residual).
5. ReLU, dropout, shared projection D→2D như PE-Net. Cosine logits với
   learned temperature dùng **global projected P**, không dùng `P_i`.
   PL cũng dùng updated projected relation và global projected P.

`P_i` chỉ là tensor cục bộ trong forward, không lưu vào parameter/buffer và
không dùng làm state của ảnh tiếp theo. Không có BatchNorm trong adapter.

Khác biệt được chủ ý giữ so với upstream:

| Thành phần | Adapter P0 |
| --- | --- |
| Thiết lập | Frozen-CLIP PredCls, không detector/SGCls/SGDet |
| Feature | CLIP + geometry và ordered pair proxy thay ROI/union feature |
| Semantic dim | 256 thay 2048 của SGGHead upstream; projection 512 |
| Context/query | Context đầy đủ độc lập nhãn; query lấy mẫu riêng khi train |
| Supervision | Soft target đa positive, background ID0; C=51 hoặc C=311 |
| Loss | PE-Net hiện có: k1=10, k2=1, gamma1=1, gamma2=7, trọng số đều 1 |
| Margin upstream | Không dùng gamma=3/gamma2=20 của AlignG upstream |
| Empty context | Không GRU update; fallback normalized global P |
| Empty query | Output [0,C], bỏ optimizer step kể cả AdamW decay/momentum |
| Memory | Chia hàng attention query và pair encoder, checkpoint activation |
| Norm classifier | Giữ epsilon 1e-8 của baseline để tránh chia cho 0 |

## API và ablation

Các file triển khai: `model.py`, `feedback.py`, `batching.py`, `training.py`.
Không tự động đọc dataset hoặc khởi chạy job khi import.

```python
from experiments.aligng_adapted_v1 import AlignGConfig, AlignGPredicateClassifier
from experiments.aligng_adapted_v1.batching import collate, inference_batch, model_inputs
from experiments.aligng_adapted_v1.training import batch_loss

model = AlignGPredicateClassifier(
    object_vectors, predicate_vectors,
    config=AlignGConfig(loss_mode="plpr"),
)
# generator riêng cho sampling; các graph dùng cache/manifest lịch sử đã kiểm.
batch = collate(graphs, model.num_classes, generator=sampler_generator)
logits, relations, global_prototypes = model(**model_inputs(batch), return_embeddings=True)
losses = batch_loss(model, batch)
# Inference không cần đọc predicate annotations:
model.eval()
prediction = model(**model_inputs(inference_batch(graphs)))
```

Batch có `x`, `object_labels`, `node_ptr`, `context_pairs`, `context_ptr`,
`query_pairs`, `query_ptr`; supervised batch thêm `targets`. `model_inputs`
chỉ chuyển các trường input đã cho phép sang device. Forward không nhận
`targets`; toàn bộ `x` được detach kể cả khi caller bật `requires_grad`.
Context kiểm đủ N(N−1) unique pairs, endpoints đúng ảnh. Query có thể đảo
thứ tự/lặp nhưng phải cùng ảnh; thứ tự đầu ra giữ nguyên. Pointers có thể
trùng ở ảnh rỗng. Từ chối batch không chứa ảnh, chấp nhận ảnh không node.

| Flag | Mặc định và ý nghĩa |
| --- | --- |
| `feedback_enabled` | True; False gọi PE-Net forward gốc |
| `prototype_update_enabled` | True; False dùng normalized global P cho reverse-only |
| `reverse_feedback_enabled` | True; False tương đương baseline, bỏ computation feedback vô ích |
| `prototype_update_mode` | `gru`; residual/EMA bị từ chối rõ ràng trong P0 |
| `loss_mode` | `plpr`; hỗ trợ `ce`, `pl`, `plpr` qua loss gốc |
| `num_attention_heads` | 2; phải chia hết D |
| `attention_query_chunk_size` | 64; mỗi hàng vẫn softmax trên toàn bộ keys |
| `pair_chunk_size` | 1024; không thay thứ tự candidate |
| `checkpoint_chunks` | True; training recompute với RNG dropout được bảo toàn |

Khi tắt feedback, model vẫn chứa tham số feedback để load state đồng nhất;
các tham số đó không nhận gradient. `load_baseline(original_model)` kiểm đầy
đủ key/shape và chép nguyên trọng số nền. Với cùng seed và vectors, phần
tham số PE-Net khởi tạo giống hệt; thêm module feedback làm thay đổi RNG
sau constructor, vì vậy P1 cần tách/reset RNG train khi thiết kế matched run.

## Memory và giới hạn kiểm chứng

Không tạo tensor [pairs, classes, D]. Attention chia theo query row, không
chia softmax thành các nhóm context độc lập. Peak attention block tỷ lệ
H×min(query_chunk, query_rows)×key_rows; K/V/context vẫn cần O(MD).
Checkpoint giúp không giữ toàn bộ attention map/encoder activation cho
backward. Output logits và loss vẫn cần O(E×C); tổng context tăng theo tổng
số pair trong batch. Đây không phải cam kết VRAM cho batch lớn.

Chunk size có thể đổi thứ tự tiêu thụ RNG dropout khi train; không bảo đảm
bitwise-equivalent train giữa hai chunk size. Với cùng chunk size, bật/tắt
checkpoint cho cùng giá trị, gradient và RNG. Evaluation không dropout
khớp dense attention trong tolerance được test. Cô lập ảnh được kiểm ở eval
và train với dropout=0; stochastic mask khi dropout>0 không được xem là
coupling ngữ nghĩa giữa ảnh.

Số tham số với text 300D/feature 517D/semantic 256D:

| Dataset | PE-Net nền | Feedback thêm | Tổng adapter |
| --- | ---: | ---: | ---: |
| VG150, 151 object classes gồm background, C=51 | 1,924,409 | 922,368 | 2,846,777 |
| Custom GQA, 1700 object classes gồm background, C=311 | 2,467,109 | 922,368 | 3,389,477 |

## Kiểm chứng CPU

Chạy từ repo root, môi trường hiện có; không cài thêm package:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 myenv/bin/python -m pytest -q tests/aligng_adapted_v1 tests/training/test_penet.py tests/training/test_gqa_pilot.py tests/training/test_gqa_full.py tests/training/test_gqa_prepared_loader.py tests/training/test_vg150_penet_v1.py tests/datasets/test_batching.py tests/graph/test_builder.py tests/evaluation/test_reference_predcls.py -k 'not cuda' --junitxml=/tmp/aligng-p0-cpu.xml
myenv/bin/python -m black --check experiments/aligng_adapted_v1 tests/aligng_adapted_v1
myenv/bin/python -m isort --check-only experiments/aligng_adapted_v1 tests/aligng_adapted_v1
myenv/bin/python -m ruff check experiments/aligng_adapted_v1 tests/aligng_adapted_v1
```

Kết quả cuối và source hashes ở `verification.json`. Bộ test kiểm:

- SGGLayer upstream small synthetic: output và toàn bộ parameter/input
  gradients double precision; chunked/unchunked, dropout và checkpoint RNG.
- Tắt feedback: logits, projected relation và global prototype **exact**
  (`atol=rtol=0`) ở C=51/311, queries sampled/all, nhiều ảnh/ảnh rỗng.
- Encoder active và classifier global đối chiếu độc lập với baseline +
  upstream layer; test tránh việc chỉ kiểm nhánh bypass.
- A một mình, [A,B], [B,A], thay đổi B; train dropout=0 lẫn eval. Query
  reorder/repeat, context permutation, direction, không giữ prototype state.
- Thay nhãn/sampling không đổi context; cùng RNG/dropout không đổi P_i;
  bỏ predicate annotations vẫn inference được; sampler RNG/targets khớp gốc.
- D=256/C=51 và 311, CE/PL/PLPR finite forward/backward, mọi parameter cần
  học có gradient hữu hạn khác 0, input frozen không nhận gradient.
- Empty/singleton không đổi parameter hoặc optimizer momentum/step/decay.
- Graph tổng hợp 126 object/15,750 context và toàn bộ 15,750 eval query
  qua forward CPU C=311/D=256, sparse query khớp full query. Không đo throughput.
- Integration qua evaluator GC/NG nguyên bản, giữ duplicate GT/denominator;
  hồi quy PE-Net, GQA, prepared loader, VG150, builder và batching.
- Inventory/hash toàn bộ nguồn GQA khóa vẫn khớp; excerpt upstream bất biến.

P0 chưa kiểm GPU, mixed precision, VRAM, real-data training, resume một
thực nghiệm adapter hoặc online W&B. Không dùng unit test để kết luận chất
lượng dự đoán hoặc tái lập điểm paper.

## Provenance và khuyến nghị P1

Future run phải khóa Git commit, hash `src/` cùng toàn bộ extension/runner,
upstream commit, version môi trường, config model/loss/sampler/chunking,
dataset/split/cohort, CLIP/cache/GloVe mapping và binary hashes, seed,
parameter count, checkpoint hash, evaluator/selection policy và deviations
ở trên. `provenance.json` lưu baseline/source và metadata hash đã đọc trong
P0; không thay thế việc verify cache/vector bytes ở preflight P1.

Khuyến nghị pilot đầu tiên (chưa có launcher trong P0, **chưa chạy**):

- Model: **PE-PLPR + AlignG**, D=256, 2 heads, GRU, dropout 0.2; margins/
  unit weights giữ nguyên. AdamW lr=0.001, weight_decay=0.01, clip_norm=5.
- Dataset: **custom GQA C=311**, tái sử dụng cohort 512 train/128 val tại
  `reports/level2/gqa-pilot-v1-cohort.json`, split/cache/GloVe hiện hữu đã pin.
- Train batch **1 ảnh**, effective batch **1**, không accumulation;
  validation batch **1**. Đây là điểm bắt đầu thận trọng, chưa đo VRAM GPU.
- Seed **42**, tối đa **5 epoch**, chọn validation GC mR@50 cao nhất, hòa
  giữ epoch sớm hơn; giữ evaluator GC/NG và 310-class denominator.
- Output riêng:
  `runs/level2/aligng-adapted-v1/p1-gqa-pilot/pe-plpr-aligng/42/`.
- Một lệnh tương lai chỉ chạy **một model/seed**, không queue/nối tiếp job,
  không tự chạy full/test; dùng shared device lock với các experiment khác.
- Viết/kiểm CPU và commit launcher P1 trước khi bàn giao người dùng chạy;
  bật W&B online `keiranoluv-hcmus/semantic-graph-editing`, kiểm resume/RNG/
  checkpoint identity. P0 không cung cấp lệnh GPU giả tới runner chưa tồn tại.
- Sau pilot kỹ thuật, chạy đối chứng PE-PLPR feedback-disabled với cùng
  cohort/batch/seed và initial weights trong một job riêng. Batch1 pilot
  không phải matched comparison với kết quả full GQA batch512 lịch sử.
  Các lượt chính sau này dùng seeds **42, 123, 3407**, mỗi lúc một model.

Giữ test held out đến khi chốt protocol/model. P1 không được tự động mở
test annotation hoặc sửa thực nghiệm cũ để thêm AlignG.
