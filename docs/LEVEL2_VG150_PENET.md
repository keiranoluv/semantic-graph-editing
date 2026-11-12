# VG150 — PE-Net adapted và đối chứng đa nhãn v1

Pipeline mới đã chuẩn bị bằng CPU, không sửa `src/`, split, cache CLIP,
KG, checkpoint hoặc báo cáo lịch sử. GPU vẫn ưu tiên MagicBrush theo yêu cầu;
assistant không chạy lệnh GPU. Các lệnh bên dưới dành cho người dùng sau khi
code đã commit và GPU rảnh.

## Phạm vi và lý do cần vòng đối chứng mới

Các checkpoint Level 1 dùng projection single-label: pair có nhiều predicate
khác nhau bị bỏ khỏi train. Vòng này dùng cùng supervision đa nhãn như GQA:
giữ mọi positive pair, phân phối CE đều trên các predicate GT khác nhau;
dòng GT trùng không tăng trọng số train nhưng vẫn giữ để chấm reference recall.
Vì thế đây là **protocol train mới**, không coi điểm Level 1 cũ là matched control.

Sáu model: **MLP, GAT, GAT+WordNet-KG, PE-CE, PE-PL, PE-PLPR**.
Mỗi model chính chạy **42, 123, 3407**, tổng **18 lượt chính**; trước đó có
**6 pilot**, mỗi model một seed 42. Mỗi lệnh Python chỉ chạy một model/seed;
không tự nối tiếp model hoặc seed kế. Không huấn luyện detector hay CLIP mới.

PE-Net vẫn là bản **adapted to frozen-CLIP PredCls** đã dùng trên GQA: GloVe
khởi tạo embedding học được, relation prototype, PL/PR; feature pair proxy từ
hai object thay union-box ROI. Không gọi đây là tái lập nguyên bản điểm paper.
Giữ attribution và MIT notice ở `licenses/PENET-MIT.txt`; mọi khác biệt đã mô
tả trong [quyết định phương pháp](LEVEL2_METHOD_DECISION.md).

## Dữ liệu, feature và GloVe

- Giữ split VG150 hiện có: **57.723 train / 5.000 val / 26.446 test**. Pin
  SHA256 `splits.json`, HDF5 annotation, aligned metadata và dictionary; kiểm
  disjoint cả row và image ID, không chia lại.
- Candidate/message topology: toàn bộ cặp có thứ tự không self-loop,
  độc lập với GT predicate. Self-GT nếu có bị loại và đếm rõ; giữ các dòng
  duplicate và nhiều predicate khác nhau trên cùng pair.
- Mọi model dùng cùng **geometry 5D + CLIP region 512D** (517D), object labels
  GT, cùng candidate/sampling. Cache có sẵn được kiểm SHA256 và bbox alignment;
  không ghi lại `data/cache/vg150-clip-vit-b32-v1/`.
- GAT+KG tái sử dụng bundle WordNet
  `data/knowledge/c2-wordnet-reviewed-filtered-v2`, gồm review provenance,
  class/tensor hash, 150 lớp hợp lệ. Không dùng embedding ID của GQA.
- Chỉ **train** được đọc để tính tần suất và GloVe coverage. Taxonomy chuẩn
  giữ 150 object/50 predicate, thêm background ID0. Không tạo vocabulary từ
  val/test; các trường `object_count`/`predicate_count` có sẵn trong dictionary
  không dùng làm thống kê train.
- GloVe 6B 300D dùng nguồn/text hash đã kiểm của GQA, mapping riêng VG150:
  **150 object exact; 30 predicate exact, 20 predicate word mean; không có
  foreground random fallback**. ID0 khởi tạo xác định theo seed 42. PE embedding
  được học tiếp, không gọi GloVe là KG WordNet.
- Test annotation labels không được load. Val chỉ dùng để kiểm kết nối và
  chọn checkpoint; không dùng chọn mapping/prototype/prior/tần suất.

Manifest, vector mapping và thống kê ở `data/processed/level2/vg150-penet-v1/`;
bản bằng chứng nhỏ tại [reports](../reports/level2/vg150-penet-v1/README.md).
Binary/source lịch sử đều giữ nguyên.

## CLI chuẩn bị và kiểm tra CPU

Chạy từ repo root, dùng requirements hiện có:

```bash
myenv/bin/python -m tools.run_vg150_penet --stage prepare
myenv/bin/python -m tools.run_vg150_penet --stage plan --scope pilot
myenv/bin/python -m tools.run_vg150_penet --stage plan --scope full
```

Prepare kiểm hash asset, audit toàn bộ train, ghi GloVe mapping và khóa cohort
pilot **512 train / 128 val** bằng SHA256 trước khi có kết quả. Các model dùng
cùng thứ tự rows. Plan không load model, không khởi tạo CUDA. Output khác nội
dung ở cùng version bị từ chối; muốn đổi mapping/protocol dùng version mới.

```bash
myenv/bin/python -m pytest -q tests/training/test_vg150_penet_v1.py tests/training/test_penet.py tests/training/test_gqa_full.py tests/evaluation/test_reference_predcls.py -k 'not cuda'
```

`not cuda` cố ý loại các test GPU khỏi lượt kiểm này; không diễn giải các test
bị loại là đã qua GPU. Tests mới dùng graph tổng hợp trên CPU, kiểm đủ sáu
model, sampler đa nhãn, last class 50/150, duplicate GT, empty batch, checkpoint
và resume mid-epoch. Private trainer namespace không thay đổi global functions
của trainer GQA đang được dùng bởi thực nghiệm khác.

Ngoài fixture, có CPU wiring smoke trên một ảnh train thật với cấu hình mặc
định: forward/backward sáu model, gradient hữu hạn, **0 optimizer update**;
một ảnh val kiểm kết nối/cache. Không dùng các kiểm này làm điểm pilot hay
bằng chứng lợi ích PE-Net. Xem `cpu-wiring-smoke.json`.

## Lệnh GPU bàn giao — chạy sau nhánh MagicBrush đang ưu tiên

Config [vg150-penet-v1.json](../configs/level2/vg150-penet-v1.json): batch
train/val **128**, AdamW lr **0.001**, weight decay **0.01**, gradient clip **5**.
Những con số này đã kiểm trên CPU; peak VRAM/throughput GPU chưa được đo.
Nếu cần đổi batch, tạo config/output version mới và commit trước khi chạy.

Pilot đầu tiên để kiểm model phức tạp nhất, **PE-PLPR seed 42**, 512/128 ảnh,
**5 epoch**, không early stopping:

```bash
mkdir -p logs
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 myenv/bin/python -u -m tools.run_vg150_penet --stage pilot --model pe-plpr --seed 42 > logs/vg150-penet-v1-pilot-pe-plpr-seed42.log 2>&1
```

Lệnh này chỉ chạy một pilot, không chạy full. Làm tương tự cho các giá trị
`--model mlp`, `gat`, `gat-kg`, `pe-ce`, `pe-pl`, cùng seed 42, tuần tự.
Khi cả sáu xong, tổng hợp pilot bằng CPU:

```bash
myenv/bin/python -m tools.run_vg150_penet --stage summary --scope pilot
```

Full training chỉ được phép khi **cả sáu pilot PASS, hash/identity khớp,
sampling traces khớp và ba biến thể PE có cùng initial state**. Gate xác nhận
tính nhất quán pipeline, không yêu cầu PE phải thắng baseline để được PASS.
Lệnh ví dụ một lượt chính:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 myenv/bin/python -u -m tools.run_vg150_penet --stage train --model mlp --seed 42 > logs/vg150-penet-v1-full-mlp-seed42.log 2>&1
```

Đổi model/seed để chạy riêng 18 tổ hợp, không dùng nhiều job GPU đồng thời.
Các lệnh giữ cùng khóa GPU với GQA/MagicBrush; nếu thiết bị đang có job giữ
khóa thì từ chối chạy, không tự chờ rồi nối tiếp.

W&B entity/project `keiranoluv-hcmus/semantic-graph-editing`; tên run
`vg150-penet-v1-{pilot|full}-{model}-bs128-seed{seed}`, tags nêu dataset,
scope, multilabel, model, batch và seed. W&B giữ RNG và chỉ log epoch đã có
checkpoint bền vững. Online resume cùng logical run; offline giữ snapshot
riêng mỗi invocation như tracker GQA đã kiểm.

Output riêng:

- Pilot: `runs/level2/vg150-penet-v1/pilot/{model}/42/`.
- Chính: `runs/level2/vg150-penet-v1/full/{model}/{seed}/`.
- `experiment.json` khóa config/cohort, source, dữ liệu, CLIP, KG, GloVe.
- `last.pt` gồm optimizer, sampler RNG, Torch/CUDA RNG, vị trí giữa epoch,
  best state và history; SIGINT/SIGTERM lưu rồi thoát. `best.pt` là bản export
  có thể dựng lại từ last sau crash. Run hoàn tất được verify và bỏ qua,
  không train lại mặc định.

## Chọn checkpoint, chấm và tổng hợp

Full: tối đa **200 epoch**, early stopping patience **15**, min_delta **0**;
chọn **validation GC mR@50**, bằng điểm giữ epoch sớm hơn. Evaluator là
reference PredCls giữ nguyên: IoU inclusive XYXY, GT object score1, duplicate
GT rows, image-macro per-predicate, denominator cố định **50 lớp**. Báo GC và
NG R/mR@50/100; tail40 xác định từ train sau khi loại 10 predicate phổ biến
nhất. Không bật PE relation NMS riêng, không chấm bằng evaluator GQA 310 lớp.

```bash
myenv/bin/python -m tools.run_vg150_penet --stage summary --scope full
```

Summary chỉ xuất khi đủ 18 kết quả PASS, hash best checkpoint khớp; tính
mean/sample SD (`ddof=1`) qua ba seed. Kiểm sampler trace giống nhau ở prefix
epoch chung dù early stopping cho độ dài khác nhau; kiểm PE initial state.
Không thêm score giả khi thiếu lượt. Kết quả đều là **validation**, chưa phải
test hay điểm paper. Chốt model/protocol rồi mới bàn giao lệnh test riêng.
