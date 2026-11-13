# MagicBrush Task B: joint-image local VLM, version 1

Ngày kiểm tra CPU/metadata: **13/11/2026**. Chưa chạy VLM, chưa có nhãn được
người duyệt xác nhận và chưa hoàn thành bộ ≥300 cặp. Đề
`Project01_Graph_Editing.pdf` yêu cầu ≥300 cặp graph/instruction/edit và kiểm
tra tay ≥20 mẫu. Mô hình thị giác chỉ chạy inference; pipeline này không train.

## Những gì tái sử dụng và phần bổ sung

- `src/editing/operations.py`: `EditGraph`, `Node`, `Edge`, sáu `Operation`,
  `diff_aligned_graphs`, transaction kiểm precondition, cạnh trùng và dangling.
- `src/editing/evaluation.py`: cùng semantics đánh giá exact payload/ID.
- `src/editing/splits.py`, `tools/prepare_magicbrush.py`: giữ nguyên manifest,
  ảnh, grouped splits, hash byte/pixel và source-group của 602 cặp hiện có.
- `tools/edit_model_v1/codec.py`: chuẩn hóa ID nguồn `n001/e001`, ID thêm mới
  `newn001/newe001`, từ chối instance thêm mới đối xứng không phân biệt được.
- `tools/edit_model_v1/data.py`: loader và cổng review có hash bằng chứng từng
  cặp, reviewer, đủ năm trục, tối thiểu 20 kết luận đúng/sai.
- `tools/review_magicbrush.py` và assets: viewer ảnh, graph, operation, xuất/lưu
  CSV thật. Wrapper mới ghi rõ graph joint-VLM và phạm vi partial.

Các pipeline cũ tạo graph từng ảnh độc lập hoặc dùng grounded anchor; chúng
không cung cấp joint correspondence có evidence và cache HTTP theo cặp. Phần
mới nằm hoàn toàn trong `tools/magicbrush_joint_v1/`, config mới và tests mới.
Không sửa `src/`, config/manifest/cache/report/review GQA, VG150, Qwen cũ hoặc
notebook cá nhân. Không cần tải lại MagicBrush.

Luồng thực hiện:

```text
manifest cũ → kiểm split và chọn 20 cặp train → snapshot manifest mới
→ HTTP local: [ảnh nguồn, ảnh đích, instruction] trong MỘT request
→ raw response bất biến → JSON/schema observations v1
→ correspondence bao phủ đầy đủ → graph cùng namespace ID
→ diff hiện có → transaction + chuẩn hóa ID downstream → compilation bất biến
→ viewer/CSV thật → cổng chất lượng → mở rộng train+val bằng lệnh riêng
```

## Checkpoint và mức độ xác minh

Giữ **`inferRouter/Qwen3.6-35B-A3B-NVFP4`**, revision cố định
`0c24da2038a5d9c373112bc59b90bd0e019fe069` của nhánh main; không tự thay model.

| Hạng mục | Bằng chứng và kết luận |
|---|---|
| Truy cập | HF API: public, không gated. HEAD weights HTTP 200. File chính 23,354,237,448 byte; MTP 1,689,283,656 byte. Chỉ tải metadata, chưa tải weights. |
| Vision | `config.json` có vision_config, image token; `processor_config.json` có image processor. Index đọc bằng CPU có 333 tensor vision. |
| Kiến trúc | `Qwen3_5MoeForConditionalGeneration`; config SHA256 `1db31af80394f75764d8af4bbe514c5c243381ede06b7376887a926898633ca1`. |
| Quantization | `compressed-tensors`, NVFP4 nhóm 16, activation 4 bit; không phải checkpoint `modelopt_fp4` của NVIDIA. |
| License/trust | Card khai báo Apache-2.0. Dùng native vLLM, `trust_remote_code=false`; không cài patch từ repo model. |
| Runtime | Môi trường repo **chưa có vLLM**. Đã kiểm mã upstream tag **v0.28.0**: registry, scheme NVFP4 và MoE NVFP4 có triển khai tương ứng. Chưa xác minh kernel trên máy này. |
| Hai ảnh | vLLM hỗ trợ nhiều ảnh cho kiến trúc này; launcher đặt image limit=2. Chưa chạy smoke test hai ảnh thực tế. |
| RTX 5090 32 GB | Card tác giả có số đo một 5090; đây là bằng chứng của tác giả, không phải benchmark của dự án. Cấu hình pilot batch 1/context 16K/eager/giới hạn pixel được kỳ vọng khả thi; VRAM và kernel vẫn phải qua smoke test. |

Nguồn sơ cấp:
[checkpoint và card tại revision đã khóa](https://huggingface.co/inferRouter/Qwen3.6-35B-A3B-NVFP4/tree/0c24da2038a5d9c373112bc59b90bd0e019fe069),
[vLLM registry v0.28.0](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/model_executor/models/registry.py),
[compressed-tensors v0.28.0](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py),
[MoE NVFP4 v0.28.0](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_nvfp4.py),
[multimodal support](https://docs.vllm.ai/en/stable/models/supported_models/),
[structured decoding](https://docs.vllm.ai/en/stable/features/structured_outputs/).

Không dùng recipe `--quantization modelopt_fp4` của checkpoint NVIDIA cho
checkpoint này. Không dùng TurboQuant, FP8-head, MTP hoặc patch riêng của tác
giả trong pilot. Runtime vLLM 0.28.0 khai báo `transformers>=5.5.3`,
`compressed-tensors==0.17.0`; GPU server nên dùng image chính thức cùng phiên
bản để giữ bộ PyTorch/CUDA/kernel tương ứng. Server receipt ghi version thực
của torch, transformers, compressed-tensors, flashinfer và safetensors, hash
source runtime, command, model metadata. Nếu receipt khác, cache từ chối trộn.

`--stage verify-model` kiểm lại metadata/HEAD bằng CPU. `--stage runtime`
kiểm version/registry/source file trong môi trường server, không khởi tạo GPU.
Hai kiểm tra này **không thay thế** smoke test. Nếu smoke lỗi, lưu log rồi dừng;
không tự fallback hoặc mở rộng. Chọn checkpoint 4-bit khác bằng config/output
mới, khóa revision/config SHA và kiểm lại license/processor/runtime. Client,
prompt và compiler độc lập checkpoint; runtime checker hỗ trợ registry cho
Qwen3.5-MoE, Qwen3-VL và InternVL với compressed-tensors. Quantization khác
cần checker version mới; không coi AWQ/ModelOpt là đã xác minh.

## Schema, biên dịch và chính sách chất lượng

`schema.py` công bố `SCHEMA` JSON Schema với version
`magicbrush_joint_observations_v1`, gửi qua `response_format=json_schema` strict.
Validator CPU độc lập không ép kiểu, sửa JSON, bỏ field hoặc suy đoán ID.
`prompt_v1.txt` là prompt đóng băng, nhấn mạnh ảnh quyết định điều gì đã xảy ra.

Observations gồm `instruction_fulfilled`, trạng thái, entity nguồn/đích,
relation nguồn/đích, correspondence entity/relation, `observed_changes`,
`unchanged_entities`, uncertainties, explanation, model_confidence. Mỗi entity,
relation và correspondence có evidence bằng ngôn ngữ; mỗi thay đổi có evidence
cả nguồn và đích và cờ unintended. Evidence này vẫn là nhận xét của VLM, chưa
phải kiểm chứng độc lập. Không yêu cầu bbox để tránh tọa độ bịa; viewer vẫn
hiển thị ảnh và graph.

Mọi node và edge phải xuất hiện đúng một lần ở correspondence. Null chỉ đại
diện thêm/xóa được khẳng định bằng quan sát. Pair replacement giữ ID qua hai
graph; không match chỉ vì ID giống nhau. Correspondence uncertain/missing,
endpoint lỗi, edge trùng, attribute trùng, thay đổi không khai báo và entity
đánh dấu unchanged nhưng thay payload đều bị từ chối compile. Cạnh incident
khi xóa node phải được biểu diễn rõ; transaction không tự xóa cạnh hộ.

Python sinh graph, diff và operations; VLM không được đưa operation JSON cuối.
`Apply(source, ops)==target` kiểm lại trước và sau chuẩn hóa ID. Giới hạn mặc
định 20 node, 60 edge mỗi graph, 12 operation. Attribute đổi thành replace_node;
predicate đổi có correspondence thành replace_edge. Sáu semantics hiện có
được giữ nguyên.

`classification` có ba mức:

- `candidate_accepted`: cấu trúc qua, supported, fulfilled=yes, có ít nhất một
  operation, không uncertainty/unintended. Xuất `status=candidate` để loader
  cũ đọc được; `eligible_for_training` là qua bộ lọc máy, vẫn phải qua gate thật.
- `needs_review`: có graph nhưng fulfillment partial/no/uncertain, no-op,
  uncertainty, unintended; hoặc compile lỗi, thiếu correspondence, chưa infer.
- `rejected`: input/inference lỗi hết budget hoặc nội dung unsupported/style-only.

Unsuccessful/no-op không bị biến thành thao tác theo instruction. Unsupported
style-only có thể có graph không đổi để xem nhưng không đủ điều kiện train.
Confidence do model tự ước lượng, không đặt tên là accuracy và không dùng để
bỏ qua review. Graph này **tập trung chỉnh sửa**, không phải scene graph toàn
cảnh; chưa thể khẳng định đã phát hiện mọi thay đổi ngoài vùng được mô tả.

Summary tách HTTP/completion/schema/compile/transaction/candidate và human
accuracy (`null` nếu chưa audit). `successful_inference_count` yêu cầu một
completion đầy đủ, content không rỗng, finish_reason=stop; không chỉ HTTP 200.
Các tỷ lệ compile/transaction có mẫu số là toàn cohort đã khóa; ops distribution
bao gồm mọi graph compile được, kể cả needs_review. Số người duyệt trong summary
compile luôn là 0 vì stage này không đọc CSV; kết quả review nằm ở audit riêng.

## Cohort và CPU preflight đã thực hiện

Manifest cũ: 602 cặp, **415 train / 97 val / 90 test** nội bộ, tất cả có nguồn
ở official train. Kiểm membership/group/session/hash byte/hash RGB giữa các
split; không đổi split hay mở official dev/test.

Pilot lấy 20 cặp train theo vòng qua sáu nhóm proxy instruction, rồi xếp bằng
SHA256(seed,id), seed 42. Proxy chỉ phục vụ chọn cohort, không tạo nhãn. Các
nhóm: addition 4, removal 4, replacement 3, attribute 3, relationship 3,
difficult/other 3. Nhãn proxy không đảm bảo loại operation thực sự hiện diện.
Đây là mẫu phân tầng tái lập được, không phải ước lượng accuracy không thiên
lệch của toàn bộ MagicBrush. Mẫu khó/style/failed không được thay bằng mẫu dễ.

CPU đã xác nhận đủ 40 file ảnh và hash byte/pixel/dimension. 20 ID sẽ lưu trong
`run.json` và `manifest.json` bản mới khi prepare. Manifest nguồn vẫn nguyên.
Nếu đổi model/prompt/policy/config, dùng output mới và kiểm/commit code trước
khi bàn giao GPU. Thay đổi source/config sau prepare bị chặn bởi provenance.

## Lệnh thực hiện từng stage

Chạy từ repo root. Các biến chỉ rút gọn đường dẫn. Từng lệnh dừng sau stage,
không tự chạy lượt kế tiếp, mở rộng, train hoặc xử lý test.

```bash
cd /workspace/workspace/semantic-graph-editing
MB_JOINT_CFG=configs/level2/magicbrush-joint-qwen36-nvfp4-v1.json
MB_JOINT_OUT=data/processed/level2/magicbrush-joint-qwen36-nvfp4-v1
```

### 1. CPU: metadata, plan, prepare

Code cần được commit trước khi prepare/chạy GPU. Dùng môi trường repo hiện có;
không cài vLLM lên `myenv` đang khóa các thực nghiệm khác.

```bash
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m tools.magicbrush_joint_v1 --config "$MB_JOINT_CFG" --stage verify-model
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m tools.magicbrush_joint_v1 --config "$MB_JOINT_CFG" --stage plan
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m tools.magicbrush_joint_v1 --config "$MB_JOINT_CFG" --stage prepare
myenv/bin/python -m tools.magicbrush_joint_v1 --config "$MB_JOINT_CFG" --stage server-plan
```

`prepare` chỉ tạo snapshot/provenance mới, không tải ảnh hoặc weights. CPU client
đã có requests/Pillow; requirements riêng không cài model. Các imports tái sử
dụng repo còn cần torch/numpy trong môi trường repo, nhưng không gọi CUDA.

### 2. Người dùng: khởi động vLLM trên GPU

Khuyến nghị môi trường server/container riêng có `vllm==0.28.0`. Nếu đã có môi
trường đúng version, chạy từ cùng checkout và đường dẫn repo:

```bash
/path/to/vllm-env/bin/python -m tools.magicbrush_joint_v1 --config "$MB_JOINT_CFG" --stage runtime
CUDA_VISIBLE_DEVICES=0 /path/to/vllm-env/bin/python -u -m tools.magicbrush_joint_v1 --config "$MB_JOINT_CFG" --stage serve-model
```

Có thể dùng Docker Linux với cùng đường dẫn workspace. `--network host` để
server vẫn bind 127.0.0.1; không đổi sang listen mọi interface. Lệnh này có thể
tải image/weights khi **người dùng chủ động chạy**, không được assistant chạy.

```bash
docker run --rm --gpus '"device=0"' --network host --ipc=host \
  -v "$PWD:$PWD" -w "$PWD" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  --entrypoint python vllm/vllm-openai:v0.28.0 \
  -m tools.magicbrush_joint_v1 --config "$MB_JOINT_CFG" --stage serve-model
```

Server checker kiểm metadata/version trước subprocess GPU và lưu `server.json`.
Cần git, requests, Pillow cùng bộ torch/numpy của server image. Launcher giữ
cùng advisory lock `runs/level2/.gqa-kg-training-device/.lock` với các launcher
graph model hiện có. Không quản lý hay kết thúc process GPU khác. Khi thiếu
VRAM/kernel hoặc cổng bận, xem log rồi dừng; không thay recipe trong cache cũ.

Phạm vi: **Qwen3.6-35B-A3B-NVFP4 revision trên, inference frozen**, TP=1,
**batch/max_num_seqs=1, seed=42**, bf16 compute/vision, compressed-tensors
NVFP4, context 16384, prefill 4096, GPU utilization 0.90, eager, MoE Marlin,
1,048,576 pixel tối đa mỗi ảnh. Không train, không MTP, không queue nối tiếp.
Endpoint `http://127.0.0.1:8017/v1`. Giữ server đang chạy cho stage 3/4 rồi tự
dừng server này sau inference; việc serve không tự bắt đầu annotation.

### 3. Người dùng: smoke đúng một cặp (hai ảnh)

Terminal thứ hai dùng `myenv` và cùng biến config:

```bash
myenv/bin/python -u -m tools.magicbrush_joint_v1 --config "$MB_JOINT_CFG" --stage infer --scope pilot --limit 1
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m tools.magicbrush_joint_v1 --config "$MB_JOINT_CFG" --stage compile --scope pilot
```

Gửi cặp đầu tiên của cohort đã khóa. Output raw/schema/graph/ops dưới
`$MB_JOINT_OUT`; compile in đường dẫn compilation và summary: một cặp đã xử lý,
19 pending. Kiểm raw, graph, lý do và log vLLM, không coi JSON hợp lệ là đúng
ảnh. Tối đa **2 request cho mỗi cặp**, timeout đọc HTTP 240 giây, response tối
đa 2 MiB. Completion tối đa 6144 token, temperature=0, top_p=1, thinking=false.

### 4. Người dùng: pilot 20 cặp

```bash
myenv/bin/python -u -m tools.magicbrush_joint_v1 --config "$MB_JOINT_CFG" --stage infer --scope pilot
```

Cùng model, batch 1, seed 42; xử lý tuần tự 20 cặp train, tái sử dụng cặp smoke
đã hoàn tất (thường còn 19). Tối đa 40 HTTP request cho cả cohort nếu cặp nào
cũng cần retry. Không tự mở rộng hoặc train sau lệnh này.

### 5. CPU: compile, báo cáo và viewer

```bash
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m tools.magicbrush_joint_v1 --config "$MB_JOINT_CFG" --stage compile --scope pilot
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m tools.magicbrush_joint_v1 --config "$MB_JOINT_CFG" --stage review-build
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m tools.magicbrush_joint_v1 --config "$MB_JOINT_CFG" --stage review-serve --port 8771
```

Mở/forward `http://127.0.0.1:8771/`. Viewer mới nằm trong
`reviews/magicbrush-joint-human-v1/<compilation-hash>/`. Graph lỗi chưa đủ để
chấm vẫn có ảnh, lý do và annotation/raw provenance trong chi tiết; không biến
chúng thành review đúng. Bấm **Lưu vào reviews/** để lưu submission thật. CSV
chỉ được tạo bởi thao tác người duyệt; pipeline không tự tạo human-approved CSV.

Compilation/summary là file theo hash trong `compilations/` và `reports/` của
output mới. `current-compilation.json` là pointer tiện dụng, được cập nhật
nguyên tử. Có thể dùng `--compilation <đường-dẫn-json>` để khóa một bản; viewer,
CSV và mọi evidence cũ không bị ghi đè khi compile lại.

### 6. CPU: audit CSV thật, sau đó cho phép mở rộng

Thay đường dẫn sau bằng submission thật do chính người dùng vừa lưu:

```bash
MB_JOINT_CSV=reviews/magicbrush-joint-human-v1/<compilation-hash>/submissions/<review-file>.csv
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m tools.magicbrush_joint_v1 --config "$MB_JOINT_CFG" --stage review-audit --review-csv "$MB_JOINT_CSV"
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m tools.magicbrush_joint_v1 --config "$MB_JOINT_CFG" --stage expand --review-csv "$MB_JOINT_CSV"
```

Audit yêu cầu đúng 20 ID đã chọn, reviewer thật, đủ năm trục, **20 kết luận
đúng/sai và ≥90% đúng** (ngưỡng khai báo cho pilot mới, không phải ngưỡng của
đề). Uncertain không tính đúng. Packet/image/graph/operation hash phải khớp.
Nếu còn graph chưa đủ để chấm, hoặc không đạt ngưỡng, dừng; sửa phương pháp ở
version mới và kiểm lại chính cohort, không thay mẫu. `expand` chỉ ghi cohort
và bằng chứng gate; không gọi GPU. Không có `--force` để bỏ qua review.

### 7. Người dùng: mở rộng sau gate; CPU compile sau khi xong

```bash
myenv/bin/python -u -m tools.magicbrush_joint_v1 --config "$MB_JOINT_CFG" --stage infer --scope expanded
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m tools.magicbrush_joint_v1 --config "$MB_JOINT_CFG" --stage compile --scope expanded
```

Phạm vi **512 cặp train+val nội bộ**, không đụng 90 test. Cùng model, batch 1,
seed 42; cache dùng lại đúng 20 pilot, thường còn 492 cặp. Kiểm lại gate trước
inference/compile expanded. Prompt/policy khóa trên train trước khi infer val.
Không tự nối tiếp train, encode CLIP, benchmark hoặc queue nào.

Xuất compilation có `records` tương thích `read_inputs/load_examples`. Khi
chuẩn bị training về sau, dùng **manifest snapshot của output mới**, compilation
expanded, packet pilot gốc và CSV thật trong config train version mới. Không
sửa config train cũ. Per-pair evidence pilot giữ nguyên nên có thể audit lại
với compilation expanded mà không tạo quyết định mới. Đếm mẫu sau lọc: phải
≥300 cặp dùng được, train/val không rỗng và mọi cổng training hiện có đều qua;
512 ảnh thô không đảm bảo đạt yêu cầu. Pipeline này chưa kích hoạt training.

## Resumability và cấu trúc output

```text
run.json                         config, model/revision, commit, source/prompt/schema hashes
manifest.json                    snapshot mới, split/group/hash và pilot IDs
server.json, inference-backend.json  runtime + model receipt đã khóa
samples/<sample-id-hash>/
  input.json                     ID, instruction, image paths/hashes, split/group, provenance
  attempt-001.started.json        intent gửi request đã ghi bền vững
  attempt-001.raw.json            HTTP status/body/error nguyên bản của transport
  attempt-001.result.json         JSON/schema/failure + hash raw
  annotation.json                observations đã qua schema
  final.json                     completed/input_error/inference_failed
compilations/<hash>.json          graph, ops, validation, reason, annotation từng cặp
reports/<hash>.json               summary của đúng compilation
events.jsonl                     append + fsync, log chẩn đoán
expansion.json                    chỉ có sau khi gate CSV thật đạt
```

File evidence được ghi nguyên tử, không overwrite. JSONL chỉ là log phụ: đứt
dòng cuối không làm mất raw/annotation. Crash sau lưu raw sẽ parse lại, không
infer lại. Crash sau started nhưng chưa có raw ghi `interrupted...` và tiêu
thụ một lượt trong budget, vì server có thể đã hoàn tất request. Hết hai lượt
thì giữ failure, không xóa hoặc retry vô hạn. Cache đã final được bỏ qua; compile
kiểm lại hash raw/annotation/input. Thay đổi code/model/config/image/evidence
làm pipeline dừng, cần version mới. Không có silent schema repair.

## Kiểm tra CPU

```bash
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m pytest tests/editing -q
myenv/bin/python -m black --check tools/magicbrush_joint_v1 tests/editing/test_magicbrush_joint_v1.py
myenv/bin/python -m isort --check-only tools/magicbrush_joint_v1 tests/editing/test_magicbrush_joint_v1.py
myenv/bin/python -m ruff check tools/magicbrush_joint_v1 tests/editing/test_magicbrush_joint_v1.py
```

Fixtures và client giả lập không tải weights, không cần GPU. Chúng kiểm
addition/removal/replacement/attribute/relation, correspondence thiếu/mơ hồ,
operation xung đột/payload thiếu, style/failed/no-op, HTTP lỗi/JSON lỗi,
missing image, crash/resume, split leakage, hash sửa, viewer và loader.
Quyết định review giả chỉ nằm trong thư mục tạm pytest; không tạo review thật.
Dùng mẫu [báo cáo pilot](LEVEL2_MAGICBRUSH_JOINT_VLM_V1_PILOT_TEMPLATE.md) khi
chưa chạy; compilation sinh summary thật sau inference.
