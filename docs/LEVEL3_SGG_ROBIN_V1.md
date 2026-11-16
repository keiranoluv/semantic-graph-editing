# Thử nghiệm SGG pretrained: ROBIN + SAM, cùng 5 ảnh Level 3

> Tài liệu thực nghiệm lịch sử. Launcher/giao diện pilot tương ứng đã nghỉ dùng; các kết quả ảnh/graph được giữ. Dùng [luồng hiện tại](LEVEL3_CLEANUP.md); code cũ có thể đọc lại tại commit `1d2d876`.

**Cập nhật cho RTX 5090:** môi trường Torch 2.5.1/CUDA 12.4 bên dưới không hỗ trợ `sm_120`. Dùng [runtime CUDA 12.8 v2 và launcher mới](LEVEL3_SGG_CUDA128_V2.md) trên máy hiện tại. Protocol/checkpoint/5 ảnh giữ nguyên; các kết quả CPU trong tài liệu này là bằng chứng của lần triển khai ban đầu.

Mục đích là kiểm tra một phương án thay thế bộ trích graph RelTR + CLIP hiện tại, trước khi dùng graph để chấm ảnh chỉnh sửa. Chưa có kết quả chất lượng ROBIN trên cả 5 ảnh: assistant chỉ chạy kiểm thử và smoke CPU; người dùng chạy inference GPU bằng lệnh bên dưới. Không huấn luyện mô hình thị giác, không chạy Task B, không thay dữ liệu hoặc pipeline cũ.

## Nguồn chính thức và lựa chọn mô hình

| Thành phần | Nguồn đã kiểm tra | Bản khóa |
| --- | --- | --- |
| ROBIN, Synthetic Visual Genome, CVPR 2025 | [Project](https://synthetic-visual-genome.github.io/), [SyntheticVG](https://github.com/jamespark3922/SyntheticVG), [checkpoint Stage 2](https://huggingface.co/jamepark3922/robin-qwen2.5-3b) | Code `29146b9d81333e0af039c617b22ccf618031c07c`; HF `cfa715637539dcda3464c69f13cbb00718fbeba8` |
| SAM ViT-H | [segment-anything](https://github.com/facebookresearch/segment-anything), [checkpoint chính thức](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) | Code `dca509fe793f601edb92606367a655c15ac00fdf`; SHA-256 `a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e` |
| PGSG, From Pixels to Graphs, CVPR 2024 | [Project](https://plus-sist.github.io/publication/year2024/conf_cvpr2024_pgsg/), [code](https://github.com/SHTUPLUS/Pix2Grp_CVPR2024), [checkpoints](https://huggingface.co/rj979797/PGSG-CVPR2024) | Code `cff742070366a6e702a08b71130eb75fcc4d5f7a`; HF `2ea7c8c33dee45b09c7cd4e83ed8c8f68f6cac10` |

Toàn bộ URL, revision, kích thước và SHA-256 ROBIN/SAM nằm trong `configs/level3/sgg-assets-v1.json`. Tổng checkpoint và tokenizer khoảng 11,2 GB; môi trường Python cần dung lượng riêng. Checkpoint ROBIN có hai shard safetensors và ConvNeXt `open_clip_pytorch_model.bin`. Link README cũ có hậu tố `-sg-stage2` không truy cập công khai được khi kiểm tra; adapter dùng repo public ở bảng trên, có model card ghi Stage 2.

ROBIN nhận **ảnh và các vùng/mask đầu vào** rồi sinh mô tả vùng và quan hệ. Đây không phải mô hình tự dự đoán bbox từ ảnh trong adapter này. SAM sinh vùng tự động, không dùng bbox/nhãn đúng từ người gán nhãn. Cấu hình giữ tối đa 32 vùng, lọc diện tích mask từ 0,2% đến 85% ảnh, sắp theo diện tích giảm dần, stability giảm dần, rồi chỉ số gốc. Có thể bỏ sót vật nhỏ hoặc chọn vùng chồng lấn; mọi vùng bị loại và lý do được lưu để kiểm tra.

PGSG đáng thử ở vòng tiếp theo vì sinh chuỗi đối tượng/quan hệ và tọa độ trong cùng mô hình, thay vì phụ thuộc SAM. Tuy nhiên code chính thức `lavis/models/blip_models/blip_rel_det_pgsg.py` gắn `generate` với `vocab2category`, category dictionary và xử lý target của evaluator. YAML có đường dẫn checkpoint/category tuyệt đối; requirements có đường dẫn `file://` của tác giả, dựa trên Torch 2.0 / torchvision 0.15 / Transformers 4.29.2. Không thể chỉ tải `.pth` rồi coi là backend dùng được cho ảnh bất kỳ. Khi tích hợp cần môi trường riêng, config SGDet (`sgcls_on=False`, `predcls_on=False`), category mapping đúng checkpoint, và chứng minh không đưa ground-truth box/label vào inference. Vòng này **chỉ khảo sát PGSG**, khóa metadata hai checkpoint `vg_ov_sgg.pth` và `psg_ov_sgg.pth`; chưa tải trọng số, viết backend hoặc đo kết quả PGSG.

## Protocol cùng 5 ảnh

`configs/level3/sgg-compare-v1.json` khóa 5 candidate ID và SHA-256 ảnh, hash manifest/tasks của pilot30, cùng hash candidates/receipt baseline. Chọn 3 sidewalk + 2 travel bằng thứ tự SHA-256 của `42|dataset|image_id`, không chọn theo dự đoán. Đây là tập phát triển nhỏ, không đại diện cho toàn bộ benchmark.

| Candidate ID | Node RelTR | Attribute CLIP | Cạnh có hướng | Node cô lập |
| --- | ---: | ---: | ---: | ---: |
| `sidewalk-b1bb633496439f` | 24 | 7 | 1 | 22 |
| `sidewalk-32a2f39d76e383` | 31 | 0 | 4 | 23 |
| `sidewalk-fce6aef5958616` | 40 | 15 | 11 | 26 |
| `travel-c779ea8e2f5b35` | 17 | 2 | 9 | 8 |
| `travel-685ed0a42dc65a` | 25 | 3 | 11 | 14 |
| Tổng | 137 | 27 | 36 | 93 |

Baseline được đọc từ kết quả inference thật `runs/level3/image-graph-v1/pilot30/`, kiểm tra receipt và hash rồi lấy đúng 5 ảnh. Không chạy lại RelTR, không đổi threshold để làm đẹp so sánh. Các số trên chỉ mô tả graph; không suy ra precision, recall hoặc chất lượng từ số node/cạnh. Ô ROBIN chưa chạy là `not_run` và thống kê `null`, không ghi 0 hay tạo kết quả giả.

## Chuẩn hóa và giới hạn representation

| Trường | RelTR + CLIP hiện tại | ROBIN + SAM v1 |
| --- | --- | --- |
| Object | Nhãn category của detector | Mô tả vùng nguyên văn làm `Node.label`; chưa phải category ontology |
| Attribute | CLIP trên crop, vocabulary giới hạn | Chỉ tách modifier rõ ở đầu mô tả; giữ nguyên mô tả đầy đủ để kiểm tra |
| Directed relation | Triple RelTR sau threshold | `regionS: regionO predicate` → cạnh S → O, giữ predicate nguyên văn |
| Grounding | Bbox detector | Bbox + mask RLE của **vùng SAM đầu vào** |
| Confidence | Điểm của detector/CLIP, chưa calibration | Confidence ngữ nghĩa `null`; predicted IoU/stability SAM lưu riêng |

Ví dụ `a small red car` có attributes `red`, `small`; `a man wearing a red shirt` không nhận attribute `red`. ROBIN không cung cấp một attribute head riêng; projection này không có độ phủ đầy đủ và chưa hiểu quan hệ sở hữu phức tạp. Không dùng regex để tự đổi cả mô tả thành nhãn category hoặc suy diễn các vật chưa có vùng.

Parser giữ đúng region ID, kể cả khi thứ tự mô tả không tăng dần. Thiếu đối tượng, tham chiếu vùng ngoài tập, văn bản sai format hoặc chạm giới hạn token đều thành `parse_error`, không âm thầm tạo graph rỗng. Triple trùng được gộp theo ràng buộc `EditGraph`, nhưng các lần lặp vẫn nằm trong evidence. Không thêm cạnh đảo hoặc tự sửa predicate. `source_graph` tương thích **schema** `EditGraph`; bbox/confidence/candidate ID/provenance nằm trong sidecar và metadata ảnh giữ nguyên.

Điều này **không chứng minh tương thích model EMNLP 2020 Task B**: category, attribute representation, vocabulary, tokenization, graph serialization và ngân sách node/cạnh có thể khác. Không nạp checkpoint Task B, thay codec, ép cắt graph hoặc báo Task B compatible. Cần một audit riêng với chính checkpoint/codec nếu sau này muốn nối model đó. So sánh SGG trong vòng này độc lập với Task B.

Có hai xử lý tương thích được ghi rõ:

- `aligned_square_pad_v1`: pad ảnh RGB, mask và bbox prompt với cùng offset trước khi chuẩn hóa tọa độ 0–1000. Demo upstream pad ảnh nhưng giữ mask/bbox gốc; dùng nguyên cách đó có thể lệch grounding trên ảnh không vuông. Đây là thay đổi preprocessing có chủ đích, **không phải tái lập nguyên xi điểm của paper**. Mask/bbox gốc vẫn được giữ.
- Transformers 4.43.3 đổi `gamma` thành `weight` trong loader, làm mất 36 tham số layer-scale ConvNeXt. Adapter chỉ chấp nhận đúng mẫu thiếu/thừa này, đọc tensor `gamma` nguyên gốc từ safetensors, copy và kiểm tra giá trị bằng nhau sau đổi dtype. Mọi thiếu/thừa khác đều dừng. `model-loading.json` ghi warning gốc, danh sách phục hồi và `unresolved_loading_keys=[]`; không sửa file checkpoint hay source upstream. Đường dẫn vision tower tuyệt đối của tác giả cũng chỉ được thay trong config ở bộ nhớ.

## Chuẩn bị và kiểm tra CPU

Chạy từ root repo. Dùng Python 3.12 và môi trường **mới**, không cài đè `myenv` hoặc môi trường Level 1/2. Cache và venv đã được tạo trong workspace hiện tại; máy mới dùng:

```bash
python3.12 -m venv data/cache/level3-sgg-v1/venv
data/cache/level3-sgg-v1/venv/bin/python -m pip install -r requirements-level3-sgg-robin-v1.txt
CUDA_VISIBLE_DEVICES='' data/cache/level3-sgg-v1/venv/bin/python -m tools.level3_sgg_compare_v1 prepare
CUDA_VISIBLE_DEVICES='' data/cache/level3-sgg-v1/venv/bin/python -m tools.level3_sgg_compare_v1 preflight
CUDA_VISIBLE_DEVICES='' data/cache/level3-sgg-v1/venv/bin/python -m pytest -q tests/level3/test_sgg_compare_v1.py
```

`prepare` chỉ clone code/tải trọng số đã khóa và kiểm tra SHA-256, không inference. Code upstream phải đúng commit, không bị sửa hoặc thêm Python file. `preflight` kiểm tra runtime, checkpoint và toàn bộ 30 ảnh nguồn; không nạp model lên GPU. Sau prepare, launcher inference dùng offline mode. Không upload ảnh tới dịch vụ bên ngoài.

Smoke thật bằng CPU, nếu cần lặp lại:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  data/cache/level3-sgg-v1/venv/bin/python -m tools.level3_sgg_compare_v1 smoke \
  --output runs/level3/sgg-compare-v1/cpu-smoke-user-v1
```

Smoke nạp SAM ViT-H với strict state dict trên CPU, giải phóng, rồi nạp ROBIN và sinh **8 token** từ ảnh thật đầu tiên với mask fixture toàn ảnh. Không chạy SAM segmentation trên ảnh trong smoke, không xuất graph hoàn chỉnh, không chấm chất lượng. `torch_cuda_initialized=false` được lưu. Cần RAM cho trọng số; không đánh đồng CPU smoke với xác minh throughput/VRAM GPU.

Tests pipeline cũ chạy trong môi trường cũ:

```bash
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m pytest -q \
  tests/level3/test_image_graph_v1.py \
  tests/level3/test_vietnam_benchmark_v1.py \
  tests/level3/test_vietnam_benchmark_v2.py
```

## Lệnh inference thật — người dùng tự chạy

```bash
bash scripts/run_level3_sgg_robin_v1.sh
```

Phạm vi: **SAM ViT-H → ROBIN Qwen2.5-3B Stage 2**, đúng 5 ảnh trong config. Batch ảnh **1**, batch điểm SAM **32**, seed **42**, BF16 cho ROBIN, greedy decoding một beam, tối đa **2048 token mới/ảnh**. Chạy SAM cho 5 ảnh, giải phóng SAM rồi chạy ROBIN cho 5 ảnh. Một model ở GPU tại mỗi thời điểm; **không có lượt tiếp theo trong hàng đợi**, không train hoặc chạy editor. `flock` ngăn hai launcher này chạy đồng thời; không quản lý những job GPU khác ngoài launcher. Chưa đo VRAM hoặc thời gian GPU.

Output mặc định: `runs/level3/sgg-compare-v1/robin-five-v1/`. Thư mục phải chưa tồn tại. Chạy lại bằng một output version mới:

```bash
OUTPUT=runs/level3/sgg-compare-v1/robin-five-v2 \
  bash scripts/run_level3_sgg_robin_v1.sh
```

Không tự giảm precision, giới hạn vùng, retry hoặc đổi model khi lỗi. Nếu OOM hoặc graph bị cắt do token budget, giữ output lỗi để chẩn đoán; thay đổi cấu hình phải thành một protocol version mới. CLI kết thúc mã 2 khi có `parse_error`/`no_regions`; receipt vẫn liệt kê tình trạng từng ảnh. Lỗi runtime dừng và không tạo receipt hoàn tất. Không tái sử dụng thư mục dang dở.

Các artifact chính:

- `run.json`, `model-loading.json`, `receipt.json`: config, code/revision/hash, runtime, thứ tự chạy và tính toàn vẹn output.
- `regions/<candidate>.json`: toàn bộ SAM proposal, mask RLE, bbox, score, vùng chọn/bỏ và lý do.
- `native/<candidate>.json`: nguyên văn ROBIN, token IDs, prompt, geometry, token budget/truncation trước khi parse.
- `robin.json`: 5 record, graph + sidecar hoặc failure rõ ràng.
- `comparison.json`, `comparison.html`: cùng ảnh, hai model, thống kê và evidence.
- `quality-review-template.json`: 10 hàng ảnh/model để người kiểm duyệt chấm; không tự điền nhãn.

## Xem ảnh và chấm chất lượng

Mở `comparison.html` bằng trình duyệt ngoài; file tự chứa bản xem trước ảnh và không cần server. Dropdown cho xem ảnh sạch, từng bbox hoặc tất cả bbox. Ảnh nhúng được thu nhỏ tối đa 1200px để xem; ảnh gốc nguyên vẹn nằm ở `image.image_path` trong locked dataset. Xem nguyên bản khi cần đánh giá vật nhỏ. Có node/attribute, danh sách cạnh với mũi tên có hướng và native output/provenance.

Baseline hiện có thể xem ngay tại `runs/level3/sgg-compare-v1/baseline-five-v1/comparison.html`; cột ROBIN ghi chưa chạy. Tạo lại baseline ở một đường dẫn mới bằng:

```bash
CUDA_VISIBLE_DEVICES='' data/cache/level3-sgg-v1/venv/bin/python \
  -m tools.level3_sgg_compare_v1 baseline \
  --output runs/level3/sgg-compare-v1/baseline-five-user-v1
```

Sau lượt thật, copy `quality-review-template.json` ra ngoài bundle, điền `status=reviewed`, `reviewer`, `notes` và năm điểm: object, attribute, directed relation, grounding, overall. Thang 0 = phần lớn sai/không dùng được; 1 = lẫn đúng sai/cần sửa nhiều; 2 = phần lớn đúng/dùng được. Attribute có thể `null` nếu không đánh giá được; ghi lý do. Ghi rõ vật/quan hệ quan trọng bị bỏ sót, sai endpoint, đảo hướng, nhận nhầm thuộc tính hoặc bbox phủ sai instance. Chấm graph tự động **trước khi sửa tay**; sửa tay không được dùng làm điểm chất lượng của extractor.

```bash
CUDA_VISIBLE_DEVICES='' data/cache/level3-sgg-v1/venv/bin/python \
  -m tools.level3_sgg_compare_v1 quality \
  --bundle runs/level3/sgg-compare-v1/robin-five-v1 \
  --review reviews/level3/sgg-five-review-v1.json \
  --output runs/level3/sgg-compare-v1/quality-summary-v1.json
```

CLI kiểm tra hash bundle, từng record và người chấm; không cho chấm dự đoán chưa có/lỗi. Có thể nhập bản chấm dở nhưng `complete_paired_review=false`; điểm là ordinal diagnostics trên 5 ảnh, không báo benchmark accuracy. Chưa có gold graph đối sánh nên không tính precision/recall giả. Failure cần được báo riêng bên cạnh điểm của graph parse thành công.

## Liên hệ yêu cầu đồ án

Theo `Project01_Graph_Editing.pdf`, Level 3 còn cần ít nhất 100 ảnh hợp lệ, 150–200 yêu cầu chỉnh sửa, editor pretrained, graph trước/sau, điểm thay đổi được yêu cầu và bảo toàn phần không liên quan, nhãn tay đạt/không đạt, tương quan với nhãn tay, đối chiếu CLIPScore/LPIPS và tối thiểu 10 ca bất đồng. Thử nghiệm 5 ảnh này chỉ chọn/kiểm tra extractor. Nếu graph sai hoặc bỏ sót vật liên quan, điểm graph sau đó không có cơ sở đáng tin; không lấy graph đẹp hơn hoặc nhiều cạnh hơn làm bằng chứng đã hoàn thành Level 3.

Quyết định sau pilot dựa trên cùng ảnh và năm tiêu chí, báo rõ failure/coverage, chi phí và sai sót của vùng SAM. Nếu ROBIN vẫn không đủ tốt, dùng kết quả này để xác định giới hạn của graph metric và chuẩn bị vòng PGSG độc lập; không suy đoán mô hình thắng trước khi có output và review.
