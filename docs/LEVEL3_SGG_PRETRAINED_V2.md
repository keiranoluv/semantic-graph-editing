# Pilot pretrained SGG v2: REACT++ → RelateAnything

Phạm vi: chọn extractor cho Task 3; chỉ inference mô hình thị giác pretrained.
Không thay Level 1/2, Task B, `src/`, dataset/config đã khóa hoặc baseline ROBIN-SAM.
Branch `feat/level3-vietnam-benchmark-v2`, không push. Mã mới nằm ở
`tools/level3_sgg_pretrained_v2/`; tất cả output là thư mục mới, không ghi đè.

## Nguồn chính thức đã kiểm tra

- [REACT++ repo và quickstart](https://github.com/Maelic/SGG-Benchmark),
  [model zoo](https://github.com/Maelic/SGG-Benchmark/blob/main/docs/MODEL_ZOO.md),
  [standalone inference](https://github.com/Maelic/SGG-Benchmark/blob/main/demo/standalone_onnx_demo.py),
  [checkpoint PSG](https://huggingface.co/maelic/REACTPlusPlus_PSG).
  Chọn YOLO12m ONNX, detector + relation head trong cùng graph; không cấp GT box/label.
  API standalone có object labels bắt đầu từ 1, relation labels từ 0;
  relation array gồm subject index, object index, predicate, triplet score, relation score.
- [RelateAnything repo](https://github.com/Maelic/RelateAnything),
  [deployment](https://github.com/Maelic/RelateAnything/blob/main/deploy/README.md),
  [score contract](https://github.com/Maelic/RelateAnything/blob/main/relsgg/scoring.py),
  [checkpoint ViT-S/16+](https://huggingface.co/maelic/relsgg-vits16plus).
  Dùng ONNX raw logits + predicate bank + calibration chính thức, preprocessing và decode upstream.
  Tránh ép runtime PyTorch của tác giả (2.13/cu130, Transformers 5.14) vào môi trường ROBIN.
- [OWLv2 model card và inference](https://huggingface.co/google/owlv2-base-patch16-ensemble).
  Detector pretrained open vocabulary thay cho YOLO26 demo. Đây là cấu hình pipeline
  riêng, **không phải tái lập điểm paper**. OWLv2 nhận cùng vocabulary text cho mọi ảnh;
  không nhận yêu cầu chỉnh sửa hoặc nhãn đúng. RelateAnything chỉ nhận pixel + box,
  confidence detector dùng xếp hạng, không đưa object label vào relation head.
- [ONNX Runtime CUDA provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).
  Runtime riêng Python 3.12, Torch 2.7.1/cu128 + torchvision 0.22.1,
  ORT GPU 1.23.2, Transformers 4.57.1. Đây là deployment ONNX, không nạp model torch DINOv3.

Revision Git, HF revision, kích thước và SHA-256 từng checkpoint/config/tokenizer
nằm trong `configs/level3/sgg-pretrained-assets-v2.json`. Code upstream không sửa;
preflight từ chối repo sai revision/dirty và checkpoint sai hash. License theo
README tại revision đã khóa: RelateAnything AGPL-3.0 + NOTICE, weights có điều kiện
DINOv3; OWLv2 Apache-2.0. Không commit trọng số hoặc code upstream.

## Protocol cố định và representation

`configs/level3/sgg-pretrained-v2.json` tham chiếu hash config cohort ROBIN và
receipt `robin-five-cu128-v2`. Chính xác 5 ảnh theo thứ tự:

1. `sidewalk-b1bb633496439f`
2. `sidewalk-32a2f39d76e383`
3. `sidewalk-fce6aef5958616`
4. `travel-c779ea8e2f5b35`
5. `travel-685ed0a42dc65a`

Kiểm tra hash ảnh, metadata, manifest/tasks, baseline trước chạy. Không chọn lại ảnh.
ROBIN baseline có 4 `ok`, 1 `parse_error`; không sửa parser/kết quả baseline để nâng điểm.

| Pipeline | Object / box | Relation | Attribute |
| --- | --- | --- | --- |
| ROBIN-SAM cũ | Mô tả vùng + SAM input box | Native directed text | Projection modifier có giới hạn |
| REACT++ PSG YOLO12m | Detector embedded; letterbox 640 | PSG vocabulary đóng | Không hỗ trợ |
| OWLv2 + RelateAnything ViT-S+ | Vocabulary text mở, bbox detector | Toàn bộ released ONNX predicate bank | Không hỗ trợ |

Open vocabulary không có nghĩa tìm mọi object/predicate tự do: OWLv2 chỉ tìm
các tên được cấp; bản ONNX RelateAnything chỉ chọn trong bank đã phát hành.
Muốn chuỗi mới ngoài bank cần text encoder/bank version mới, không âm thầm đổi protocol.
Nhãn object: COCO + danh từ phổ thông đường phố/công trình, khóa trước xem output.
Không dùng nhãn source/target để cấp prompt riêng. Mọi graph ID chỉ có nghĩa
trong một ảnh, không tự suy ra correspondence qua ảnh.

REACT dùng threshold triplet 0.05, object 0.4; giữ box confidence thấp nếu là
endpoint của relation như demo upstream. Khôi phục letterbox về pixel gốc,
clip box tại biên ảnh có lưu before/after. Không đảo cạnh hoặc tự bổ sung quan hệ.
OWLv2 threshold 0.1, NMS class-agnostic IoU 0.5 để tránh synonym trùng instance,
giữ tối đa 32 box. RelateAnything resize vuông 448 theo upstream, threshold 0.4,
top 100, tối đa một predicate cho mỗi ordered pair; dùng calibration trong ONNX
sidecar, pair weight 1.0. Relation score và score xếp hạng nhân confidence hai
box được lưu riêng. Các ngưỡng khác nhau có ý nghĩa khác nhau, không so confidence
trực tiếp giữa mô hình và không tinh chỉnh theo 5 ảnh.

`source_graph` là `EditGraph`; bbox/confidence/evidence nằm trong `grounding`.
Không tự tạo attribute. Duplicate triple được gộp nhưng giữ native row references.
Sai shape/index/box/score là lỗi parse, không tạo graph rỗng giả.
Giữ `.npz` raw logits/boxes/relations trước lọc, JSON sau lọc, model metadata,
provenance hash code/config/checkpoint/runtime. Receipt kiểm tra tính toàn vẹn.

## Cài đặt và CPU

Môi trường mới độc lập, không thay `myenv` hoặc runtime ROBIN:

```bash
python3.12 -m venv data/cache/level3-sgg-pretrained-v2/venv
data/cache/level3-sgg-pretrained-v2/venv/bin/pip install -r requirements-level3-sgg-pretrained-v2.txt
CUDA_VISIBLE_DEVICES='' data/cache/level3-sgg-pretrained-v2/venv/bin/python -m tools.level3_sgg_pretrained_v2 prepare
CUDA_VISIBLE_DEVICES='' data/cache/level3-sgg-pretrained-v2/venv/bin/python -m tools.level3_sgg_pretrained_v2 preflight
CUDA_VISIBLE_DEVICES='' data/cache/level3-sgg-pretrained-v2/venv/bin/python -m pytest -q tests/level3/test_sgg_pretrained_v2.py
```

`prepare` chỉ tải đúng revision/hash. Nếu download dang dở, giữ file lỗi để audit;
không tự ghi đè hoặc xóa. Chuẩn bị cache mới qua `--cache` nếu cần.

CPU smoke thật, **một ảnh đầu**, không W&B, không hàng đợi:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
 data/cache/level3-sgg-pretrained-v2/venv/bin/python -m tools.level3_sgg_pretrained_v2 \
 smoke --model react --device cpu --output runs/level3/sgg-pretrained-v2/react-cpu-user-v1
# Chạy lệnh này sau khi REACT kết thúc:
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
 data/cache/level3-sgg-pretrained-v2/venv/bin/python -m tools.level3_sgg_pretrained_v2 \
 smoke --model relate --device cpu --output runs/level3/sgg-pretrained-v2/relate-cpu-user-v1
```

## GPU smoke nhỏ và launcher đủ 5 ảnh

GPU smoke dùng cùng threshold/vocabulary/graph budget, batch 1, seed 42,
chỉ ảnh đầu; REACT trước, RelateAnything sau, mỗi lệnh độc lập:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
 data/cache/level3-sgg-pretrained-v2/venv/bin/python -m tools.level3_sgg_pretrained_v2 \
 smoke --model react --device cuda --output runs/level3/sgg-pretrained-v2/react-gpu-user-v1
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
 data/cache/level3-sgg-pretrained-v2/venv/bin/python -m tools.level3_sgg_pretrained_v2 \
 smoke --model relate --device cuda --output runs/level3/sgg-pretrained-v2/relate-gpu-user-v1
```

Session yêu cầu CUDA provider và profile phải có CUDA kernel thực thi;
CPU shape/control ops có thể vẫn xuất hiện. Không chấp nhận toàn bộ inference
fallback CPU rồi báo GPU pass. OWLv2 ghi device và chạy inference thật.

**Hai lệnh dưới đây để người dùng tự chạy, assistant không chạy đủ 5 ảnh.**
Cần W&B đã đăng nhập (`wandb login` hoặc `WANDB_API_KEY`); full pilot bắt buộc
online, không âm thầm tắt tracking. Project `level3-sgg-pretrained-v2`.
Chỉ log config/provenance, parse status, thời gian; không upload ảnh/raw tensors.

```bash
bash scripts/run_level3_sgg_pretrained_v2.sh react
# Đợi REACT kết thúc rồi chạy riêng:
bash scripts/run_level3_sgg_pretrained_v2.sh relate
```

REACT++ PSG YOLO12m → output `runs/level3/sgg-pretrained-v2/react-five-v1/`.
OWLv2 ViT-B/16 ensemble → giải phóng GPU → RelateAnything ViT-S/16+ → output
`runs/level3/sgg-pretrained-v2/relate-five-v1/`. Cả hai batch **1**, seed **42**,
FP32, không train/fine-tune, **không tự chạy model/lượt kế tiếp**. CLI dùng cùng
`inference.lock` với ROBIN. Đổi `OUTPUT` sang version mới nếu đường dẫn đã có.
Bộ nhớ/latency smoke không đại diện toàn bộ tập; elapsed ở record RelateAnything
là relation stage, detector nằm ở stage riêng, không dùng để so tốc độ end-to-end.

## Xem cùng 3 mô hình và duyệt tay

```bash
data/cache/level3-sgg-pretrained-v2/venv/bin/python -m tools.level3_sgg_pretrained_v2 compare \
 --react-run runs/level3/sgg-pretrained-v2/react-five-v1 \
 --relate-run runs/level3/sgg-pretrained-v2/relate-five-v1 \
 --output runs/level3/sgg-pretrained-v2/comparison-five-v1
```

Mở `comparison.html`: ảnh nhúng, chọn bbox, node-link graph có hướng, danh sách
đỉnh/cạnh, confidence và native/provenance. Smoke comparison vẫn có 5 hàng,
4 ảnh chưa chạy phải hiển thị `not_run`, không dùng dự đoán CPU lấp vào GPU run.
Có thể dùng `compare` với hai thư mục smoke để xem một ảnh thật cạnh baseline.

Sao chép `review-template.json` thành file submission mới và điền từng object/relation:
`correct`, `grounded` (object), `hallucinated`, ID tham chiếu cùng reviewer/notes.
Object đúng cần đúng instance/ý nghĩa và grounding IoU ≥0.5 theo box người duyệt;
relation đúng cần hai endpoint đúng và predicate/chiều đúng. Người duyệt đối chiếu
mô tả ROBIN theo ý nghĩa, không dùng exact string matching bất lợi cho ROBIN.
Không lấy output của mô hình khác làm GT. Duplicate prediction không được cùng
claim một reference ID. Hallucination là nội dung không được ảnh hỗ trợ,
không đánh đồng mọi FP hoặc missed object với hallucination.

Liệt kê `missing_objects` / `missing_relations` bằng ID tham chiếu duy nhất;
chỉ đặt `inventory_complete=true` sau khi người duyệt xác nhận inventory đầy đủ.
Với review chưa đầy đủ, PR/F1 để null. Có review đầy đủ: TP=prediction đúng,
FP=prediction sai, FN=phần thiếu do người duyệt liệt kê; micro P/R/F1 chỉ trên
những ảnh đã hoàn tất review, kèm mẫu số. Zero denominator cho P/R/F1 là null.
Grounding và hallucination báo tỷ lệ trên item đã chấm, kèm coverage; không
ngầm coi unknown là false. Parse success tính `ok / attempted`, `not_run` không
nằm mẫu số; parse failure phải báo riêng, không bị che bởi F1 trên subset.

```bash
data/cache/level3-sgg-pretrained-v2/venv/bin/python -m tools.level3_sgg_pretrained_v2 evaluate \
 --comparison runs/level3/sgg-pretrained-v2/comparison-five-v1/comparison.json \
 --review PATH_TO_REVIEW_SUBMISSION.json \
 --output runs/level3/sgg-pretrained-v2/reviewed-metrics-v1.json
```

Review ràng buộc hash comparison/record, từ chối thiếu item, stale hash, người
chấm trống, duplicate TP, relation đúng với endpoint sai. Đây là đánh giá tay
có kiểm chứng cấu trúc, không tự sinh graph GT hoặc tự nhận đã có F1 benchmark.

## Liên hệ yêu cầu đề và giới hạn kết luận

Đề Task 3 đòi graph thay đổi đúng yêu cầu **và** bảo toàn phần không liên quan,
Spearman với nhãn tay, đối chứng CLIPScore/LPIPS và ít nhất 10 ca bất đồng.
Năm ảnh nguồn này chưa có paired target evaluation: không thể kết luận extractor
ổn định source/target hoặc tin cậy cho Graph Consistency Scoring từ smoke.
Bước tiếp cần manual inventory cùng quy tắc cho 5 ảnh, sau đó paired source/target
với instance alignment, recall phần liên quan edit, preservation false alarms,
độ ổn định nhãn/predicate giữa hai ảnh. Không dùng node/edge count chọn model.
Cả hai phương pháp mới thiếu attribute head: edit màu/chất liệu chưa chấm được
bằng graph này; cần module attribute pretrained riêng và audit độc lập nếu mở rộng.

## Kiểm tra trước GPU

- CPU suite mới: 24 tests đạt. ROBIN regression: 37 tests đạt ở runtime ROBIN;
  image graph + benchmark v1/v2: 57 tests đạt ở `myenv`.
- REACT CPU inference thật: `react-cpu-smoke-v1`, ảnh đầu, parse thành công.
- RelateAnything CPU lần đầu thiếu scipy (`relate-cpu-smoke-v1`); đã pin scipy 1.16.3.
  `relate-cpu-smoke-v2` chạy detector và relation inference thật thành công.
- Chạy cả regression cũ trong venv mới ban đầu có 2 lỗi do thiếu CLIP/scipy;
  kiểm tra lại trong đúng môi trường cũ đạt, không sửa pipeline cũ.
- GPU smoke và kết luận thực tế sẽ được ghi bổ sung sau commit code/CPU checks.

## Kết quả thực tế sau commit 9a0cd80

Đã chạy tuần tự GPU smoke thật **một ảnh** cho mỗi pipeline trên RTX 5090,
không chạy đủ 5 ảnh. Evidence:

- `runs/level3/sgg-pretrained-v2/react-gpu-smoke-v1/`: parse đạt;
  profile 1.162 CUDA kernel events và 354 CPU events.
- `runs/level3/sgg-pretrained-v2/relate-gpu-smoke-v1/`: detector CUDA + relation
  inference đạt; relation profile 2.105 CUDA events và 798 CPU events.
  Released ONNX bank có **243 predicates**, không phải toàn bộ 19.103 chuỗi
  mà API PyTorch full-vocabulary của tác giả có thể dùng.
- Graph semantic (ID/label/attribute/directed triple) CPU/GPU giống nhau cho
  từng phương pháp ở ảnh này. Không suy ra độ ổn định source/target hoặc mọi ảnh.
- REACT smoke 9 node/25 cạnh; RelateAnything 32 node/16 cạnh. Chỉ thống kê cấu trúc,
  **không dùng các số này xếp hạng chất lượng**. Chưa có human ground truth nên
  object/relation P/R/F1, grounding accuracy, hallucination rate vẫn null.
- Parse success: ROBIN cũ 4/5; REACT mới 1/1; RelateAnything mới 1/1. Mẫu số khác
  nhau nên không kết luận hai model mới ổn định hơn ROBIN trên cả pilot.
  Lỗi ROBIN ở `sidewalk-32a2f39d76e383` là `generation_hit_token_limit`.
- First-call timing trong receipt: REACT khoảng 30 giây, relation head khoảng
  4,2 giây (không gồm OWLv2). Có chi phí khởi tạo kernel, **không phải throughput**,
  không so với latency paper. Chưa chạy benchmark tốc độ hoặc đo peak VRAM.
- ORT cảnh báo ScatterND (REACT, chỉ bảo đảm đúng nếu index không lặp) và các
  Memcpy nodes (RelateAnything). Không bỏ qua warning trong báo cáo; graph
  CPU/GPU ở smoke giống nhau nhưng chưa chứng minh mọi input đều không có vấn đề.
- Transformers cảnh báo API postprocess OWLv2 sẽ đổi ở v5; đã pin 4.57.1 và
  slow processor để không đổi preprocessing theo version. Không đổi checkpoint.
- 25 tests mới đạt sau bổ sung kiểm tra cùng inventory GT giữa mô hình; cộng
  regression 37 + 57 = **119 tests CPU đạt** trong các môi trường phù hợp.

Viewer thực tế: `runs/level3/sgg-pretrained-v2/comparison-smoke-v1/comparison.html`.
Ảnh đầu có cả ba phương pháp; bốn ảnh còn lại chỉ có baseline và `not_run` cho
hai phương pháp mới. Hash/evidence tổng hợp trong
[validation JSON](LEVEL3_SGG_PRETRAINED_V2_VALIDATION.json); log chẩn đoán tại
`runs/level3/sgg-pretrained-v2/validation-v1/`. Receipt không hash các file W&B
có thể tiếp tục thay đổi, nhưng hash `wandb-run.json`, config và toàn bộ output mô hình.

### Quan sát định tính, chưa phải nhãn GT

Đối chiếu ảnh nguồn đầu tiên với graph/box (assistant quan sát, **không điền thay
human review**):

- REACT phát hiện người và các ghế rõ ràng, nhưng `person n003 sitting on n000`
  trỏ ghế vàng ở xa người; `sitting on n001` trỏ ghế xanh ở bên trái. Quan hệ
  này không mô tả đúng instance. Box `truck n007` bao xe bán đồ uống bên phải,
  cho thấy lỗi phân loại từ vocabulary đóng.
- OWLv2 + RelateAnything liên kết người với ghế sát người (`n008 → sitting in → n003`),
  nhưng có `n008 → wearing → n028 (hat)`, box mũ nằm trên tóc; ảnh không cho thấy
  mũ rõ ràng. Một số cạnh `chair supporting chair`, `stall hanging from stall`
  cần duyệt vì detector có nhiều vùng chồng lấn/khái niệm rộng. Vocabulary đã
  khóa thiếu `bucket`, dù thùng đỏ là vật nổi bật; **không sửa vocabulary sau khi
  xem ảnh để làm đẹp kết quả**.
- ROBIN mô tả được thùng đỏ và túi trắng, nhưng nhiều SAM regions được diễn giải
  thành cùng một đối tượng/mặt đường; ở ảnh này graph không có người. Đây là vấn
  đề vùng đầu vào và instance identity, không giải quyết bằng việc đếm thêm cạnh.

**Nhận xét cho Task 3:** RelateAnything phù hợp hơn về khả năng mở rộng vocabulary
và tách detector/relation để kiểm soát grounding; kết quả smoke hiện tại vẫn
phụ thuộc mạnh vào detector và chưa đủ để chọn làm extractor chính. REACT++ là
đối chứng triển khai gọn, nhãn hữu hạn dễ giữ thống nhất, nhưng đóng vocabulary
và các cạnh sai endpoint hiện tại có thể tạo consistency score sai. Giữ cả hai
cho lượt 5 ảnh và human review; **chưa chọn model nào là đủ tin cậy cho GCS**.
Ưu tiên kiểm tra semantic precision, recall đối tượng liên quan edit, hallucination
và false changes trên cặp source/target; không chọn bằng độ dày graph hoặc parse
success đơn thuần. Hai phương pháp đều chưa đủ cho yêu cầu edit attribute.
