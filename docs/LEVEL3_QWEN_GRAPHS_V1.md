# REACT++ cho 100 cặp nguồn/đích Qwen

Chạy SGG độc lập với đánh giá chất lượng chỉnh sửa: không cần các nhãn Đúng yêu cầu / Bảo toàn để trích graph. Người dùng đã cho phép assistant chạy GPU lượt này, với điều kiện commit trước inference. Đây là lượt trích 200 graph, chưa phải đánh giá độ chính xác hoặc hoàn tất Graph Consistency Scoring.

## Dữ liệu và mô hình

Đầu vào cố định: `runs/level3/qwen100-v1/seed42-v1/`, đúng 100 cặp/200 PNG. Config mới `configs/level3/qwen-react100-v1.json` pin receipt đầu vào, asset lock và config SGG cha. Xác minh toàn bộ receipt, hash ảnh, task/source duy nhất và trạng thái sinh thành công. Không đổi input hoặc output Qwen, dataset, nguồn/code đã khóa của thí nghiệm cũ.

REACT++ bản PSG YOLO12m ONNX chính thức đã dùng ở pilot: `maelic/REACTPlusPlus_PSG`, revision `25aaf1838c3f6cee32f69a43871bab0870af4bea`, SHA256 ONNX `38c4bdbb7e2a077131ffcbf57af726eb30574318e2344d5b4bfc1d3a48f5b2fe`. Repo `Maelic/SGG-Benchmark` commit `a6642012e66047c3544bed8784bbf9acdc87411b`; preflight kiểm tra repo sạch, đúng commit, hash checkpoint/config/README. Dùng nguyên preprocessing chính thức: letterbox 640, BGR→RGB, CHW, float32 /255. Ngưỡng box 0,4 và triplet relation 0,05 giữ như pilot; decoder chính thức giữ cả object dưới ngưỡng box nếu tham gia relation đã chọn.

Env riêng có sẵn: `data/cache/level3-sgg-pretrained-v2/venv`, Torch 2.7.1+cu128, ONNX Runtime GPU 1.23.2, CUDA 12.8 / sm_120 cho RTX 5090. Không dùng BF16 của editor cho ONNX; giữ kiểu dữ liệu native của export. Batch 1, seed 42, một model và một session, không train/fine-tune. Dùng inference lock chung với SGG cũ. W&B online project `level3-qwen-react100-v1` chỉ ghi config/trạng thái/thời gian; không upload ảnh.

## Chuẩn hóa ảnh và graph

Mỗi ảnh nguồn/đích được EXIF transpose, ghép RGBA trên nền trắng rồi lưu RGB PNG mới, giữ kích thước pixel của từng ảnh. Cách ghép giống viewer Qwen; không chỉ bỏ alpha bằng convert RGB/OpenCV. Hai phía dùng cùng preprocessing 640 của REACT++; độ phân giải ảnh trước letterbox có thể khác nhau và được ghi riêng. Không resize nguồn sang kích thước đích rồi xem đó là cùng giao thức cũ.

Mỗi case có `source-input.png`, `target-input.png`, native ONNX arrays `.npz`, decoded objects/relations `.json` và `source-graph.json`, `target-graph.json`. Graph theo EditGraph hiện tại; field `source_graph` bên trong record là tên schema chung cho graph của một ảnh, vai trò thực ghi ở `image_role`. Có box xyxy theo input RGB, confidence, indices native, provenance/run hash, native/input hashes và thời gian. Mô hình không có attribute head nên attributes rỗng, ghi rõ unsupported. Node ID chỉ cục bộ từng ảnh; không giả định n000 nguồn là n000 đích.

Parse lỗi được ghi riêng và giữ raw arrays nếu đã có; graph rỗng hợp lệ vẫn là dự đoán rỗng cần duyệt, không gộp với lỗi. Lỗi runtime/OOM dừng lượt và lưu phần đã chạy. Mỗi case hoàn tất được đóng receipt ngay. Không tự sinh lại ảnh, đổi threshold hoặc lọc mẫu dựa trên chất lượng.

## CPU trước GPU

```bash
CUDA_VISIBLE_DEVICES='' data/cache/level3-sgg-pretrained-v2/venv/bin/python -m pytest \
  tests/level3/test_qwen_graphs_v1.py tests/level3/test_sgg_pretrained_v2.py -q
CUDA_VISIBLE_DEVICES='' data/cache/level3-sgg-pretrained-v2/venv/bin/python \
  -m tools.level3_qwen_graphs_v1 preflight
```

Đã chạy: 32 tests CPU đạt; preflight 100 cặp/200 ảnh đạt; Bash syntax đạt. Chromium `--disable-gpu` trên fixture tổng hợp kiểm tra phân trang 100 cặp, mũi tên có hướng, bật/tắt bbox và escape prompt. Fixture này không phải kết quả mô hình hoặc nhãn người dùng. Runner từ chối inference khi code/config/launcher liên quan chưa commit.

## Chạy và chạy lại

Sau commit, lượt đầu:

```bash
OUTPUT=runs/level3/qwen-react100-v1/seed42-v1 \
  bash scripts/run_level3_qwen_graphs_v1.sh
```

Chạy lại cùng bộ 100 cặp, cùng model/threshold/seed:

```bash
bash scripts/run_level3_qwen_graphs_v1.sh
```

Mặc định mỗi lần tạo `runs/level3/qwen-react100-v1/run-<UTC timestamp>/`. Có thể chỉ định `OUTPUT` khác; thư mục đã tồn tại sẽ bị từ chối, không ghi đè. Mỗi lệnh tự đi qua đúng 200 ảnh rồi dừng, không chạy tiếp seed/model khác, không chạy editor/CLIP/LPIPS. Không có resume: nếu lỗi kỹ thuật, giữ run lỗi và chọn output mới để chạy lại toàn bộ.

## Xem và diễn giải

```bash
python3 -m http.server 8005 --bind 127.0.0.1 \
  --directory runs/level3/qwen-react100-v1/seed42-v1
```

Mở http://localhost:8005/comparison.html. Chọn từng cặp để xem hai ảnh với bbox, graph có hướng, danh sách node và quan hệ kèm confidence. Trang chấm hai nhãn Qwen trên cổng 8004 vẫn độc lập.

`execution.json` ghi parse success trên tổng 200 ảnh, lỗi, thời gian và VRAM toàn GPU lấy mẫu; ONNX profile chứng minh kernel chạy CUDA, vẫn có các shape/helper ops CPU hợp lệ. Parse success, số node/cạnh và confidence không phải độ chính xác. Precision/recall/F1, grounding accuracy, hallucination và Graph Consistency để null khi chưa có GT/kiểm tra tương ứng. Bước kế tiếp là correspondence, kiểm tra tay độ tin cậy extractor và xây điểm thực hiện thay đổi/bảo toàn; chỉ sau khi có nhãn tay mới đo tương quan.
