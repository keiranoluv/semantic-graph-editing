# Level 3: đánh giá 200 cặp Qwen / REACT++

Phần này đọc bộ **100 nguồn, 200 ảnh đích, 300 SG** đã có. Không sinh lại ảnh/SG, không huấn luyện và không sửa các manifest hoặc kết quả cũ. Hai lệnh tạo ảnh/SG trong [LEVEL3_PIPELINE.md](LEVEL3_PIPELINE.md) giữ nguyên. Code đánh giá ở `tools/level3_evaluation/`; cấu hình mới `configs/level3/evaluation200-v1.json` khóa bằng SHA256 bộ `seed42` hiện tại. Lượt tái lập ảnh mới phải được xác minh/grounding riêng trước khi đánh giá, không tái sử dụng mù node ID của lượt cũ.

## Chạy và resume

Môi trường riêng (đã chuẩn bị trên máy hiện tại):

```bash
python3 -m venv data/cache/level3-evaluation-v1/venv
data/cache/level3-evaluation-v1/venv/bin/pip install -r requirements-level3-evaluation-v1.txt
bash scripts/run_level3_eval.sh check
```

`check` chỉ đọc, kiểm tra 200 cặp, hash checkpoint và giới hạn caption; không chạy model. Checkpoint CLIP/LPIPS dùng cache đã giữ ở `data/cache/level3-task3-v1/`, theo hash trong `configs/level3/task3-assets-v1.json`. Không tải ngầm khi inference và không cần checkpoint IP2P. Với máy khác, tải các mục `clip/`, `lpips/`, `torch/` từ lock đó trước khi `check`.

Sau khi code đã commit, chấm graph trên CPU:

```bash
CUDA_VISIBLE_DEVICES='' bash scripts/run_level3_eval.sh graph
```

Kết quả: `runs/level3/evaluation-v1/seed42/graph/{results,summary}.json`. Lặp lại cùng lệnh chỉ xác minh và dùng lại kết quả đã hoàn tất. Không có GPU hoặc lời gọi editor/SGG.

CPU smoke thật của hai metric trên đúng một cặp có sẵn:

```bash
CUDA_VISIBLE_DEVICES='' bash scripts/run_level3_eval.sh metrics --device cpu --smoke
```

Đầu ra `smoke-cpu/` được đánh dấu **không dùng để đánh giá**. Nhánh này không cần W&B. Kiểm tra thêm LPIPS của cùng một ảnh phải bằng 0.

Lệnh GPU để người dùng tự chạy:

```bash
# Một cặp thật, không đưa vào bảng thực nghiệm
bash scripts/run_level3_eval.sh metrics --device cuda --smoke

# Toàn bộ 200 cặp hiện có, có resume
bash scripts/run_level3_eval.sh metrics --device cuda
```

Phạm vi: **CLIP ViT-B/32** (`openai/clip-vit-base-patch32`, revision khóa) rồi **LPIPS AlexNet v0.1**, từng model một, **batch 1, FP32, seed 42**. CUDA 12.8, Torch có kernel `sm_120` cho RTX 5090. Lệnh đầy đủ tự chuyển từ CLIP sang LPIPS trên cùng 200 cặp, rồi dừng; không chạy tiếp Qwen, SGG hoặc thực nghiệm khác. GPU smoke và CPU smoke là các lượt riêng. Output metric chính ở `runs/level3/evaluation-v1/seed42/metrics/`, có W&B online, runtime packages, thời gian từng cặp, peak VRAM allocated/reserved của process và hash checkpoint. Cần W&B đã đăng nhập cho lượt đầy đủ.

Mỗi metric/cặp có receipt riêng. Ngắt hoặc OOM thì chạy lại đúng lệnh để tiếp tục phần thiếu; không tự giảm độ phân giải hoặc đổi device. Nếu đổi code, cấu hình hoặc device cho một lượt metric đã có, dùng `OUTPUT=runs/level3/evaluation-v1/<ten-moi>` để giữ lượt cũ. Lượt metrics mới chưa có điểm graph cần chạy lại lệnh `graph` với cùng `OUTPUT` trước khi report.

## Chấm graph và grounding

100 cặp đầu dùng lại grounding cũ; 100 yêu cầu mới đã được assistant kiểm tra ảnh nguồn và bbox graph nguồn để chọn node hoặc vùng 2D. File mới `configs/level3/evaluation200-authoring-v1.json` giữ nguồn, hash, tác giả và ghi chú. Không sửa manifest instruction, graph hoặc bounding box. Vật nhìn thấy nhưng thiếu node, hoặc bbox gộp nhiều vật/bao cả cảnh, không được giả làm anchor hợp lệ.

Scorer thêm/xóa/thay dùng nguyên thuật toán v1: ghép một-một theo nhãn + IoU, kiểm tra edit trong vùng yêu cầu, Dice bảo toàn node và quan hệ có hướng, **G = E × P**. Ngưỡng giữ nguyên; không điều chỉnh theo nhãn tay. `null` và reason codes được giữ khi bằng chứng mơ hồ. P thô vẫn được lưu để chẩn đoán, nhưng P có alignment mơ hồ không được dùng trong tương quan thành phần (`preservation_evaluable=false`). Không gọi thêm LLM, không dùng mô hình Task 2 và không tuyên bố tự hiểu câu tự nhiên hoàn toàn.

10 yêu cầu đổi màu: **E/G luôn null**, vì REACT++ không có attribute head. P chỉ kiểm tra bảo toàn cấu trúc, giữ cả đối tượng đang đổi màu trong tập node/cạnh cần giữ. Grounding giúp chỉ vật theo màu trong ảnh nguồn nhưng không bổ sung thuộc tính màu vào graph đích.

Đây là heuristic trên graph dự đoán. Điểm không phải xác suất ảnh đúng hoặc F1 so với graph GT. Bản thân detection/relation sai, thay đổi occlusion, mất instance hoặc bbox không ổn định vẫn giới hạn bộ chấm. Caption/grounding do assistant chuẩn bị không phải nhãn tay của nhóm. Một số kết quả 100 cặp đầu đã được xem trong các bước phát triển trước; không gọi đây là thử nghiệm mù hay held-out test.

## CLIPScore và LPIPS

- CLIPScore = `2.5 * max(cos(CLIP(target), CLIP("A photo depicts " + target_caption)), 0)`. Caption mong muốn được chuyển thành câu mô tả từ instruction cố định và bối cảnh nguồn; không caption ảnh Qwen để thưởng cho chính kết quả nó tạo. Có tác giả/provenance, không truncate quá 77 token. Phủ định và thay thế vẫn có thể khó với CLIP; điểm cao không chứng minh bảo toàn ảnh nguồn.
- LPIPS so ảnh nguồn và ảnh đích, RGB `[-1,1]`. Ảnh nguồn resize LANCZOS về kích thước native ảnh đích, không crop. Lưu kích thước trước/sau, raw LPIPS và `negative_lpips`. LPIPS thấp có thể là ảnh không được sửa, không tự suy ra edit thành công.
- Cả hai là pretrained inference, không fine-tune. Triển khai theo [CLIPScore](https://github.com/jmhessel/clipscore) và [LPIPS](https://github.com/richzhang/PerceptualSimilarity); CLIP dùng adapter Hugging Face, không tuyên bố bitwise tương đương script gốc.

## Review và đánh giá sau khi có nhãn

```bash
bash scripts/run_level3_qwen.sh serve
# http://localhost:8008/comparison.html
```

Giao diện hiện tại vẫn chỉ hai trường **Đúng yêu cầu / Bảo toàn**; không hiện điểm tự động, không có Pix2Pix và không yêu cầu ghi note. Nhãn nằm trong localStorage của browser/origin đang dùng; nên xuất `qwen200-reviewed.json` để sao lưu. Có thể nhập lại JSON vào đúng bộ ảnh. Kiểm tra trình duyệt dùng context độc lập, không sửa nhãn người dùng.

Sau khi metric và nhãn tay hoàn tất:

```bash
bash scripts/run_level3_eval.sh report --review /duong/dan/qwen200-reviewed.json
```

Mặc định yêu cầu đủ 200 nhãn hợp lệ; `--allow-partial` chỉ tạo báo cáo sơ bộ. Từ chối ID trùng/lạ, sai bộ ảnh/hash, boolean giả dạng số và nhãn cũ khác kết quả. Nhãn overall pass = cả hai trường đều true. Không dùng nhãn để tính các điểm tự động.

Report tạo thư mục mới theo hash file nhãn, không ghi đè: `reports/<hash>/summary.json`, `results.csv`, `comparison.html`, `disagreement-candidates.json`, bản nhãn đầu vào. Spearman chính so **G, CLIPScore, −LPIPS trên cùng giao của các mẫu có đủ điểm và nhãn**; bảng phụ dùng mọi mẫu hợp lệ của từng metric và ghi rõ N. Có bảng riêng theo thao tác/nguồn dataset, E với nhãn fulfillment và P hợp lệ với nhãn preservation. Thiếu nhãn/điểm giữ null; vector hằng hoặc dưới hai mẫu trả undefined, không bịa 0. Không báo p-value giả định các yêu cầu độc lập, vì mỗi ảnh có hai yêu cầu.

10 ứng viên phân tích được chọn theo chênh lệch thứ hạng chuẩn hóa giữa G và chỉ số ảnh. Đây mới là danh sách để xem ảnh, graph và nhãn; **không tự coi là đã hoàn thành yêu cầu phân tích định tính 10 trường hợp**. Nhãn Level 3 chỉ dùng đánh giá; nếu sau này dùng để tinh chỉnh, phải tách theo ảnh nguồn và công bố phần phát triển.

## Kiểm tra CPU

```bash
CUDA_VISIBLE_DEVICES='' data/cache/level3-evaluation-v1/venv/bin/python -m pytest -q \
  tests/level3/test_evaluation200.py tests/level3/test_qwen_scoring_v1.py
# Browser checks dùng môi trường có Playwright/Chromium, không cần GPU:
CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  tests/level3/test_qwen_unified_browser.py tests/level3/test_qwen_sgg_browser.py
```

Tests bao gồm edit không thực hiện, sai instance, cạnh đảo chiều, thuộc tính không hỗ trợ, alignment mơ hồ, review sai binding, nhãn thiếu/trùng, Spearman có ties/hằng, cùng tập so sánh, metric resume sau ngắt, chống sửa output và report từ chối smoke. Kết quả chạy thực tế được ghi riêng trong `LEVEL3_EVALUATION_VALIDATION.json` sau khi thực hiện.

## Kết quả thực tế của lượt hiện tại

Code đã commit tại `bbffc05` trước khi chạy bộ chấm đầy đủ. **48 tests CPU + 2 tests Chromium qua**; đã kiểm tra thêm giao diện 200 ảnh thật bằng browser context riêng: ảnh đầu/cuối tải được, hai trường lưu sau reload và round-trip JSON hoạt động. Không thay đổi nhãn trong trình duyệt người dùng.

Bộ chấm CPU đã xử lý **200 cặp: 91 có G, 109 abstain, coverage 45,5%**. E có ở 162 cặp; P đủ điều kiện đối chiếu nhãn ở 109 cặp. 100 điểm cũ khớp toàn bộ trường cũ, 100 cặp mở rộng có 46 G xác định. Cả 10 yêu cầu màu có E/G=null. Nguyên nhân chính là 91 ca alignment phần cần bảo toàn bị mơ hồ; các reason codes có thể cùng xuất hiện trên một ca. Đây không phải tỷ lệ ảnh chỉnh sửa đúng. Chạy lại lệnh graph đã xác nhận resume kết quả hiện có.

CPU smoke thật của CLIP/LPIPS trên `sidewalk-76268403d14e5f-q100-1` thành công; LPIPS của hai ảnh giống hệt nhau bằng 0. Thư viện LPIPS phát cảnh báo API `pretrained` cũ của torchvision, nhưng checkpoint khóa tải và inference thành công. **Chưa chạy GPU hoặc metric toàn bộ 200 cặp**, chưa dùng nhãn tay và chưa có Spearman. 100 nguồn, 200 cặp Qwen và 300 prediction SGG gồm native output/receipt đã được kiểm tra lại hash; ảnh/graph cũ được giữ nguyên.
