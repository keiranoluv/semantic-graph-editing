# Chấm Graph Consistency cho 100 cặp Qwen

Bộ chấm CPU dùng graph REACT++ nguồn/đích đã đóng receipt, câu yêu cầu gốc và phần gắn ý định với ảnh nguồn do assistant thực hiện. Không chạy thêm mô hình ảnh, không huấn luyện, không dùng nhãn chất lượng người dùng để chỉnh ngưỡng. Người dùng yêu cầu commit code trước khi chạy chấm toàn bộ; runner kiểm tra code/config/launcher liên quan đã commit.

## Đầu vào và diễn giải ngôn ngữ

Graph run: `runs/level3/qwen-react100-v1/seed42-v1`; Qwen run: `runs/level3/qwen100-v1/seed42-v1`. Cả 100 cặp được giữ trong kết quả, kể cả trường hợp không đủ bằng chứng chấm. Config mới pin receipts, manifest instruction và source grounding; không sửa dữ liệu, graph hoặc code thí nghiệm cũ.

`configs/level3/qwen-score-grounding-v1.json` là phần assistant diễn giải 100 câu từ **ảnh nguồn + graph nguồn**: add có vùng 2D dự kiến cho tâm bbox đối tượng mới; remove/replace có source node cụ thể nếu nhìn thấy trong graph, cùng vùng nguồn mở rộng 0,05 mỗi phía. Vùng add được vẽ xấp xỉ thủ công theo nội dung câu, không học từ output. Mỗi binding lưu source graph hash, tác giả và giới hạn; đây không phải dự đoán Task B hoặc graph GT. Lượt gắn vùng này không đọc ảnh/graph đích; một số mẫu pilot/đích đã từng được xem ở các lượt trước, nên không gọi đây là test mù hoặc held-out.

Đã xem 13 contact sheets, đủ 100 nguồn; xem thêm ảnh gốc hai trường hợp khó. Bốn đối tượng yêu cầu không có trong graph nguồn: mẫu 10 (bicycle), 13 (stool), 49 (stool), 82 (stool). Giữ binding null thay vì chọn nhầm đối tượng khác. Mẫu 46 mô tả "yellow and black scooter" chưa chính xác: thân xe đỏ sẫm/đen, phần vàng là phụ kiện phủ lên xe; giữ instruction cũ và đánh dấu mơ hồ. Bằng chứng source-only: `runs/level3/qwen-graph-score-v1/source-grounding-evidence-v1/`.

Đây là **parser/grounding có assistant hỗ trợ cho 100 instruction đã cố định**, không phải mô hình hiểu câu tự do chạy tự động. Khi thêm câu mới phải tạo phiên bản source grounding mới. Bộ chấm chỉ đánh giá loại đối tượng, instance và vùng 2D; không đủ head/evidence để kiểm tra màu, vật liệu, subtype (rowboat/sailboat), tư thế, kích thước, texture hay chiều sâu. Các phần này phải được kiểm tra qua ảnh; không tuyên bố đã hiểu đầy đủ mọi mệnh đề chỉ từ graph.

## Ghép và chấm

1. Ghép one-to-one theo nhãn chuẩn hóa case/whitespace và IoU bbox chuẩn hóa theo kích thước riêng của mỗi ảnh, ngưỡng 0,3. Hungarian với dummy cho phép unmatched; ưu tiên số ghép hợp lệ rồi tổng IoU. Không dùng node ID giống nhau để suy ra cùng instance. Chênh lệch IoU dưới 0,05 với phương án cạnh tranh được ghi mơ hồ.
2. **Edit E:** kiểm tra instance được yêu cầu, không chỉ đếm tổng node. Remove cần đúng source anchor biến mất, số instance cùng nhãn giảm, không có bằng chứng mơ hồ về ghép/đổi nhãn. Add cần detection mới chưa ghép, đúng nhãn, tâm trong vùng nguồn, confidence ít nhất 0,4; biến động cùng nhãn có thể là di chuyển thì từ chối kết luận. Replace kiểm tra cả bỏ instance cũ và thêm đúng loại mới tại vùng cũ; E là trung bình hai điều kiện. Không có detection mới phù hợp cho E=0 trên graph, kèm cảnh báo rằng extractor có thể bỏ sót edit đúng trong ảnh.
3. **Preservation P:** loại đúng node được phép thay đổi và các cạnh incident, cùng tối đa một instance mới được hỗ trợ. So tập node còn lại và tập cạnh **có hướng, đúng predicate** sau alignment bằng Dice `2|A∩B|/(|A|+|B|)`. P là trung bình hai phần khả dụng; hai tập cạnh cùng rỗng là không quan sát được, không tự gán edge preservation=1. Cạnh/node mất và thêm được xuất để kiểm tra. Không so pixel nên không đo được mọi hư hại ảnh.
4. **G = E × P:** ảnh không thực hiện thay đổi có E=0; xóa toàn cảnh có P=0; cả hai không thể được điểm cao chỉ nhờ một trục còn lại. Nếu edit không đủ bằng chứng, không có phần bảo toàn quan sát được, hoặc alignment bảo toàn mơ hồ, `graph_consistency=null` và `status=abstained`. Điểm thành phần và tích chẩn đoán vẫn có thể hiện riêng để giải thích; không dùng tích đó thay điểm tổng đã abstain.

Confidence chỉ là score mô hình, chưa được hiệu chỉnh. Ngưỡng IoU/ambiguity kế thừa ý tưởng pilot (0,3/0,05), confidence 0,4 cùng ngưỡng object REACT đã dùng; overlap 0,5 chỉ phát hiện khả năng đổi nhãn. Tất cả cố định trước khi chấm cohort, không fit nhãn tay. Không thay đổi graph hay threshold REACT để làm điểm đẹp hơn.

Dice ở đây đo tính nhất quán giữa **hai graph dự đoán**, không phải object/relation precision/recall/F1 so GT. Graph có thể sai cùng nhau hoặc detector mất/đổi nhãn, nên G không phải xác suất edit đúng. Báo cáo coverage và lý do abstain trên toàn bộ 100 mẫu; không chỉ xuất những mẫu dễ chấm.

## Kiểm tra và chạy

```bash
CUDA_VISIBLE_DEVICES='' data/cache/level3-sgg-pretrained-v2/venv/bin/python -m pytest \
  tests/level3/test_qwen_scoring_v1.py tests/level3/test_task3_v1.py -q
CUDA_VISIBLE_DEVICES='' data/cache/level3-sgg-pretrained-v2/venv/bin/python \
  -m tools.level3_qwen_scoring_v1 preflight
```

Đã chạy trước commit/cohort scoring: **56 tests CPU đạt**, preflight 100 cặp đạt, Bash syntax và Chromium fixture đạt. Tests có xóa nhầm instance, thêm nhầm vùng, di chuyển vật cũ, thay nửa yêu cầu, xóa toàn cảnh, đảo hướng quan hệ, đổi nhãn, confidence thấp, missing/ambiguous anchors, ID permutation, khác độ phân giải, output bất biến và phân biệt assistant review với GT người dùng.

Sau commit, chấm 100 cặp:

```bash
OUTPUT=runs/level3/qwen-graph-score-v1/score-v1 \
  bash scripts/run_level3_qwen_scoring_v1.sh
```

Chạy lại:

```bash
bash scripts/run_level3_qwen_scoring_v1.sh
```

CPU, không có model/batch GPU; duyệt tuần tự 100 cặp, seed protocol 42, không hàng đợi. Output mặc định `runs/level3/qwen-graph-score-v1/run-<UTC timestamp>/`, không ghi đè. Env SGG cũ cung cấp NumPy/SciPy/Pillow/W&B; không đổi dependencies. W&B project `level3-qwen-graph-score-v1` chỉ nhận config/metrics, không ảnh/nhãn người dùng.

## Review của assistant

20 task được chọn và pin trước khi tính điểm: tất cả 6 replace, cùng add/remove và các source-extractor gaps. Đây là lựa chọn chẩn đoán có chủ đích, không phải mẫu ngẫu nhiên hoặc ước lượng chất lượng toàn bộ. Assistant đã xem cặp ảnh thật và graph đóng băng, ghi nhận xét cụ thể cho hai trục, với giá trị `appears_satisfied`, `appears_not_satisfied` hoặc `uncertain`. File có `author=assistant`, `human_ground_truth=false`, gắn score-run receipt. Không chép vào nhãn tay hoặc dùng để báo Spearman như ground truth.

Review và dashboard bổ sung được ghi thành output mới, không sửa score run. Trang người dùng chấm hai trường Qwen vẫn độc lập; không tự điền các lựa chọn của bạn.

```bash
CUDA_VISIBLE_DEVICES='' data/cache/level3-sgg-pretrained-v2/venv/bin/python \
  -m tools.level3_qwen_scoring_v1 review \
  --score-run runs/level3/qwen-graph-score-v1/score-v1 \
  --assistant-review reports/level3/qwen-graph-score-v1/assistant-review.json \
  --output runs/level3/qwen-graph-score-v1/review-v1
```

CLIPScore/LPIPS và tương quan với nhãn tay chưa được tính trong lượt này. Những bước đó cần triển khai/kiểm tra trên cùng cohort và xử lý coverage rõ ràng; không dùng review assistant thay nhãn nhóm.

## Kết quả thực chạy

Code commit **`255b78200a2aaeb5d684e0da2ad26128ca76a35b`** trước khi chấm toàn bộ. Run `score-v1` hoàn tất 100/100 cặp trên CPU, không GPU inference, không OOM. Thời gian được runner ghi là 1,820 giây cho khởi tạo tracking và vòng chấm; không bao gồm đọc/kiểm tra hash đầu vào, render HTML và kết thúc upload W&B. [W&B run amu21v09](https://wandb.ai/keiranoluv-hcmus/level3-qwen-graph-score-v1/runs/amu21v09).

| Đại lượng | Số mẫu có giá trị | Giá trị |
|---|---:|---:|
| Coverage điểm tổng G | 45/100 | 0,450 |
| Trung bình G trên phần có điểm | 45 | 0,7303 |
| Trung bình E chẩn đoán | 87 | 0,9195 |
| Trung bình P chẩn đoán | 100 | 0,7070 |

**Đây không phải accuracy hoặc tỷ lệ edit đúng.** 55 mẫu abstain không có nghĩa Qwen làm sai 55 mẫu. Các trung bình có mẫu số khác nhau; đặc biệt G chỉ phản ánh 45 trường hợp chấm được, không đại diện mặc nhiên cho toàn bộ 100 ảnh. P vẫn hiện để chẩn đoán khi alignment chưa chắc, không có nghĩa P đã được xác nhận đáng tin.

Lý do abstain có thể trùng trên một mẫu: alignment bảo toàn mơ hồ 50; count/movement instability 4; thiếu đối tượng yêu cầu trong graph nguồn 4; đối tượng mới confidence thấp 3; instruction nguồn mơ hồ 1; alignment đối tượng mới mơ hồ 1. Giữ nguyên ngưỡng và 100 mẫu sau khi xem kết quả. GT F1, Spearman với nhãn người dùng, CLIPScore và LPIPS đều `null`; số nhãn người dùng đọc là 0.

Artifacts:

- `runs/level3/qwen-graph-score-v1/score-v1/`: điểm, evidence, config/code hashes, môi trường, W&B và receipt.
- `runs/level3/qwen-graph-score-v1/assistant-image-evidence-v1/`: 5 contact sheets chứa 20 cặp ảnh thật, không hiển thị điểm.
- `reports/level3/qwen-graph-score-v1/assistant-review.json`: 20 nhận xét assistant, từng nhận xét pin ảnh/graph/score hash.
- `runs/level3/qwen-graph-score-v1/review-v1/comparison.html`: 100 cặp, 3 điểm, nguyên nhân abstain, ghép node, graph/box/confidence và 20 review.
- `runs/level3/qwen-graph-score-v1/validation-v1/`: kiểm tra artifacts, trình duyệt và ảnh chụp dashboard.

Mở dashboard bằng:

```bash
python -m http.server 8007 --bind 127.0.0.1 \
  --directory runs/level3/qwen-graph-score-v1/review-v1
```

Truy cập <http://localhost:8007/comparison.html>. Nếu server đã chạy thì mở URL trực tiếp; môi trường remote cần forward cổng 8007. Dashboard này tách riêng giao diện Qwen hai trường của người dùng ở cổng 8004, không ghi localStorage hay nhãn tay.

Xem [nhận xét và giới hạn thực tế](LEVEL3_QWEN_SCORING_REVIEW_V1.md). Chưa nên dùng một ngưỡng G duy nhất để tự động kết luận edit đúng/sai. Bước tiếp theo là đánh giá ghép instance và false detections trên các ca đã phát hiện, sau đó so tương quan với nhãn tay độc lập; mọi sửa thuật toán cần phiên bản mới, giữ nguyên lượt này.
