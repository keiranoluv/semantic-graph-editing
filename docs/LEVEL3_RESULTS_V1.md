# Level 3 — kết quả đã chốt nhãn, seed 42

Ngày phân tích: 2026-11-16. Bộ hiện tại có **100 ảnh nguồn, 200 câu yêu cầu, 200 ảnh Qwen-Image-2.1 và 300 prediction REACT++**. Người dùng đã chạy CLIPScore/LPIPS trên GPU và xác nhận không sửa nhãn sau khi xem lại 25 ca không đạt. Lượt phân tích này chỉ dùng CPU, không sinh lại ảnh/graph, không đổi ngưỡng hoặc nhãn.

## Kết quả chính

Nhãn tổng hợp = **đúng yêu cầu AND bảo toàn**. So sánh trên đúng **91 cặp cùng có cả ba điểm**, thuộc 62 ảnh nguồn:

| Chỉ số | Spearman với nhãn tổng hợp | Khoảng bootstrap 95% |
|---|---:|---:|
| Graph G = E × P | 0,1767 | [−0,0304; 0,3571] |
| CLIPScore | 0,0561 | [−0,1245; 0,2233] |
| −LPIPS | 0,1962 | [−0,0178; 0,3908] |

**Chưa có bằng chứng graph tốt hơn baseline ảnh.** G có hệ số quan sát cao hơn CLIPScore nhưng thấp hơn −LPIPS. Khoảng của chênh lệch G−CLIPScore là [−0,1795; 0,3964]; G−(−LPIPS) là [−0,3179; 0,2659], đều chứa 0. Tương quan thấp, nhiều ties và ít nhãn không đạt làm kết luận còn yếu.

Bootstrap là phân tích bất định bổ sung: 2.000 lượt, seed 42, lấy lại mẫu **theo ảnh nguồn**, giữ các edit của cùng ảnh trong một cụm và dùng cùng lượt lấy mẫu cho ba chỉ số. Có 2.000 lượt hợp lệ cho mỗi khoảng; nếu một lượt có vector hằng thì để undefined. Khoảng chỉ phản ánh tập 91 cặp hiện được chấm, không khắc phục thiên lệch do abstention, không phải bằng chứng khái quát hóa và không thay thế thực nghiệm nhiều seed. Không báo p-value giả định 200 yêu cầu độc lập.

## Nhãn và độ phủ

- 175/200 cặp đạt cả hai tiêu chí; 25 không đạt ít nhất một tiêu chí. Có 10 ca không đạt đúng yêu cầu, 18 ca không đạt bảo toàn, giao hai nhóm là 3 ca.
- Theo chỉ dẫn **tường minh** của người dùng, 287 trường trống ở 144 cặp được hiểu là đạt; mọi giá trị boolean đã điền được giữ nguyên. Đây là quy ước ghi nhãn của người duyệt, không phải assistant suy đoán nhãn. Không có hàng thiếu được tự thêm.
- Người dùng xác nhận: “Không sửa, dùng bộ nhãn hiện có”. Bản raw giữ nguyên; bản cuối chứa chính sách chuẩn hóa và xác nhận này. Đây là một bộ nhãn của người duyệt, chưa có đo đồng thuận giữa nhiều người.
- Graph có G ở **91/200 = 45,5%**, abstain 109/200 = 54,5%. Tập 91 có 82 đạt/9 không đạt; tập abstain có 93 đạt/16 không đạt. Không bỏ qua độ phủ khi báo cáo tương quan, không quy null thành 0 hoặc PASS.
- E tương quan với nhãn đúng yêu cầu: **0,2008, N=162**. P hợp lệ tương quan với nhãn bảo toàn: **0,0302, N=109**. Phần bảo toàn hiện rất yếu.
- Bảng phụ trên toàn bộ 200 cặp: CLIPScore **0,0329**, −LPIPS **0,2285**. Không dùng hai số này để tuyên bố so sánh ngang với G chỉ có ở 91 cặp.

| Thao tác | Tổng cặp | Có G | ρ G | ρ CLIPScore | ρ −LPIPS |
|---|---:|---:|---:|---:|---:|
| Thêm | 106 | 60 | 0,0753 | −0,0066 | 0,1744 |
| Xóa | 66 | 23 | 0,1271 | −0,0636 | 0,3814 |
| Thay | 18 | 8 | 0,2474 | −0,2474 | −0,5774 |
| Màu | 10 | 0 | Không xác định | Không so trên giao rỗng | Không so trên giao rỗng |

Ba hệ số trong mỗi hàng dùng cùng các cặp có G. Nhóm thay chỉ có 8 cặp, không đủ để kết luận ổn định theo loại edit. 10 yêu cầu màu có E/G=null vì graph không chứa thuộc tính màu; P cấu trúc vẫn được lưu riêng. Nguyên nhân abstention chính là ghép đối tượng bảo toàn mơ hồ (91 ca; reason codes có thể chồng lấp).

## Mười ca đã kiểm tra bằng ảnh thật

Chọn theo quy tắc đã khóa trước khi nhập nhãn: 10 chênh lệch phân vị thứ hạng lớn nhất giữa G và CLIPScore/−LPIPS trên tập chung. Đây là **bất đồng xếp hạng**, không có ngưỡng tự động PASS/FAIL. Cả 10 được chọn thuộc loại thêm vật; không đại diện các loại edit khác và không phải mẫu ngẫu nhiên kiểm định SGG. Các nhận xét dưới đây của assistant, không thay thế nhãn tay.

| Mẫu | Quan sát và nguyên nhân chính |
|---|---|
| #58 — thuyền buồm | G=1, hai nhãn đạt. CLIP xếp thấp; đầu vào crop 224×224 thực sự cắt mất thuyền mới ở phía trái ảnh toàn cảnh. |
| #69 — thuyền chèo | G=1 và −LPIPS xếp cao, ảnh có thuyền và nền gần như giữ nguyên. CLIP xếp thấp dù edit đạt; chưa xác định nguyên nhân embedding cụ thể. |
| #178 — chim trên đá | Ảnh có chim; REACT++ gán bbox tại đó thành person (confidence 0,361), không có bird. G=0 là lỗi do extractor, người duyệt cho đạt/đạt. |
| #168 — mèo ở gốc cây | G=1 dù nhiều chi tiết cây/nền bị vẽ lại; người duyệt cho bảo toàn không đạt. Graph chỉ giữ nhãn/cạnh thô. Crop CLIP cũng cắt mất mèo ở đáy ảnh. |
| #22 — chậu cây | G=0,2 dù hai nhãn đạt. Tủ vẫn còn trong ảnh nhưng mất detection refrigerator; wall đổi thành wall-brick. Chỉ pavement ghép được, gây phạt bảo toàn. |
| #188 — chim bên chùa | Chim đậu mái thay vì lan can; người duyệt cho đúng yêu cầu không đạt. CLIP/−LPIPS xếp cao. G=0 đồng ý với nhãn nhưng vì **thiếu bird detection**, không phải đã hiểu sai vị trí. |
| #160 — chim phía trên đá | Bird được nhận, confidence 0,509; tâm x≈0,6903 nằm ngoài ROI có biên trái 0,72, nên E=G=0. Người duyệt chấp nhận edit; đây là giới hạn của ROI cứng. |
| #187 — chó bên thác | E=1 nhưng G=1/3 do thêm detection river/person/rock và mất cạnh khớp. Ảnh có đổi chi tiết nước/đá (LPIPS cao), mức này người duyệt vẫn chấp nhận. |
| #198 — chim bên ao | G≈0,441, CLIP xếp cao. Nguồn có train giả; đích cũng thật sự thêm lan can lớn ở tiền cảnh. Không thể quy mọi mất điểm cho extractor; giữ nguyên nhãn đạt/đạt của người duyệt. |
| #48 — ghế gỗ | G=0,875 và hai nhãn đạt, CLIP xếp thấp. Crop vẫn chứa rõ ghế nên không thể giải thích bằng mất đối tượng do crop; LPIPS phản ánh thay đổi bề mặt gạch. |

HTML kèm ảnh nguồn/đích nguyên bản, bật/tắt bbox và ROI, confidence, node/cạnh có hướng, phép ghép và giải thích từng ca. Ba hình crop CLIP (#58/#168/#48) được khôi phục từ pixel sau đúng processor đã khóa; chỉ tiền xử lý CPU, không inference lại CLIP và không thay đổi điểm cũ. Việc đối tượng bị crop mất đã được kiểm tra trực tiếp; mức tác động nhân quả lên điểm chưa được đo bằng ablation.

Điểm graph hiện mới kiểm tra category/instance/vùng 2D và bảo toàn các node/cạnh phát hiện được. Nó chưa kiểm chứng đầy đủ tư thế, vật liệu, kích thước, subtype, màu hay kết cấu. Việc một cạnh thấp confidence tồn tại hoặc biến mất cũng có thể làm Dice thay đổi; confidence native không phải xác suất đã hiệu chuẩn trên bộ này. Không đánh giá chất lượng SGG bằng số node/cạnh và không gọi Dice giữa hai graph dự đoán là F1 so graph ground truth.

## Mở và tái lập báo cáo

HTML hiện có:

`runs/level3/evaluation-v1/seed42/reports/0fc28934ad71812a/analysis-final-v1/index.html`

Chạy từ root repo:

```bash
python -m http.server 8010 --bind 0.0.0.0 \
  --directory runs/level3/evaluation-v1/seed42/reports/0fc28934ad71812a/analysis-final-v1
# Mở http://localhost:8010/
```

Nếu server này đang chạy thì chỉ mở URL, không khởi chạy thêm trên cùng port. Ở môi trường từ xa cần forward port 8010. JSON/CSV có liên kết tải ngay trong trang.

Tái lập phần phân tích bằng CPU, không model inference và không chạy hàng đợi:

```bash
bash scripts/analyze_level3_results.sh
```

Lệnh kiểm tra hash đầu vào, giữ quy tắc chọn 10 ca, tính bootstrap, tạo HTML/JSON và receipt. Nếu output đã tồn tại và hash còn đúng thì dùng lại; không ghi đè. Muốn dựng một bản độc lập từ cùng bằng chứng:

```bash
bash scripts/analyze_level3_results.sh --output /tmp/level3-analysis-copy
```

Phần bảng tương quan gốc được tạo bằng lệnh dưới đây **khi chưa có thư mục report theo hash nhãn**; chạy lại sẽ báo `report_already_exists` theo cơ chế bảo toàn kết quả:

```bash
CUDA_VISIBLE_DEVICES='' bash scripts/run_level3_eval.sh report \
  --review runs/level3/evaluation-v1/seed42/labels-final-v1/qwen200-final.json
```

Các lệnh inference ảnh và SGG không đổi; xem [LEVEL3_PIPELINE.md](LEVEL3_PIPELINE.md). Tái tạo ảnh mới không tự kế thừa nhãn tay của ảnh cũ: nếu hash ảnh đổi, cần duyệt lại và lập protocol đánh giá mới.

## Bằng chứng và kiểm tra

- Báo cáo bảng gốc: `runs/level3/evaluation-v1/seed42/reports/0fc28934ad71812a/` — đủ 200 hàng, `preliminary=false`.
- Nhãn raw: `qwen200-reviewed.json`, SHA256 `3e71a883fad961e7c5fc9a503b97b841eee92789c4c7eca0f5a97e6b50ca3e54`; không sửa hoặc commit file này.
- Graph/metric/nhãn/cấu hình và 10 phân tích được khóa hash trong [visual-review.json](../reports/level3/evaluation200-v1/visual-review.json). Tóm tắt số liệu có provenance trong [results-summary.json](../reports/level3/evaluation200-v1/results-summary.json). Ảnh, graph native và nhãn chi tiết tiếp tục nằm trong artifacts local.
- Metric chính do người dùng chạy: 200 cặp hợp lệ, CLIP ViT-B/32 rồi LPIPS AlexNet, FP32, batch 1, seed 42, RTX 5090. [W&B của lượt metric](https://wandb.ai/keiranoluv-hcmus/level3-evaluation200-v1/runs/73zepw44). Không chạy thêm GPU ở lượt phân tích này.
- Code mới ở `tools/level3_analysis_v1/`; không thay đổi các module scorer/metric đã khóa ở `tools/level3_evaluation/`. Tests CPU kiểm tra ties/constant vectors, bootstrap giữ cụm và ghép cặp, loại abstention, từ chối sửa điểm/nhãn, chọn sai ứng viên hoặc ảnh sai hash. Kiểm tra trình duyệt với context riêng, không đụng localStorage review.

## Phần còn lại trước khi nộp

Đã có kết quả tương quan thật và phân tích 10 ca theo yêu cầu Task 3. Chưa nên tuyên bố toàn bộ Level 3 hoàn tất: còn đóng gói data card/nguồn và bằng chứng giấy phép từng ảnh, báo cáo/slide và notebook demo, đồng thời làm rõ yêu cầu nhiều seed theo đề. Notebook cá nhân hiện có không được sửa trong công việc này.

Nếu phát triển phiên bản chấm tốt hơn, ưu tiên độ ổn định ghép đối tượng, xử lý thiếu detection, kiểm tra quan hệ được yêu cầu và bảo toàn chi tiết. Thay đổi CLIP preprocessing cũng phải có phiên bản/đối chứng riêng. Nhãn hiện tại đã được xem để phân tích: nếu dùng để chỉnh phương pháp thì công bố là dữ liệu phát triển, tách theo ảnh nguồn và không gọi kết quả đó là kiểm tra độc lập. Mẫu 10 ca chọn theo bất đồng không thay thế một đợt kiểm định SGG ngẫu nhiên có nhãn tay.
