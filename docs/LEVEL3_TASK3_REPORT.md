# Nhiệm vụ 3: đánh giá tính nhất quán của chỉnh sửa ảnh bằng scene graph

**Project 1 — Biểu diễn đồ thị ngữ nghĩa cho chỉnh sửa ảnh có kiểm soát**
Phiên bản kết quả: seed42, chốt nhãn ngày 16/11/2026 (Asia/Bangkok).
Phạm vi: báo cáo riêng Nhiệm vụ 3, dùng để ghép vào báo cáo chung của đồ án.

## Tóm tắt

Chúng tôi xây bộ đánh giá gồm 100 ảnh bối cảnh Việt Nam từ hai dataset công khai, mỗi ảnh có hai yêu cầu chỉnh sửa độc lập. Qwen-Image-2.1 tiền huấn luyện tạo 200 ảnh đích; REACT++ trích 300 scene graph cho 100 nguồn và 200 đích. Bộ chấm dựa trên graph gồm mức thực hiện yêu cầu E, mức bảo toàn phần không liên quan P, và điểm tổng hợp G = E × P, có quyền từ chối chấm khi bằng chứng mơ hồ.

Người duyệt cung cấp hai tiêu chí đạt/không đạt cho toàn bộ 200 kết quả. Trên 91 cặp cùng có G, CLIPScore và LPIPS, Spearman với nhãn tổng hợp lần lượt là **0,1767; 0,0561; 0,1962** với LPIPS đã đảo dấu. Độ phủ G là **45,5%**. Phân tích 10 ca bất đồng cho thấy lỗi phát hiện/đổi nhãn đối tượng, ROI cứng, thiếu biểu diễn thuộc tính và kết cấu ảnh là các giới hạn chính. Chưa có bằng chứng graph vượt baseline ảnh; đóng góp của thử nghiệm là pipeline có thể tái kiểm tra, công bố abstention và giải thích được các ca lỗi bằng bằng chứng cụ thể.

## 1. Bài toán và yêu cầu đánh giá

Đầu vào của một case gồm ảnh nguồn I_s, yêu cầu t và ảnh đã chỉnh sửa I_t. Mục tiêu là đánh giá đồng thời:

1. **Edit fulfillment:** thay đổi được yêu cầu đã xảy ra đúng đối tượng/vùng hay chưa.
2. **Preservation:** phần không liên quan có được giữ lại hay không.

Theo đề, Nhiệm vụ 3 cần ít nhất 100 ảnh, 150–200 yêu cầu, công cụ chỉnh sửa có sẵn, graph trước/sau, nhãn tay toàn bộ kết quả, so sánh Spearman với CLIPScore/LPIPS và phân tích ít nhất 10 trường hợp bất đồng. Chỉ sử dụng pretrained inference cho các mô hình thị giác; không huấn luyện editor hoặc SGG.

Điểm graph trong thử nghiệm là heuristic có grounding nguồn được chuẩn bị trước. Nó chưa phải mô hình học trực tiếp từ câu yêu cầu, và chưa sử dụng dự đoán thao tác của mô hình Nhiệm vụ 2 để chấm toàn bộ bộ ảnh này.

## 2. Bộ đánh giá

### 2.1. Nguồn và quy mô

| Nguồn | Ảnh nguồn | Yêu cầu/ảnh đích | Giấy phép công bố |
|---|---:|---:|---|
| [Vietnamese Sidewalk Violation VQA](https://huggingface.co/datasets/manhdungcr7/dataset_viahe_vqa) | 50 | 100 | CC BY-NC 4.0 |
| [VietTravelVQA-v2](https://huggingface.co/datasets/phunguyendinhkh/VietTravelVQA-v2), [DOI bản v2](https://data.mendeley.com/datasets/fvxtpnh8mh/2) | 50 | 100 | CC BY 4.0 cho dataset release |
| Tổng | **100** | **200** | Ghi công theo từng nguồn |

Nguồn, revision, URL từng ảnh và SHA256 nằm trong manifest. Ảnh được chọn có chủ đích cho nội dung bối cảnh Việt Nam và khả năng chỉnh sửa, không phải mẫu đại diện ngẫu nhiên. VietTravelVQA dẫn ảnh từ Wikimedia Commons và giữ các điều kiện ảnh gốc. [Data card](LEVEL3_DATA_CARD.md) mô tả chi tiết nguồn, cách chọn và ghi nhãn.

Hai yêu cầu của cùng ảnh đều chỉnh từ ảnh nguồn. Phân bố gồm **106 thêm, 66 xóa, 18 thay đối tượng và 10 đổi màu**. Nhóm màu được đưa vào để khảo sát giới hạn biểu diễn, không được loại khỏi báo cáo tổng số 200 cặp.

### 2.2. Nhãn và khả năng kiểm chứng

Giao diện duyệt chỉ có hai trường: đúng yêu cầu và bảo toàn. Theo xác nhận của người dùng, mỗi trường trống có nghĩa là đạt; mọi giá trị đã điền rõ được giữ nguyên. Có 287 trường null ở 144 cặp được chuẩn hóa theo quy tắc này. Raw JSON, bản chuẩn hóa và xác nhận xem lại 25 ca không đạt đều được giữ với hash.

Nhãn cuối có 175 cặp đạt cả hai, 25 cặp không đạt ít nhất một tiêu chí; 10 không đạt fulfillment, 18 không đạt preservation và 3 không đạt cả hai. Nhãn tổng hợp là phép AND. Nhận xét ảnh của assistant là phân tích bổ sung, không phải nhãn tay thay thế; chưa đo độ đồng thuận nhiều người duyệt.

### 2.3. Phạm vi sử dụng nhãn

Scorer, ngưỡng, caption và quy tắc chọn ca bất đồng được khóa trước bước đối chiếu nhãn. Không sửa điểm dựa trên nhãn cuối. Toàn bộ cohort vẫn được mô tả là đánh giá phát triển: một số ảnh/kết quả đã được xem trong các bước xây pipeline. Không tuyên bố đây là held-out test mù. Nếu dùng các phát hiện hiện tại để chỉnh v2, cần công bố tập phát triển và chia theo ảnh nguồn, không tách hai yêu cầu cùng ảnh sang hai nhóm.

## 3. Pipeline pretrained và provenance

### 3.1. Sinh ảnh

Qwen dùng checkpoint `Qwen/Qwen-Image-2.1`, revision `d26bb61231c349cf6b7896fa83353113880e1ba3`, Diffusers, BF16, batch 1, seed 42, 40 inference steps và CPU offload trên RTX 5090/CUDA 12.8. Cấu hình khóa có `true_cfg_scale=1.0`, `use_kv_cache=true`, output resolution 1024 theo pipeline Qwen; kích thước output thực tế được lưu từng ảnh và có thể khác nhau theo tỷ lệ khung hình.

Mỗi case lưu ảnh nguyên bản tham chiếu, source sau EXIF/RGB, target, prompt, cấu hình, thời gian, VRAM, hash và receipt. Không chọn ảnh tốt nhất qua nhiều lần sinh theo nhãn chất lượng. Hai lệnh Qwen/SGG độc lập có resume; kết quả lịch sử được giữ nguyên.

### 3.2. Trích graph

REACT++ dùng model PSG `maelic/REACTPlusPlus_PSG`, revision `25aaf1838c3f6cee32f69a43871bab0870af4bea`, ONNX `yolo12m/react_pp_yolo12m.onnx`. Mã decoder/repository được khóa trong `configs/level3/sgg-pretrained-assets-v2.json`. Batch 1, seed 42; chỉ inference.

Output gồm object label, bbox, confidence, quan hệ subject–predicate–object có hướng và provenance. Native NPZ, decoded JSON và EditGraph đều được giữ. Có 300 prediction độc lập; source graph được tái sử dụng cho hai instruction, vì vậy dữ liệu theo cặp chứa 400 graph binding. REACT++ trong cấu hình này không dự đoán thuộc tính màu.

Việc chọn REACT++ xuất phát từ pilot so sánh trực quan và nhu cầu graph dễ kiểm tra. Chưa có graph ground truth để kết luận extractor có object/relation F1 cao hơn mô hình khác. Số node/edge ít hoặc nhiều không được dùng làm thước đo độ chính xác.

## 4. Bộ chấm graph

### 4.1. Grounding yêu cầu

Instruction cố định được kèm loại thao tác, nhãn cần sửa và anchor/ROI 2D. Grounding do assistant chuẩn bị từ ảnh và graph **nguồn**, không phải nhãn tay chất lượng ảnh đích hoặc đầu ra mô hình Task 2. Phiên bản giữ provenance của phần kế thừa và phần authoring bổ sung.

### 4.2. Ghép đối tượng

Với bbox chuẩn hóa về [0,1], chỉ cho ghép khi nhãn đối tượng bằng nhau sau chuẩn hóa chữ và IoU ≥ 0,3. Hungarian tìm phép ghép một-một, ưu tiên số cặp hợp lệ rồi tổng IoU. Nếu ứng viên cạnh tranh gần nhau với margin dưới 0,05, phép ghép có thể bị đánh dấu mơ hồ. Matching không dựa vào node ID giữa hai ảnh.

### 4.3. Edit fulfillment E

- **Thêm:** tìm instance nhãn yêu cầu chưa ghép, confidence ≥ 0,4 và tâm bbox nằm trong ROI. Kiểm thêm số lượng để tránh xem vật thể di chuyển là một vật mới.
- **Xóa:** kiểm anchor nguồn không còn ghép và số lượng phù hợp. Các dấu hiệu đổi nhãn hoặc di chuyển có thể dẫn đến abstain; IoU 0,5 được dùng trong kiểm tra đổi nhãn khả nghi.
- **Thay:** thực hiện kiểm xóa đối tượng cũ và thêm đối tượng mới; E là trung bình hai giá trị nếu cả hai xác định. Vì vậy E có thể bằng 0,5, không phải chỉ boolean.
- **Đổi màu:** E không xác định vì graph không có thuộc tính màu.

E chủ yếu kiểm category/instance/vùng 2D. Nó chưa kiểm chứng đầy đủ tư thế, vật liệu, kích thước, subtype hoặc mọi quan hệ ngôn ngữ. Thiếu detection không chứng minh việc xóa đã thực sự xảy ra trên ảnh.

### 4.4. Preservation P và điểm tổng hợp

Loại khỏi tập bảo toàn các node nguồn được phép sửa và node mới được phép thêm. Biểu diễn các node đích đã ghép bằng ID nguồn; node chưa ghép vẫn giữ danh tính đích riêng. Cạnh được so bằng bộ ba **có hướng**, trên các node được bảo toàn.

Với hai tập A, B:

`Dice(A,B) = 2 × |A ∩ B| / (|A| + |B|)`.

Nếu cả hai tập rỗng, thành phần đó không xác định; không tự cho bằng 1. P là trung bình các thành phần Dice node/cạnh xác định. Khi E và P đủ bằng chứng và không có reason code abstention, `G = E × P`; các trường hợp còn lại giữ G=null.

P thô có thể được lưu để chẩn đoán, nhưng P có alignment mơ hồ không tham gia tương quan thành phần. Với edit màu, P chỉ đánh giá cấu trúc. Dice ở đây so hai graph dự đoán, không phải object/relation F1 so graph GT.

## 5. Baseline ảnh và giao thức thống kê

**CLIPScore:** `2,5 × max(cos(CLIP(I_t), CLIP(caption)), 0)`, dùng `openai/clip-vit-base-patch32` đã khóa revision. Caption là mô tả đích mong muốn từ instruction và ngữ cảnh nguồn, thêm tiền tố `A photo depicts `; không caption chính ảnh Qwen để tự thưởng kết quả đã tạo. Processor resize rồi center crop 224×224; caption không bị truncate quá 77 token.

**LPIPS:** AlexNet v0.1, so source/target RGB chuẩn hóa [-1,1]; source resize LANCZOS về kích thước target, không crop. Điểm thấp thể hiện ảnh ít thay đổi, không xác nhận yêu cầu được thực hiện. Dùng **−LPIPS** trong bảng tương quan để cùng chiều “cao hơn tốt hơn”.

CLIP rồi LPIPS chạy tuần tự, FP32, batch 1, seed 42 trên RTX 5090; đủ 200 cặp có metric hợp lệ. Model/checkpoint và inference receipt được giữ, cùng [W&B của lượt metric](https://wandb.ai/keiranoluv-hcmus/level3-evaluation200-v1/runs/73zepw44).

So sánh chính: Spearman của G, CLIPScore và −LPIPS với nhãn tổng hợp **trên cùng tập giao có đủ điểm**. Không đặt điểm 0 cho mẫu abstain, không gán 0 cho tương quan của vector hằng. Bảng phụ ghi rõ mẫu số riêng.

Phân tích bất định bổ sung dùng paired cluster bootstrap theo ảnh nguồn: 2.000 lượt, seed 42, percentile 95%. Mỗi lượt giữ các edit của cùng nguồn cùng nhau và dùng cùng mẫu cho cả ba chỉ số. Không dùng p-value giả định 200 yêu cầu độc lập. Đây là hậu kiểm có điều kiện trên tập được chấm, không thay thế thực nghiệm nhiều seed.

## 6. Kết quả

### 6.1. Độ phủ

| Đại lượng | Số cặp |
|---|---:|
| Tổng đã có ảnh, graph và nhãn | 200 |
| E xác định | 162 |
| P đủ điều kiện đối chiếu nhãn | 109 |
| G xác định | **91 (45,5%)** |
| G abstain | **109 (54,5%)** |

Tập có G gồm 82 đạt/9 không đạt, thuộc 62 ảnh nguồn; tập abstain gồm 93 đạt/16 không đạt. Nguyên nhân chính là alignment phần bảo toàn mơ hồ ở 91 ca; các reason codes có thể cùng xuất hiện. Độ phủ không phải accuracy và không nên bỏ qua khi đọc hệ số tương quan.

### 6.2. So sánh chính trên cùng 91 cặp

| Chỉ số | Spearman | Bootstrap 95% |
|---|---:|---:|
| Graph G | **0,1767** | [−0,0304; 0,3571] |
| CLIPScore | 0,0561 | [−0,1245; 0,2233] |
| −LPIPS | **0,1962** | [−0,0178; 0,3908] |

G−CLIPScore có khoảng [−0,1795; 0,3964], G−(−LPIPS) có khoảng [−0,3179; 0,2659]. Cả hai chứa 0. Kết quả chưa chứng minh graph vượt baseline ảnh; cả ba hệ số đều yếu và bất định lớn.

Tương quan thành phần: **E với fulfillment = 0,2008 (N=162)**; **P với preservation = 0,0302 (N=109)**. Phần bảo toàn hiện đặc biệt yếu. Trên toàn bộ 200 cặp, bảng phụ cho CLIPScore = 0,0329 và −LPIPS = 0,2285; không so ngang hai số này với G chỉ có ở 91 cặp.

### 6.3. Theo loại chỉnh sửa

| Loại | Tổng | Có G | ρ G | ρ CLIPScore | ρ −LPIPS |
|---|---:|---:|---:|---:|---:|
| Thêm | 106 | 60 | 0,0753 | −0,0066 | 0,1744 |
| Xóa | 66 | 23 | 0,1271 | −0,0636 | 0,3814 |
| Thay | 18 | 8 | 0,2474 | −0,2474 | −0,5774 |
| Màu | 10 | 0 | Không xác định | Giao rỗng | Giao rỗng |

Trong mỗi hàng, ba chỉ số dùng cùng tập giao. Nhóm thay chỉ có 8 cặp, không đủ để kết luận xếp hạng mô hình ổn định. Nhóm màu giữ đầy đủ ảnh/nhãn và metric ảnh nhưng không tạo điểm G giả.

## 7. Phân tích 10 trường hợp bất đồng

Các case được chọn theo chênh lệch phân vị thứ hạng lớn nhất giữa G và mỗi baseline ảnh trên 91 cặp. Đây là bất đồng **xếp hạng**, không có ngưỡng tự động PASS/FAIL. Cả 10 thuộc nhóm thêm đối tượng; không phải mẫu ngẫu nhiên để đo độ chính xác SGG. Các quan sát đã đối chiếu ảnh thật và node/bbox/cạnh, không dùng để sửa nhãn hoặc scorer.

| Case | Nhãn người duyệt E/P | G | CLIPScore | LPIPS | Quan sát và giải thích |
|---|---|---:|---:|---:|---|
| #58 — thuyền buồm | Đạt/Đạt | 1,000 | 0,589 | 0,084 | Thuyền xuất hiện ở trái vùng nước, graph ghi nhận boat mới và nền giữ nguyên. Crop CLIP thực tế cắt mất hoàn toàn thuyền trong ảnh panorama. Đây là mất thông tin đầu vào, chưa lượng hóa riêng tác động lên điểm bằng ablation. |
| #69 — thuyền chèo | Đạt/Đạt | 1,000 | 0,612 | 0,105 | Thuyền chèo được thêm, núi và mặt nước gần như giữ nguyên. G/−LPIPS xếp cao, CLIP xếp thấp. Chưa xác định một nguyên nhân embedding duy nhất; graph cũng chỉ hiểu boat, không kiểm kiểu thuyền. |
| #178 — chim trên đá | Đạt/Đạt | 0,000 | 0,666 | 0,101 | Chim nhìn thấy trên đá; bbox tại đó bị gán person, confidence 0,361, không có bird. E=0 do extractor sai nhãn. LPIPS thấp nói về mức thay đổi nhỏ, không tự chứng minh chim đúng yêu cầu. |
| #168 — mèo ở gốc cây | Đạt/Không đạt | 1,000 | 0,652 | 0,297 | Có mèo nhưng chi tiết cành/lá/nền bị vẽ lại. Graph thô tree/grass/sky vẫn ghép nên P=1 quá lạc quan. Crop CLIP còn cắt mất mèo ở đáy ảnh. |
| #22 — chậu cây | Đạt/Đạt | 0,200 | 0,859 | 0,261 | Chậu cây mới có thật; tủ trắng còn nhưng mất detection refrigerator, wall đổi wall-brick. Chỉ pavement ghép được, làm P thấp vì nhãn/phát hiện nền không ổn định. |
| #188 — chim bên chùa | Không đạt/Đạt | 0,000 | 0,839 | 0,133 | Chim đậu mái thay vì lan can. CLIP/−LPIPS vẫn xếp cao. G=0 trùng nhãn không đạt nhưng do thiếu bird detection, không chứng minh scorer đã hiểu quan hệ vị trí sai. |
| #160 — chim trên đá lớn | Đạt/Đạt | 0,000 | 0,837 | 0,158 | Bird được nhận, confidence 0,509; tâm x≈0,6903 nằm ngoài ROI bắt đầu x=0,72. E=0 phản ánh giới hạn grounding hình chữ nhật cứng dù người duyệt chấp nhận vị trí. |
| #187 — chó bên thác | Đạt/Đạt | 0,333 | 0,861 | 0,310 | Chó xuất hiện đúng; graph nền thêm river/person/rock và mất cạnh khớp nên P thấp. Nước/đá cũng thật sự đổi kết cấu, giải thích LPIPS cao; mức thay đổi vẫn được người duyệt chấp nhận. |
| #198 — chim bên ao | Đạt/Đạt | 0,441 | 0,869 | 0,170 | Nguồn có train không thấy trong ảnh; đích thêm cả lan can lớn ở tiền cảnh. G thấp do cả detection sai lẫn thay đổi cấu trúc thật. Giữ nhãn người duyệt, không quy mọi chênh lệch cho extractor. |
| #48 — ghế gỗ | Đạt/Đạt | 0,875 | 0,564 | 0,262 | Ghế mới nằm đúng vùng và các node chính ghép được. Crop CLIP vẫn thấy ghế nên không thể viện mất vật thể do crop. Bề mặt gạch có thay đổi; nguyên nhân cụ thể của CLIP thấp chưa được tách bằng đối chứng. |

Ảnh và bằng chứng đầy đủ ở HTML phân tích được đóng trong gói; `reports/level3/evaluation200-v1/visual-review.json` giữ task ID, nhận xét và hash đầu vào. Ba preview đầu vào CLIP (#58/#168/#48) được tạo từ đúng processor đã khóa, không chạy lại CLIP hoặc thay điểm cũ.

## 8. Thảo luận và giới hạn

1. **Sai số SGG lan sang scorer.** Node biến mất có thể do detector, không phải vật thể bị xóa. Đổi nhãn và phân đoạn khác nhau làm P giảm dù cảnh ổn định. Confidence native chưa được hiệu chuẩn trên bộ đánh giá.
2. **Biểu diễn bỏ sót thay đổi.** Các nhãn/cạnh thô có thể giữ nguyên trong khi kết cấu và chi tiết nền đổi nhiều. Không có màu nên 10 case không thể được chấm E/G bằng extractor này.
3. **Grounding chưa phản ánh hết ngôn ngữ.** ROI tâm bbox là một xấp xỉ; điều kiện quan hệ, vật liệu, kích thước và tư thế có thể bị bỏ qua. Source-only authoring vẫn là công sức chuẩn bị của assistant, không phải suy luận end-to-end của mô hình đã huấn luyện.
4. **Baseline ảnh có giới hạn riêng.** CLIP center crop có thể làm mất vùng edit; LPIPS ưu tiên ít thay đổi, kể cả trường hợp không thực hiện yêu cầu. Cần đọc cả hai tiêu chí nhãn, không gọi một metric đơn lẻ là chất lượng ảnh toàn diện.
5. **Mẫu nhỏ và lệch nhãn.** Có 175/200 overall pass, chỉ 9 overall fail trong tập chung. Hai edit dùng chung nguồn. Bootstrap theo cụm xử lý phần phụ thuộc này nhưng không giải quyết missingness không ngẫu nhiên của G.
6. **Chưa có SGG ground truth và nhiều seed tạo ảnh.** Không báo object/relation F1 giả. Mẫu 10 ca bất đồng không thay kiểm định ngẫu nhiên. Seed42 là một lượt editor, chưa đáp ứng bằng chính nó yêu cầu nhiều seed chung của đề.

Phiên bản v1 nên được giữ làm mốc có provenance. Các cải tiến hợp lý gồm ghép đối tượng ổn định hơn, xử lý không chắc chắn của detection, kiểm quan hệ được yêu cầu và bổ sung thuộc tính. Mọi thay đổi cần protocol/version và tập phát triển được công bố; không sửa riêng ROI/nhãn của các ca test để tăng tương quan.

## 9. Tái lập và demo

Hai launcher sinh ảnh/graph độc lập có resume; xem [LEVEL3_PIPELINE.md](LEVEL3_PIPELINE.md). Bộ đánh giá giữ receipt theo case và report theo hash nhãn. Gói ZIP chứa code tại commit, ảnh/graph/native output, prompt, nhãn và điểm; checkpoint mô hình thị giác tải riêng theo asset lock. [Hướng dẫn backup/restore](LEVEL3_SUBMISSION.md) mô tả rõ điều kiện đường dẫn của launcher lịch sử.

Notebook mới `notebooks/level3_task3_demo.ipynb` đọc được ở workspace hoặc root bundle đã giải nén. Nó hiển thị source/target, bbox/ROI, graph có hướng, confidence, phép ghép và E/P/G; tính lại đủ 200 điểm graph cùng Spearman trên CPU. CLIPScore/LPIPS và dự đoán Task 2 được đọc từ kết quả đã chạy thật.

Phần Task 2 dùng riêng ca đúng/sai của GNN seed42 EMNLP: áp dụng predicted operations vào source để dựng predicted target. Chưa có adapter được kiểm chứng từ graph EMNLP sang REACT++; phần replay này không được trình bày thành một chuỗi Task 2 → Task 3 trên cùng ảnh.

## 10. Kết luận

Thử nghiệm đã hoàn tất một vòng đánh giá Task 3 có đủ số ảnh/yêu cầu, pretrained editor/SGG, nhãn tay, baseline ảnh, tương quan và 10 ca phân tích. Điểm graph hiện có ưu điểm là truy nguyên được bằng chứng node/cạnh nhưng độ phủ chỉ 45,5% và tương quan chưa vượt baseline. Kết quả cho thấy chất lượng và tính ổn định của graph, khả năng biểu diễn yêu cầu cùng cơ chế ghép đối tượng là những yếu tố quyết định; không thể suy chất lượng từ số node/edge hoặc một vài ảnh edit đẹp.

## Nguồn và khai báo hỗ trợ

- Đề bài nội bộ: `Project01_Graph_Editing.pdf`, Nhiệm vụ 3 và yêu cầu chia dữ liệu/đánh giá/sản phẩm nộp, trang 3–6.
- Nguồn ảnh: Sidewalk VQA và VietTravelVQA-v2/DOI ở mục 2; revision và ghi công trong data card.
- [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1); model/revision thực dùng nằm trong `configs/level3/qwen21-assets-v1.json`.
- [REACT++ PSG](https://huggingface.co/maelic/REACTPlusPlus_PSG), [SGG-Benchmark](https://github.com/Maelic/SGG-Benchmark); lock repository/model trong `configs/level3/sgg-pretrained-assets-v2.json`.
- [CLIPScore](https://github.com/jmhessel/clipscore), [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32), [LPIPS](https://github.com/richzhang/PerceptualSimilarity). Triển khai dùng adapter Hugging Face và protocol đã công bố, không tuyên bố bitwise tương đương mọi script gốc.
- Assistant/Codex hỗ trợ code, authoring instruction/grounding/caption, kiểm tra CPU, tổng hợp và nhận xét ảnh. Người dùng thực hiện/xác nhận nhãn chất lượng. Nhóm cần bổ sung tên thành viên và phân công thật khi ghép báo cáo chung; không tự gán người thực hiện chưa được xác nhận.
