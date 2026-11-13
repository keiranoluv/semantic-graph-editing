# Pilot grounded v2 không tạo được nhãn: chẩn đoán CPU

Kết luận từ cache đã chạy: **0/20 cặp có thao tác; 0/20 qua bộ lọc**.
18 cặp có giao dịch rỗng và trạng thái `needs_review`; 2 cặp không có giao dịch
do không grounding được object nguồn. Đây là **độ phủ gán nhãn bằng 0 trên
pilot này**, không phải phép đo accuracy của toàn MagicBrush.

Người dùng chỉ ra 5 cặp bỏ sót thay đổi thật. Assistant đã xem cả ảnh nguồn/đích
của 5 cặp, đối chiếu cached detector/CLIP và code. Không tạo CSV chấm thay người
dùng, không tự đánh dấu cả 20 cặp là đã được người kiểm, không approve nhãn.

| Cặp | Edit chính cần biểu diễn theo phản hồi | Bằng chứng về chỗ pipeline bị chặn |
|---|---|---|
| Tree → piñata | replace_node | Source có `christmas tree`, nhưng code tìm chính xác `tree`; score target `pinata` 0.251 dưới 0.45. |
| Add a pear | add_node | Source 1 và target 2 detection `pear`; code yêu cầu target chỉ có đúng một instance và lớp mới vắng toàn ảnh nguồn. Điều kiện này không hỗ trợ thêm instance cùng lớp. Detection cũ cũng có thể sai nhãn. |
| Chair → yellow | replace_node thuộc tính | Box IoU 0.872 nhưng crop cosine 0.785 < 0.85 nên không ghép cùng chiếc ghế; màu nguồn/đích đều bị từ chối bởi CLIP-margin/HSV. |
| Cat → hedgehog | replace_node | Cat source 0.633, hedgehog target 0.597 đã được phát hiện. Score hedgehog ở source 0.298 > 0.15; box IoU 0.364 < 0.5; contrast lớp mới 0.032 < 0.035. Đổi kích thước object hợp lệ bị chặn. |
| Dog → balloon | replace_node, ontology lớp/material cần quy định | Target score balloon 0.351 < 0.45 nên thiếu node đích; source balloon 0.183 cũng vượt điều kiện “vắng” 0.15. |

Những số trên giải thích cơ chế bị reject; **không phải đề xuất hạ các ngưỡng
đến vừa 5 ca**. 5 ca này đã trở thành tập chẩn đoán, không thể dùng lại như mẫu
độc lập để tuyên bố độ tin cậy sau khi tune.

## Lỗi thiết kế cần xử lý

Policy bắt đầu target bằng bản sao graph nguồn, chỉ sửa khi mọi điều kiện cục
bộ thỏa. Khi reject, nó vẫn xuất target chưa đổi và `operations=[]`. Viewer cũ
hiện đồ thị này giống một dự đoán no-op, dù `eligible_for_training=false`.
**Chưa xác minh được edit phải là nhãn chưa gán/abstention**, không được diễn giải
là ground truth “không thay đổi”. `target_reconstruction_exact=true` chỉ nói
giao dịch tái tạo được graph do code dựng, không xác minh ngữ nghĩa ảnh.

Các vấn đề độc lập:

- Ghép/reference theo đúng chuỗi lớp bỏ sót tên bao quát/cụ thể.
- “Thêm object” bị hiểu thành thêm lớp hoàn toàn mới, thay vì thêm instance.
- Matching phụ thuộc độ giống appearance ngay cả khi yêu cầu đổi appearance.
- Ngưỡng box overlap cứng chặn replacement làm object nhỏ hơn.
- CLIP crop và HSV trên cả bbox không đủ để xác nhận màu của riêng object.
- Context bbox có thể chứa cả vùng edit. Kiểm ổn định toàn crop có thể reject
  đúng thay đổi cục bộ thay vì chỉ đánh giá phần ngữ cảnh ngoài vùng edit.

## Quyết định sau audit

Giữ nguyên dữ liệu và bằng chứng, **không dùng bộ này để train hoặc chạy full**.
Không cần hoàn thành thêm review chỉ để xác nhận lỗi thiếu edit đã được cache
chứng minh. Độ tin cậy human của dataset đầy đủ vẫn chưa được đo.

Theo PDF trang 2–5: cấp độ 2 cần ≥300 tuple có thao tác đích, kiểm tay ≥20 mẫu
ngẫu nhiên, báo độ tin cậy và train/evaluate mô hình dự đoán 6 loại thao tác.
Pilot hiện tại chưa đáp ứng phần dữ liệu; số test CPU không thay thế chất lượng
nhãn. Graph cục bộ chỉ có spatial relations cũng cần được công bố phạm vi.

Thiết kế tiếp theo phải tách **quan sát graph/detection ảnh đích**, **đề xuất edit
theo correspondence từng instance** và **quyết định accept/defer**. Giữ nguyên
quan sát thô khi reject; không ghi graph nguồn thành đáp án ảnh đích. Các thay
đổi về correspondence/appearance cần kiểm bằng bằng chứng thực, rồi dùng mẫu
train độc lập mới để đánh giá chất lượng. Không tự sinh đáp án chỉ từ câu yêu cầu.

Chưa triển khai hay yêu cầu chạy GPU cho một phương pháp gán nhãn mới trong
audit này. Cần giải quyết các lỗi thiết kế trên trước khi khóa recipe tiếp theo.

## Tái tạo

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 myenv/bin/python -m tools.audit_magicbrush_grounded_v2
```

CPU-only. `audit.json` ghi 20 ca và hash đầu vào; diagnostic HTML nằm ở
`reviews/magicbrush-grounded-diagnostics-v1/index.html`. Mở trực tiếp trong browser.
Trang này vẽ **box detector nguồn và đích thực đã cache**, đồng thời ghi rõ nhãn
chưa gán; không hiển thị graph nguồn sao chép như đáp án của ảnh đích. Không có
nút approve hay CSV review tự điền. Các viewer, CSV và packet cũ được giữ nguyên.
