# 5 hit và 5 missed — mẫu phân tích ban đầu

Nguồn: checkpoint **GAT baseline seed 42**, validation pool 100 ảnh, selection
seed 2026. Script `scripts.extract_level1_examples` xáo thứ tự ảnh bằng seed và
lấy tối đa một GT triplet đầu tiên cho mỗi loại trên mỗi ảnh. Đây là ứng viên
để nhóm xem và giải thích; **chưa phải phân tích GAT+KG cuối**. `hit` nghĩa GT
triplet nằm trong top 50; `missed` nghĩa không nằm trong top 50 theo cùng
PredCls evaluator. Các đối tượng và quan hệ bên dưới lấy từ annotation;
chưa kiểm tra nội dung JPEG vì host ảnh không truy cập được ở môi trường hiện tại.

## Năm triplet được dự đoán đúng ở R@50

| Image ID | GT triplet | Model | Hạng | Nhận xét kỹ thuật |
|---:|---|---|---:|---|
| 40 | flower — in — vase | in | 1 | Khớp nhãn GT và đứng đầu ranking ảnh. |
| 7 | woman — sitting on — chair | sitting on | 7 | Khớp predicate hành động/tư thế trong top 50. |
| 89 | door — on — table | on | 43 | Khớp annotation, nhưng ở sát ngưỡng top 50 hơn các ví dụ khác. Cần xem ảnh để xác nhận ngữ cảnh. |
| 81 | door — on — cabinet | on | 35 | Khớp annotation ở top 50; cần xem ảnh để phân biệt vật thể. |
| 3 | girl — has — hair | has | 5 | Khớp quan hệ thuộc tính đối tượng trong annotation. |

## Năm GT triplet bị bỏ sót ở R@50

| Image ID | GT triplet | Model | Hạng cặp | Dấu hiệu cần xem xét |
|---:|---|---|---:|---|
| 40 | table — near — chair | with | 21 | Predicate của cùng cặp khác GT: `near` vs `with`. |
| 7 | laptop — above — desk | on | 1 | Khác biệt `above` vs `on`; cần xem ảnh để biết đây là lỗi mô hình hay quan hệ gần nghĩa trong nhãn. |
| 89 | shelf — on — table | on | 82 | Predicate đúng nhưng cặp nằm sau top 50: lỗi ranking theo R@50. |
| 81 | door — for — cabinet | on | 40 | Khác biệt quan hệ mục đích `for` và vị trí `on`. |
| 3 | girl — with — hair | has | 5 | `with` và `has` có thể gần nghĩa trong ngữ cảnh này; evaluator exact-match vẫn tính missed. |

Các cột “model” và “hạng cặp” dựa trên foreground predicate top-1 cho đúng
ordered pair của GT, với rank tính trên mọi candidate pairs trong ảnh.
Một triplet missed có thể vì chọn predicate khác hoặc vì rank ngoài 50.
Những nhận xét trên chỉ mô tả sự khác biệt giữa dự đoán và annotation;
không suy đoán vật thể/vị trí thật trong ảnh khi chưa xem JPEG.

Để hoàn thiện yêu cầu báo cáo, nhóm cần xem từng ảnh, xác nhận object boxes,
ghi nguyên nhân bằng ngôn ngữ của mình và lặp lại trên model GAT+KG được chọn
sau khi train 3 seeds. Dùng `human_analysis` trong JSON output của script làm
nơi ghi chú trước khi chuyển sang báo cáo. Không nên chọn case dựa riêng trên
việc chúng làm kết quả trông tốt hơn; ghi rõ selection seed và pool ảnh.
