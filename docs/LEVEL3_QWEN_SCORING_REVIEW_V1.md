# Review assistant: Qwen và độ tin cậy của graph scorer

Bộ chấm đã chạy trên 100 cặp, 45 có G xác định và 55 abstain. Review dưới đây dùng 20 cặp chọn trước khi chấm, gồm tất cả 6 replace cùng các add/remove và trường hợp extractor thiếu đối tượng nguồn. Không dùng nhóm chẩn đoán này để suy ra tỷ lệ thành công của Qwen. Các nhận xét là của assistant, chưa phải GT hay nhãn người dùng.

Assistant xem source/target thật trước khi đọc điểm từng ca, sau đó đối chiếu node, cạnh có hướng, alignment và evidence của bộ chấm. Khi xem ảnh đã biết thống kê coverage toàn cục; đây không phải nghiên cứu đánh giá mù độc lập. Có mở thêm ảnh đầy đủ ở mẫu 13, 31, 82 và 96 để kiểm tra các chi tiết khó thấy trên contact sheet. Ảnh nguồn/đích và graph đều giữ nguyên; không sinh lại ảnh để minh họa nhận xét.

## Những ca cho thấy bộ chấm có ích

| Mẫu | Nhận xét từ ảnh | Bằng chứng graph |
|---|---|---|
| 9 · xóa ghế vàng | Ghế vàng mất, nhưng ghế xanh bên trái cũng mất ngoài yêu cầu. | Hai chair unmatched, chỉ anchor ghế vàng được miễn trừ; E=1, P=G≈0,755. Có phạt mất ghế xanh, song điểm trung bình vẫn tương đối cao; không coi là PASS. |
| 15 · ghế hồng → chậu cây | Thay đúng vùng, các vật chính khác có vẻ được giữ. | E=1, P=G≈0,912; hai kiểm tra remove/add cùng được hỗ trợ. |
| 58 · thêm thuyền buồm | Có thuyền với buồm trắng trên mặt nước phía trước; bố cục bờ còn. | E=P=G=1 với 4 node và 4 cạnh bảo toàn. Graph chỉ biết boat, không biết sailboat; điểm 1 chưa xác nhận mọi chi tiết prompt. |
| 73 · xóa người cạnh thuyền | Xóa cả người đội nón và người áo đỏ/xanh bên cạnh. | Anchor n004 và person n003 ngoài yêu cầu cùng unmatched; E=1, P≈0,705, G=null do một person khác ghép mơ hồ. |

Các ca này cho thấy cần tách **thực hiện thay đổi** và **bảo toàn**. E=1 không có nghĩa ảnh hoàn toàn đúng, và phần node/cạnh mất ngoài yêu cầu có ích hơn chỉ nhìn tổng số lượng node/edge.

## Những ca graph chưa đủ tin cậy

- **Mẫu 10 và 13:** ảnh cho thấy xe đạp/ghế đẩu đã được xóa, nhưng graph nguồn không có đối tượng đó nên E/G=null. Ở mẫu 13, cardboard xuất hiện thành node mới ở target mặc dù tấm bìa đã tồn tại trong ảnh nguồn. Đối tượng mới trong graph có thể chỉ là detection mới.
- **Mẫu 31:** ảnh nguồn có hai xe máy chồng nhau; Qwen thay bằng một xe đạp và làm mất cả xe bên cạnh. Source graph chỉ có một motorcycle node nên không thể nhận biết đầy đủ lỗi bảo toàn instance thứ hai. Đây là lỗi thiếu thông tin trước cả bước ghép.
- **Mẫu 54:** ảnh xóa xe tải có vẻ hợp lý và giữ bố cục tòa nhà, nhưng G≈0,569 vì nhiều road/gravel predictions không ghép được. Xóa vật lớn làm lộ thêm nền; chưa có cơ chế xử lý occlusion/disocclusion bằng mask.
- **Mẫu 71:** mèo được thêm đúng cạnh cửa, nhưng texture tường/lá/ánh sáng thay đổi nên bảo toàn ảnh còn chưa chắc. G=0,25 chủ yếu do graph nguồn chỉ có wall, còn target nhận thêm tree/door vốn đã có trong ảnh. Không thể dùng điểm này để định lượng mức hư hại texture.
- **Mẫu 82:** ảnh cho thấy hai ghế xanh bị xóa dù chỉ yêu cầu ghế ở dưới bên phải. Graph nguồn không có stool; các nhãn tie/umbrella/teddy bear cũng không mô tả đáng tin cảnh dệt. Scorer không có cơ sở để chấm lỗi ghế này.
- **Mẫu 96:** person mới nằm trong vùng 2D xấp xỉ nên E=1, nhưng ảnh chưa làm rõ người có đứng trên central path hay ở rìa lối đi trước chậu cây. Graph đồng thời có walking on và sitting on với các nền khác nhau. ROI không thay thế hiểu quan hệ không gian đầy đủ.

Mẫu 1 cũng cho thấy cạnh suy đoán không chắc: người bán vẫn ngồi cạnh bàn, nhưng graph target có person → riding → bicycle. Vì thế graph có nội dung ngữ nghĩa hữu ích, song không thể xem mọi relation như sự thật chỉ vì parse thành công.

## Hướng sử dụng cho Task 3

V1 phù hợp làm công cụ chẩn đoán: lưu được cặp ghép, đối tượng/cạnh mất ngoài yêu cầu, điều kiện edit, giới hạn và các ca chưa thể kết luận. Chưa đủ căn cứ dùng G làm bộ quyết định chất lượng cuối cùng. Nút thắt đang thấy gồm thiếu instance ở nguồn, detection thừa/chồng nhau, relation sai hoặc đổi giữa hai ảnh, và phần prompt không được vocabulary/head của REACT++ biểu diễn.

Ưu tiên tiếp theo là người dùng duyệt nhãn hai trục độc lập, sau đó kiểm tra coverage và tương quan trên đúng các mẫu có nhãn; giữ cả các ca abstain trong báo cáo. Một phiên bản cải tiến có thể dùng bằng chứng appearance/context khi ghép và xử lý che khuất, nhưng cần thí nghiệm riêng. Không chỉnh ngưỡng theo 20 review này rồi báo lại cùng tập như kết quả khách quan. CLIPScore/LPIPS và các yêu cầu còn lại của đề vẫn cần làm tiếp.

Toàn bộ 20 nhận xét, không chỉ ví dụ trong tài liệu, nằm ở [assistant-review.json](../reports/level3/qwen-graph-score-v1/assistant-review.json) và dashboard `runs/level3/qwen-graph-score-v1/review-v1/comparison.html`. Các lựa chọn người dùng trong giao diện hai trường vẫn độc lập và chưa được scorer đọc.
