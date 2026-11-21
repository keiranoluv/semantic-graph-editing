# Đối chiếu 20 câu lệnh MagicBrush với scene graph REACT++

Đánh giá định tính của trợ lý dựa trên node, cạnh và delta trong pairs.json. Rõ = thấy thay đổi chính ở đúng thực thể hoặc lớp khái quát phù hợp; Một phần = có dấu hiệu đúng nhưng thiếu hoặc sai thành phần chính; Thiếu = SG chưa cho phép xác nhận thay đổi yêu cầu. Đây là đánh giá theo câu lệnh, không chấm độ đúng của toàn bộ delta hay độ chính xác detector.

| ID | Câu lệnh | Mức | Bằng chứng từ SG | Số thao tác trong diff |
|---|---|---|---|---:|
| magicbrush:124157:1 | Remove the tin pans. | Một phần | Bốn bowl biến mất. SG không xác định đây là tin pan. | 90 |
| magicbrush:171050:1 | Make the hydrant red. | Thiếu | fire hydrant có ở cả hai phía, thiếu thuộc tính red. | 12 |
| magicbrush:171050:2 | Make all the bricks concrete. | Một phần | wall-brick biến mất, đích còn wall nhưng thiếu concrete. | 10 |
| magicbrush:21217:2 | change one of the apples for an corn cob | Một phần | Hai apple giảm còn một, thiếu corn ở đích. | 9 |
| magicbrush:21972:2 | add a kid looking at the plane | Rõ | Thêm person và looking at tới airplane đã có. SG biểu diễn người ở mức chung, thiếu tuổi. | 7 |
| magicbrush:270722:2 | make a boy sit on the bear | Một phần | Thêm person nhưng cạnh riding trỏ tới sheep. Hai bear nguồn đều mất. | 15 |
| magicbrush:286723:1 | Give the woman black pants. | Thiếu | Không có pants hoặc black. Delta chỉ thêm playingfield. | 1 |
| magicbrush:298692:1 | make that backpack all black | Thiếu | backpack tồn tại hai phía, không có black. | 9 |
| magicbrush:298692:2 | make the girl ginger | Thiếu | Hai SG giống nhau sau ghép, không có thuộc tính tóc. | 0 |
| magicbrush:321886:2 | let a cop point at the SUV | Thiếu | Đích có person và car nhưng thiếu quan hệ chỉ tay, vai trò cop và loại SUV. | 81 |
| magicbrush:338078:1 | Have there be a gorilla crossing the road | Một phần | Xuất hiện bear walking on road. Hành động gần yêu cầu, sai loài gorilla. | 9 |
| magicbrush:352276:1 | have the cow eat from a pile of carrots. | Thiếu | Giữ cow nhưng thiếu carrot và quan hệ ăn. | 4 |
| magicbrush:376047:1 | the refrigerator should be white. | Thiếu | Giữ refrigerator nhưng thiếu thuộc tính white. | 20 |
| magicbrush:378186:1 | The microwave should be white. | Thiếu | Microwave bị xóa rồi thêm lại sau ghép, thiếu white. | 31 |
| magicbrush:456569:2 | let a lion sleep near the giraffes | Thiếu | Thêm giraffe, thiếu lion ở đích. | 6 |
| magicbrush:477497:1 | take the almonds out of the box | Thiếu | Nguồn không nhận diện almond; mất bowl không xác nhận đã lấy hạt ra. | 17 |
| magicbrush:484847:1 | put a laptop on the table | Rõ | Thêm laptop và cạnh on tới table. Đích cũng có laptop on rug và on tới hai instance table. | 32 |
| magicbrush:540097:1 | remove things on the table | Một phần | Nguồn có cup, bottle, wine glass on dining table; một số biến mất. Bàn và ghế cũng bị xóa, không xác định được toàn bộ nhóm things. | 38 |
| magicbrush:71476:1 | Let the bear be calm. | Thiếu | Nguồn có cat và person, thiếu bear và thuộc tính calm. | 2 |
| magicbrush:9772:1 | Remove the TV screen. | Rõ | Có tv ở nguồn và mất tv ở đích. Delta còn thêm light và towel. | 15 |
