# Audit 20 cặp validation: GNN/42

Mẫu ngẫu nhiên cố định seed 20261114; đây là phân tích của assistant, không thay thế kiểm tra tay của người dùng. Mọi trường hợp đều sai toàn bộ tập thao tác; các thao tác đúng một phần được nêu rõ. Chi tiết graph/thao tác trong `cases.json`.

## 1. magicbrush:124157:1

Yêu cầu: **Remove the tin pans.**

[Ảnh nguồn](/workspace/workspace/semantic-graph-editing/data/processed/level2/magicbrush-v1/images/f8aaf8d1de8bf4de0c9619ddb505b0d92044da26b7188d6e6f635ef736cacfe1.png) · [Ảnh đích](/workspace/workspace/semantic-graph-editing/data/processed/level2/magicbrush-v1/images/27ab4d0baddd3340dcb08955bc95214cacad8fb105a2aa5b0c0669a2d2de21c3.png)

Ảnh đích bỏ hai khay; chuối còn nguyên. Nhãn xóa khay và chuyển quan hệ chuối lên bàn phù hợp thay đổi nhìn thấy.

Mô hình xóa bananas n001 thay vì foil pans n002; đây là sai đối tượng, không phải chỉ khác cách ghi thuộc tính.

Mức tin cậy nhận định: high. Thao tác khớp chính xác: 1/4; dự đoán 2 thao tác.

## 2. magicbrush:171050:1

Yêu cầu: **Make the hydrant red.**

[Ảnh nguồn](/workspace/workspace/semantic-graph-editing/data/processed/level2/magicbrush-v1/images/6e578a1a2cf28837b002b416144657a8e30484747b92c013b20512c032186acd.png) · [Ảnh đích](/workspace/workspace/semantic-graph-editing/data/processed/level2/magicbrush-v1/images/1aac2a47f8c4cdd87028c0f9b4abe105fdfd6419210d16768766ee57f80917cb.png)

Yêu cầu đổi màu đỏ, nhưng ảnh đích cũng thay hình dáng trụ: thân nhỏ, nắp khác, có dây xích. Nhãn phản ánh cả các chi tiết ngoài câu yêu cầu.

Chọn đúng replace_node cho hydrant nhưng sinh brown/white red, giữ wide, và thêm car; lỗi nội dung thật sự dù có bỏ chi tiết phụ.

Mức tin cậy nhận định: high. Thao tác khớp chính xác: 0/1; dự đoán 3 thao tác.

## 3. magicbrush:171050:2

Yêu cầu: **Make all the bricks concrete.**

[Ảnh nguồn](/workspace/workspace/semantic-graph-editing/data/processed/level2/magicbrush-v1/images/1aac2a47f8c4cdd87028c0f9b4abe105fdfd6419210d16768766ee57f80917cb.png) · [Ảnh đích](/workspace/workspace/semantic-graph-editing/data/processed/level2/magicbrush-v1/images/cecc2ed842d6c1d0bb00ed527905a1d9b1749c913468a94a539157b64da34578.png)

Tường gạch thành bê tông; vùng trái trên xuất hiện một phần cửa sổ. Add window là thay đổi quan sát được nhưng không được yêu cầu rõ.

Thêm dog và quan hệ on sidewalk, không sửa wall; sai mục tiêu chính.

Mức tin cậy nhận định: high. Thao tác khớp chính xác: 0/3; dự đoán 2 thao tác.

## 4. magicbrush:21217:2

Yêu cầu: **change one of the apples for an corn cob**

[Ảnh nguồn](/workspace/workspace/semantic-graph-editing/data/processed/level2/magicbrush-v1/images/5f657db704f17e292539ad868c3107ef0fd187d31243a849e00915e8bf61b86a.png) · [Ảnh đích](/workspace/workspace/semantic-graph-editing/data/processed/level2/magicbrush-v1/images/379c86a47534950a19b5b46abee2e925760ced41570a8f25f81be09ff8a40891.png)

Câu one of the apples không chọn trái/phải; ảnh chọn táo bên phải. Partly cropped mô tả bố cục của bắp trong ảnh đích, không có trong câu lệnh.

Mô hình sửa cereal thay vì một trong hai apple; lỗi chọn đối tượng vượt quá vấn đề tham chiếu mơ hồ.

Mức tin cậy nhận định: high. Thao tác khớp chính xác: 0/1; dự đoán 1 thao tác.

## 5. magicbrush:21972:2

Yêu cầu: **add a kid looking at the plane**

[Ảnh nguồn](/workspace/workspace/semantic-graph-editing/data/processed/level2/magicbrush-v1/images/eda9220eb431d465216e546a3a76fc5c97fdba361e6963f9455f17806bb9f3a4.png) · [Ảnh đích](/workspace/workspace/semantic-graph-editing/data/processed/level2/magicbrush-v1/images/b2db9dd4246c3172a0c1736d66790ae1246f2b3e24cda7fa0c2250cdd4b2cf06.png)

Đứa trẻ đứng quay lưng về máy bay là phù hợp yêu cầu; tóc vàng và áo nâu là lựa chọn cụ thể của ảnh đích.

Mô hình thêm duck và quan hệ on airplane; sai nhãn và quan hệ, không chỉ thiếu màu áo.

Mức tin cậy nhận định: high. Thao tác khớp chính xác: 0/3; dự đoán 2 thao tác.

## 6. magicbrush:270722:2

Yêu cầu: **make a boy sit on the bear**

[Ảnh nguồn](/workspace/workspace/semantic-graph-editing/data/processed/level2/magicbrush-v1/images/8533590534412b098723ff23234b6f3f300c6a39136a58c095788b7939564b90.png) · [Ảnh đích](/workspace/workspace/semantic-graph-editing/data/processed/level2/magicbrush-v1/images/22587bcd886bbf886745ca496d0a5fe912c653671800ad1e20162668a0290abf.png)

Có trẻ ngồi trên gấu. Tóc vàng, áo hồng và quần xám là chi tiết bổ sung; từ ảnh thu nhỏ không xác nhận riêng từng chi tiết bàn chân.

Thêm monkey, sửa thuộc tính gấu thành mustache/yellow nose, cạnh tự nối; nhiều lỗi ngữ nghĩa không phụ thuộc các chi tiết nhãn khó.

Mức tin cậy nhận định: medium. Thao tác khớp chính xác: 0/2; dự đoán 3 thao tác.

## 7. magicbrush:286723:1

Yêu cầu: **Give the woman black pants.**

[Ảnh nguồn](/workspace/workspace/semantic-graph-editing/data/processed/level2/magicbrush-v1/images/4fb812a073dacf894640d9a662c67ba016e5e49769f6d4b75d608fcabd0acd8a.png) · [Ảnh đích](/workspace/workspace/semantic-graph-editing/data/processed/level2/magicbrush-v1/images/80b6f6a1adfd9f45dd90c637b815f267a0123640238a6c0c709a83126d64e4f9.png)

Quần chuyển thành quần dài đen; có mảng áo tím ở eo. Việc gọi mảng này là một undershirt mới cần quy tắc gán thực thể khi vùng áo/quần thay đổi.

Sửa đúng vị trí shorts nhưng không đổi label thành pants, sinh white icing; còn sửa shirt không được yêu cầu.

Mức tin cậy nhận định: medium. Thao tác khớp chính xác: 0/3; dự đoán 2 thao tác.

## 8. magicbrush:298692:1

Yêu cầu: **make that backpack all black**

[Ảnh nguồn](/workspace/workspace/semantic-graph-editing/data/processed/level2/magicbrush-v1/images/203ff8c678db6d935ae6c115e1f5b79534a0d9f744e51d0abbe3a8a6d7c2d658.png) · [Ảnh đích](/workspace/workspace/semantic-graph-editing/data/processed/level2/magicbrush-v1/images/ea4b71bc72059bcdf968634460a897d4e1956c2d1b2c6efd1531685f1e6c328a.png)

Ba lô đen trơn thay ba lô có hình in; nhãn black/plain bám khá trực tiếp yêu cầu all black.

Chọn đúng ba lô nhưng sinh black red, đồng thời sửa girl và thêm baseball cap; có lỗi ngay ở mẫu tương đối rõ.

Mức tin cậy nhận định: high. Thao tác khớp chính xác: 0/1; dự đoán 4 thao tác.

## 9. magicbrush:298692:2

Yêu cầu: **make the girl ginger**

[Ảnh nguồn](/workspace/workspace/semantic-graph-editing/data/processed/level2/magicbrush-v1/images/ea4b71bc72059bcdf968634460a897d4e1956c2d1b2c6efd1531685f1e6c328a.png) · [Ảnh đích](/workspace/workspace/semantic-graph-editing/data/processed/level2/magicbrush-v1/images/ba3391e63b1c66d69c69f5216b99dcd36b2e39ad09e3892155e10f6fb0ea22ec.png)

Tóc đổi màu và kiểu từ buộc đuôi sang bob. Việc thêm person phía trong xe còn chịu mơ hồ do che khuất; cần xem quy tắc hiện diện, chưa kết luận nhãn person đúng hay sai.

Chọn đúng girl nhưng sinh teeth visible/white face, không đổi tóc đúng; việc có thêm person trong nhãn không giải thích lỗi chính này.

Mức tin cậy nhận định: medium. Thao tác khớp chính xác: 0/4; dự đoán 1 thao tác.

## 10. magicbrush:321886:2

Yêu cầu: **let a cop point at the SUV**

[Ảnh nguồn](/workspace/workspace/semantic-graph-editing/data/processed/level2/magicbrush-v1/images/08f93fd3409414c4adfb964eb1ea1cb25e36a8ea4b899eb5304fe1d4d6e2aee8.png) · [Ảnh đích](/workspace/workspace/semantic-graph-editing/data/processed/level2/magicbrush-v1/images/5e1697872ed1a860542d93e8681672618a25a426210f5cdc6d6692683a4d9067.png)

Ảnh thêm người mặc đồ phản quang ở vị trí người đi xe đạp phía trái. Người cũ bị thay/che khuất; quần áo cụ thể và việc xóa cyclist không xác định duy nhất từ câu yêu cầu.

Không thêm người, chỉ thêm cạnh intersection on intersection; không thực hiện chỉnh sửa chính.

Mức tin cậy nhận định: high. Thao tác khớp chính xác: 0/5; dự đoán 1 thao tác.

## 11. magicbrush:338078:1

Yêu cầu: **Have there be a gorilla crossing the road**

[Ảnh nguồn](/workspace/workspace/semantic-graph-editing/data/processed/level2/magicbrush-v1/images/ab66c6e4e4272da48e00c5413cf5d02dd4f2e3eaa771ea108eab64e1139640ac.png) · [Ảnh đích](/workspace/workspace/semantic-graph-editing/data/processed/level2/magicbrush-v1/images/50c63bbf505c46c6f4433200875f16f9b562c7f7241d543fd5a19760469d7345.png)

Có gorilla trên đường; hướng facing camera không được chỉ định trong crossing the road.

Mô hình thêm duck. Cạnh on road vẫn được bộ chấm tính đúng cấu trúc do cùng ID vị trí mới; điều đó không có nghĩa thực thể đã đúng.

Mức tin cậy nhận định: high. Thao tác khớp chính xác: 1/2; dự đoán 2 thao tác.

## 12. magicbrush:352276:1

Yêu cầu: **have the cow eat from a pile of carrots.**

[Ảnh nguồn](/workspace/workspace/semantic-graph-editing/data/processed/level2/magicbrush-v1/images/865a5bd7e2534cc2d3be5136520c04157691cbf59d1badf5eb4fdccb4e9c13af.png) · [Ảnh đích](/workspace/workspace/semantic-graph-editing/data/processed/level2/magicbrush-v1/images/841eefc85c38859551949be8be3f320ddce8ed7606a92a9112ed03499d1932bd.png)

Cà rốt xuất hiện gần đầu bò trên mặt đất. Nhãn dùng near và on, chưa biểu diễn hành động eating; ảnh tĩnh không đủ kết luận bò đang ăn.

Mô hình thêm lion ở vị trí khác; sai nhãn ngay cả khi chỉ yêu cầu biểu diễn vật thể và quan hệ nhìn thấy.

Mức tin cậy nhận định: medium. Thao tác khớp chính xác: 0/3; dự đoán 2 thao tác.

## 13. magicbrush:376047:1

Yêu cầu: **the refrigerator should be white.**

[Ảnh nguồn](/workspace/workspace/semantic-graph-editing/data/processed/level2/magicbrush-v1/images/843ab0d0a76f1245c91f0b9c0d277f200fa5766eb2a063e7fae80e808c5fddf8.png) · [Ảnh đích](/workspace/workspace/semantic-graph-editing/data/processed/level2/magicbrush-v1/images/e1a2b65d6023f85e3cc0e282e5d57b2270547e9e4c1cc1a33d19f9589f65d45c.png)

Ảnh đổi tủ lạnh sang trắng nhưng cũng đổi đồ trên nóc, mất áo đỏ tiền cảnh và đổi số ngăn/giấy dán. Nhãn có 8 thao tác, vượt thay đổi màu được nêu trong câu yêu cầu.

Mô hình chọn đúng tủ lạnh nhưng sinh white red và thêm cream; nhãn phụ khó dự đoán là một vấn đề riêng với lỗi này.

Mức tin cậy nhận định: high. Thao tác khớp chính xác: 1/8; dự đoán 3 thao tác.

## 14. magicbrush:378186:1

Yêu cầu: **The microwave should be white.**

[Ảnh nguồn](/workspace/workspace/semantic-graph-editing/data/processed/level2/magicbrush-v1/images/9bf001b640cecc0a7b16a6ddcbcea15b780d975dfa2574d862b7a87944aeab03.png) · [Ảnh đích](/workspace/workspace/semantic-graph-editing/data/processed/level2/magicbrush-v1/images/9fc999657b692694d51a978cfff1c91be3b94888c2efde8eab98f6bf9aa122b7.png)

Lò vi sóng thành trắng đồng thời nhỏ đi; compact là thay đổi thực trong ảnh đích nhưng không bắt buộc bởi yêu cầu đổi màu.

Chọn đúng microwave, giữ wide và sinh white icing. Có dấu hiệu lẫn cụm thuộc tính giữa các loại vật thể.

Mức tin cậy nhận định: high. Thao tác khớp chính xác: 0/1; dự đoán 1 thao tác.

## 15. magicbrush:456569:2

Yêu cầu: **let a lion sleep near the giraffes**

[Ảnh nguồn](/workspace/workspace/semantic-graph-editing/data/processed/level2/magicbrush-v1/images/1e9fc5a073866dbb01277eb2f82ac58f9095f7e84703b5526a59c9e95ba18501.png) · [Ảnh đích](/workspace/workspace/semantic-graph-editing/data/processed/level2/magicbrush-v1/images/78ec8876ea9d6226026edeb94b709207f364d1f01eebbdc2fbde39ed04a4ff8c.png)

Sư tử nằm phía sau con hươu bên phải; vị trí right và chọn đúng hươu cụ thể không được chỉ định bởi near the giraffes.

Mô hình chọn đúng label lion nhưng attrs brown/dark khác lying/right; cạnh on ground đúng, cạnh lion on giraffe sai. Không được gọi cả mẫu là dự đoán đúng.

Mức tin cậy nhận định: high. Thao tác khớp chính xác: 1/3; dự đoán 3 thao tác.

## 16. magicbrush:477497:1

Yêu cầu: **take the almonds out of the box**

[Ảnh nguồn](/workspace/workspace/semantic-graph-editing/data/processed/level2/magicbrush-v1/images/d9e738f6fd8135394f72104bd5f0f260c7c22fc6bdf78335da5310843c418429.png) · [Ảnh đích](/workspace/workspace/semantic-graph-editing/data/processed/level2/magicbrush-v1/images/327c6a1b3c5509f1ecfbdb51d5b7bf903b6b1aaec81a5e516c1b2e6f738f3f25.png)

Ảnh bỏ cả hạnh nhân lẫn cốc xanh chứa chúng. Việc bỏ cốc không được nói rõ trong take the almonds out.

Mô hình xóa đúng almonds và cạnh in cup, giữ lại cup. Đây là ví dụ dự đoán đúng phần yêu cầu chính nhưng thiếu thay đổi phụ so với ảnh đích.

Mức tin cậy nhận định: high. Thao tác khớp chính xác: 2/4; dự đoán 2 thao tác.

## 17. magicbrush:484847:1

Yêu cầu: **put a laptop on the table**

[Ảnh nguồn](/workspace/workspace/semantic-graph-editing/data/processed/level2/magicbrush-v1/images/41e956f773071f23883a03d9aaf3fba40301a1eb243f19482bcd020afaff2334.png) · [Ảnh đích](/workspace/workspace/semantic-graph-editing/data/processed/level2/magicbrush-v1/images/872aa3f4eaffdad1ffec0ab74df86cacd585e286da413de755482aeaac6a9952.png)

Laptop được đặt lên bàn và sách/remote trên bàn biến mất; màu máy và màn hình xanh là chi tiết ảnh đích.

Mô hình thêm dog lên bàn: quan hệ đặt lên bàn đúng dạng nhưng vật thể sai; không thể quy toàn bộ lỗi cho đồ vật biến mất.

Mức tin cậy nhận định: high. Thao tác khớp chính xác: 1/6; dự đoán 2 thao tác.

## 18. magicbrush:540097:1

Yêu cầu: **remove things on the table**

[Ảnh nguồn](/workspace/workspace/semantic-graph-editing/data/processed/level2/magicbrush-v1/images/0edbe943fdf9b997905815f1265e542d2e2d179b42812b0086ce4c74363f47a8.png) · [Ảnh đích](/workspace/workspace/semantic-graph-editing/data/processed/level2/magicbrush-v1/images/c88bf34bb86975fb8171afa6becbbf28e88ffaffd9fac695d6faffaaeed7fdf7.png)

Ảnh dọn đồ trên cả bàn giữa và bàn sau; câu the table có nhiều ứng viên trong nguồn. Nhãn tả thay đổi đồ vật nhìn thấy.

Xóa đúng hai condiment set nhưng bỏ sót papers, xóa thêm chairs/urn, thêm chair và cạnh tự nối; lỗi phạm vi chỉnh sửa rõ.

Mức tin cậy nhận định: high. Thao tác khớp chính xác: 4/6; dự đoán 10 thao tác.

## 19. magicbrush:71476:1

Yêu cầu: **Let the bear be calm.**

[Ảnh nguồn](/workspace/workspace/semantic-graph-editing/data/processed/level2/magicbrush-v1/images/ae787004d08b07b808e6311745fa3e73e73e01ca94ae1b0bfeb94e1a6716771b.png) · [Ảnh đích](/workspace/workspace/semantic-graph-editing/data/processed/level2/magicbrush-v1/images/8c41f54420b3b2b3cefed10bdf75be754064a618cce48d446f173e50cec41dfc.png)

Miệng đóng phù hợp calm; mõm nhạt màu/đường nét mới là chi tiết hình ảnh bổ sung.

Mô hình giữ ý miệng mở/răng lộ với cả mouth open, teeth visible và visible teeth, thêm elephant; các cụm gần đồng nghĩa làm schema phân mảnh nhưng dự đoán còn ngược yêu cầu.

Mức tin cậy nhận định: high. Thao tác khớp chính xác: 0/1; dự đoán 2 thao tác.

## 20. magicbrush:9772:1

Yêu cầu: **Remove the TV screen.**

[Ảnh nguồn](/workspace/workspace/semantic-graph-editing/data/processed/level2/magicbrush-v1/images/a5b609161adc18ad24f4edadaa3ec4daf4ab7dc57f3bef297389d6ab35be42a6.png) · [Ảnh đích](/workspace/workspace/semantic-graph-editing/data/processed/level2/magicbrush-v1/images/ea88c48105246d13a7e939126a67bbd8835e3864133b3481725a38e47cf03578.png)

TV biến mất, phần còn lại nhìn chung giữ nguyên; nhãn hai thao tác xóa TV và cạnh above là rõ.

Mô hình xóa đúng TV và cạnh đó nhưng xóa thêm person reflection và nhiều cạnh khác; thất bại bảo toàn phần không liên quan.

Mức tin cậy nhận định: high. Thao tác khớp chính xác: 2/2; dự đoán 6 thao tác.
