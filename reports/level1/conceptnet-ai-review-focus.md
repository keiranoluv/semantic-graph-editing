# ConceptNet — AI review v2 cho 27 nhãn từng `defer`

Bảng này lưu bản đề xuất AI v2 để đối chiếu. Sau đó Toan đã duyệt 134 đề xuất ngoài focus và quyết định trực tiếp 16 nhãn focus: accept ID 11, 54, 123, 143; reject 12 ID khác. Bản chính thức là `data/knowledge/conceptnet-review-toan-v3/review.csv` (138 accept, 12 reject, 0 pending). Nghĩa VG150 tham chiếu WordNet representative mapping của Toan.

## 13 nhãn có thể đề xuất `accept`

| ID | Nhãn | URI | Bằng chứng tiêu biểu | Nhận định |
| ---: | --- | --- | --- | --- |
| 21 | branch | `/c/en/branch` | RelatedTo /c/en/tree | Các cạnh mạnh nhất nói về cây, cành và nhánh cây; `/n` lại chỉ chi nhánh tổ chức. |
| 25 | cap | `/c/en/cap/n` | IsA /c/en/headgear/n | URI danh từ là mũ, có nhiều loại mũ con. |
| 59 | handle | `/c/en/handle` | PartOf /c/en/door | Cạnh cấu trúc và công dụng chủ yếu nói về tay cầm vật lý. |
| 86 | orange | `/c/en/orange/n` | IsA /c/en/citrus_fruit/n | URI danh từ có loại là quả cam; URI trần trộn quả và màu. |
| 95 | plane | `/c/en/plane` | CapableOf /c/en/arrive_at_airport | Cạnh máy bay và đi lại chiếm ưu thế; `/n` là mặt phẳng hình học. |
| 96 | plant | `/c/en/plant` | IsA /c/en/organism | Cạnh sinh vật, vườn và thực vật chiếm ưu thế. |
| 98 | player | `/c/en/player` | PartOf /c/en/team | Người chơi thể thao hoặc trò chơi khớp nghĩa VG; `/n` còn trộn diễn viên và nhạc cụ. |
| 100 | post | `/c/en/post/n` | IsA /c/en/supporting_structure/n | URI danh từ là trụ đỡ vật lý; URI trần thiên về thư từ. |
| 107 | screen | `/c/en/screen` | PartOf /c/en/monitor | URI trần có cạnh trực tiếp với màn hình máy tính và TV. |
| 118 | ski | `/c/en/ski` | RelatedTo /c/en/snow | Phần lớn cạnh liên quan dụng cụ trượt tuyết; còn một cạnh nhiễu về thương hiệu sữa chua. |
| 125 | surfboard | `/c/en/surfboard/n` | IsA /c/en/riding_board/n | URI danh từ là ván cưỡi sóng và thiết bị di chuyển trên nước. |
| 127 | tail | `/c/en/tail` | PartOf /c/en/cat | Cạnh cấu trúc chủ yếu nói về đuôi động vật; có ít cạnh về đuôi diều. |
| 147 | wing | `/c/en/wing` | PartOf /c/en/bird | Cạnh cánh chim và máy bay chiếm ưu thế; nghĩa cánh tòa nhà yếu hơn. |

Các assertion URI chính xác nằm trong `data/knowledge/conceptnet-review-ai-v2/review_suggestions.ai-draft.csv`. Đây là review liên kết **nhãn → URI**; các cạnh xung quanh URI có thể còn nhiễu và chưa được xác nhận từng cạnh.

## 16 nhãn focus và quyết định của Toan

| ID | Nhãn | Nghĩa VG đại diện | URI ứng viên | AI v2 | Lý do AI v2 | Toan quyết định |
| ---: | --- | --- | --- | --- | --- | --- |
| 11 | bike | motorcycle.n.01 | `/c/en/bike` | defer | VG là xe máy, còn URI này có nhiều cạnh về xe đạp. | accept |
| 32 | counter | counter.n.01 | `/c/en/counter` | defer | Trộn quầy, mặt bếp với thiết bị hoặc hành động đếm. | reject |
| 54 | glass | glass.n.02 | `/c/en/glass`<br>`/c/en/glass/n` | defer | Trộn cốc uống nước và chất liệu kính; `/n` thiên về chất liệu. | accept |
| 67 | jean | jean.n.01 | `/c/en/jean`<br>`/c/en/jean/n` | defer | Ít cạnh về quần jean; `/n` có thể nói về vải denim. | reject |
| 68 | kid | child.n.01 | `/c/en/kid` | defer | Cùng URI chỉ cả trẻ em và dê con. | reject |
| 75 | letter | letter.n.02 | `/c/en/letter` | defer | Trộn chữ cái với thư từ. | reject |
| 76 | light | light.n.02 | `/c/en/light`<br>`/c/en/light/n` | defer | VG là thiết bị chiếu sáng, còn URI trộn với ánh sáng như năng lượng hoặc tính chất. | reject |
| 85 | number | numeral.n.01 | `/c/en/number`<br>`/c/en/number/n` | defer | Trộn ký tự số, số lượng và số điện thoại. | reject |
| 87 | pant | trouser.n.01 | `/c/en/pant` | reject | URI ứng viên chủ yếu là động từ thở hổn hển, không phải quần. | reject |
| 89 | paw | paw.n.01 | `/c/en/paw` | defer | Cạnh còn giữ quá yếu để xác nhận bàn chân động vật. | reject |
| 101 | pot | pot.n.04 | `/c/en/pot` | defer | VG là chậu cây, còn cạnh chủ yếu nói về nồi nấu ăn. | reject |
| 104 | rock | rock.n.01 | `/c/en/rock`<br>`/c/en/rock/n` | defer | Trộn đá với nhạc rock. | reject |
| 113 | short | short_pants.n.01 | `/c/en/short`<br>`/c/en/short/n` | reject | VG là quần short, còn URI ứng viên chỉ tính từ “ngắn” hoặc phim ngắn. | reject |
| 123 | stand | base.n.08 | `/c/en/stand`<br>`/c/en/stand/n` | defer | VG là chân đế, còn URI trộn động từ đứng, quầy và đồ nội thất. | accept |
| 138 | trunk | trunk.n.01 | `/c/en/trunk`<br>`/c/en/trunk/n` | defer | VG là thân cây, còn cạnh trộn cốp xe và vòi voi. | reject |
| 143 | wave | wave.n.01 | `/c/en/wave` | defer | Có thể là sóng nước, cử chỉ vẫy tay hoặc dao động khác. | accept |

Các nhãn Toan reject có vector KG bằng 0; 138 nhãn accept có vector sau khi TransE và C.2 export. Đề xuất AI v2 đầy đủ còn ở `data/knowledge/conceptnet-review-ai-v2/review_suggestions.ai-draft.csv`. URI `bike` và `glass` vẫn bị trộn nghĩa; hồ sơ v3 ghi rõ rủi ro này.

Nguồn: ConceptNet 5.7.0 assertions snapshot, CC BY-SA 4.0; Robyn Speer, Joshua Chin, Catherine Havasi (2017), *ConceptNet 5.5: An Open Multilingual Graph of General Knowledge*, AAAI.
