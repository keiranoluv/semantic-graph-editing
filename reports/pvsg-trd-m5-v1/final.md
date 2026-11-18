# Báo cáo cuối · Phát hiện quan hệ theo thời gian trên PVSG

**Đã nghiệm thu M0–M5 cho phần mở rộng TRD của đồ án GML.** Trên 62 video test, Conv k5 × 3 tìm lại được nhiều quan hệ nhất với R@50 **8,63%**. ST-GAT đầy đủ đạt mR@50 cao nhất (**2,38%**) khi áp dụng quy tắc kiểm tra hội tụ muộn cho cả 24 cặp mô hình/seed.

## Bài toán và thiết kế so sánh

Mỗi mô hình nhận mask, lớp và ID thật của đối tượng qua các frame, rồi dự đoán bộ ba (chủ thể, quan hệ, đối tượng) cùng các khoảng frame quan hệ có hiệu lực. Chẳng hạn, hai track có thể cùng xuất hiện trong cả video nhưng quan hệ “cầm” chỉ đúng ở một đoạn. Cách thiết lập này đánh giá phần dự đoán quan hệ của đồ thị video. Phần video trong đề bài gốc còn yêu cầu đánh giá tính nhất quán khi chỉnh sửa trên DAVIS/YouTube-VOS; PVSG cung cấp nhãn quan hệ theo thời gian cho câu hỏi GML hẹp hơn. [Đề bài gốc](../../Project01_Graph_Editing.pdf)

Dùng split phát hành của tác giả gồm 338 video train và 62 test. Trong 338 video, 287 dùng để fit và 51 để chọn checkpoint. M2 đồng bộ mask với thời gian của video, dùng CLIP đóng băng và loại khỏi supervision các nhãn thiếu hỗ trợ theo quy tắc đã khóa. Mọi mô hình xét cùng các cặp có hướng đồng hiện, dùng BCE đa nhãn và cùng bộ giải mã. Temporal Conv được chuyển thể từ khối dự đoán quan hệ của PVSG CVPR 2023 sang đầu vào GT-track/CLIP. So sánh với điểm công bố của pipeline đầy đủ còn phụ thuộc đầu ra segmentation/tracking và evaluator của tác giả. [Protocol](../../experiments/pvsg_trd_v1/protocol.json) · [Nguồn và phạm vi chuyển thể](../../experiments/pvsg_trd_v1/m4/provenance/sources.json)

## Nghiệm thu theo mốc

| Mốc | Kết quả đã đối chiếu |
|---|---|
| M0 | Khóa danh sách video, 57 predicates và protocol GT-track TRD. |
| M1 | Parser, graph, decoder và evaluator dùng khoảng frame inclusive, temporal set IoU; có đối chứng CPU trên fixture. |
| M2 | Cache CLIP đủ 400/400 video; split nội bộ 287/51/62. |
| M3 | MLP, Static GAT và ST-GAT hoàn thành 9 lượt, ba seed mỗi mô hình. |
| M4 | Hai Conv và ba ablation cạnh hoàn thành 15 lượt. Các ablation giữ self-loop và tập cặp cần dự đoán. |
| M5 | Đủ 24 checkpoint gốc và 7 checkpoint bổ sung đã chọn bằng validation. Mỗi checkpoint khớp hash đã khóa và được chấm trên cùng 62 ID, 931 bộ ba quan hệ GT. Có điểm theo nguồn/lớp/nhóm tần suất và 5 ví dụ đúng, 5 ví dụ bỏ sót. |

## Kết quả test

Các số dưới đây là phần trăm, trung bình ± độ lệch chuẩn mẫu qua seeds 17, 42, 2026. Mỗi video lấy tối đa 50 dự đoán xếp hạng đầu. Một lượt trúng cần đúng bộ ba và temporal IoU ≥ 0,5. R tính trên tổng 931 bộ ba; mR lấy trung bình recall của 54 lớp có mặt trong test.

**Bảng chính tính cả hội tụ muộn.** Sau khi xem validation và trước khi chấm test, nhóm xác định quy tắc dùng cho mọi cặp mô hình/seed: nếu lượt gốc dừng ngay ở epoch 16 với cả R@50 và mR@50 validation bằng 0, chạy lại từ đầu với cùng seed, cho phép dừng từ epoch 50 và dùng checkpoint của lượt chạy lại. Đúng 7/24 cặp thỏa điều kiện; 17 cặp còn lại dùng checkpoint gốc. Việc chọn lượt đưa vào bảng chỉ dựa trên validation.

Mỗi mô hình đóng góp đủ ba seed, mỗi seed đúng một lần. Các lượt cùng dùng batch 32, tối đa 150 epoch, patience 15 và cùng tiêu chí chọn checkpoint (mR@50 trên validation, hòa thì xét R@50 rồi lấy epoch sớm hơn). Số epoch train thực tế có thể khác nhau theo quy tắc trên.

| Mô hình | Seed dùng min50 | R@50 (%) | mR@50 (%) |
|---|---|---:|---:|
| mlp | — | 3,80 ± 0,43 | 1,62 ± 0,12 |
| static_gat | — | 3,80 ± 0,65 | 1,68 ± 0,17 |
| st_gat | 17, 2026 | 4,98 ± 0,27 | 2,38 ± 0,54 |
| pvsg_conv_k5_l3 | — | 8,63 ± 0,61 | 1,25 ± 0,25 |
| st_no_temporal | 17, 2026 | 3,58 ± 1,35 | 1,28 ± 0,75 |
| st_no_spatial | — | 3,47 ± 1,20 | 1,07 ± 0,70 |
| st_no_edges | 17, 42, 2026 | 3,62 ± 0,16 | 1,51 ± 0,04 |
| mlp_conv_k3_l2 | — | 6,77 ± 0,65 | 1,29 ± 0,26 |

[Bảng đối chiếu với kết quả trước khi kiểm tra hội tụ muộn](combined_min50.md)

## Diễn giải và giới hạn

- **Ngữ cảnh thời gian.** Conv k3 × 2 và ST-GAT cùng có ngữ cảnh tối đa 5 frame, nhưng R@50 trong bảng chính lần lượt là 6,77% và 4,98%. Conv k5 × 3 nhìn tối đa 13 frame và đạt 8,63%. Hai Conv còn khác độ sâu và số tham số, nên số liệu này chưa tách riêng được tác dụng của độ dài ngữ cảnh.
- **Vì sao R và mR cho thứ hạng khác nhau?** Với Conv k5 × 3, mR@50 trung bình theo ba nhóm lớp phổ biến/vừa/hiếm là 3,56% / 0,00% / 0,00%. Với Static GAT, các số tương ứng là 1,80% / 3,33% / 0,00%. Mỗi nhóm gồm 19 predicates xếp theo tần suất bộ ba trong tập fit; chỉ lớp có hỗ trợ trên test tham gia tính mR. Conv thu được nhiều lượt trúng hơn ở nhóm phổ biến. Recall cũng thay đổi theo nguồn: trên Ego4D / EPIC-Kitchens / VidOR, R@50 của Conv lần lượt là 11,72% / 10,67% / 7,18%.
- **Kiểm tra hội tụ muộn theo yêu cầu.** Bảy cặp model/seed gốc dừng ở epoch 16 với recall bằng 0. Khi chạy lại từ đầu với cùng seed và chỉ lùi thời điểm được phép dừng, cả bảy vẫn bằng 0 qua 16 epoch đầu, sau đó có lượt trúng validation đầu tiên ở epoch 17–20. Cả bảy checkpoint bổ sung cũng có recall dương trên test. ST-GAT đầy đủ đạt mR@50 2,38% trong bảng chính. Đây là bằng chứng về việc học muộn trong những ca đã chọn. Kết quả mô tả hiệu năng theo quy tắc dừng có điều kiện nêu trên. [Bảy cặp trước/sau](report.md)
- **Ví dụ lỗi.** Ở Static GAT seed 17, video `0004_11566980553`, bộ ba ID `(8,20,4)` xuất hiện trong top 50 nhưng đoạn dự đoán chỉ đạt tIoU 0,289. Đây là lỗi định vị thời gian dù đã tìm được đúng bộ ba. Trong video `0039_6951351121`, bộ ba `(14,43,3)` đạt đúng ngưỡng 0,50 và được tính trúng. Hình dùng thứ tự video và bộ ba đã định trước; các ví dụ minh họa cách chấm. [Timeline GT và dự đoán](case_timelines.png)

Kết quả cho thấy tác dụng của temporal Conv trên recall tổng và độ nhạy của các mô hình đồ thị với tiêu chí dừng. Ba seed và độ phân tán quan sát được chỉ cho phép kết luận trong protocol GT-track này. Đầu vào track thật loại bỏ lỗi phát hiện và liên kết đối tượng. Ngoài ra, bốn nhóm nguồn Ego4D có ID nguồn chung giữa hai phía của outer split phát hành, nên việc phân chia theo video chưa bảo đảm độc lập hoàn toàn theo nguồn quay.

**Dấu vết kiểm tra.** Bảng chính được tính lại từ kết quả từng seed; bảy cặp bổ sung được đối chiếu riêng cả ở validation và test. [Danh sách checkpoint và ID test đã khóa](frozen.json) · [Số liệu tổng hợp](summary.json) · [Kế hoạch và tài liệu kỹ thuật](../../docs/research/pvsg_trd_plan.md)
