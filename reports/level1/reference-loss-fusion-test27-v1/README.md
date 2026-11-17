# VG150 test — phân tích cải tiến loss và fusion

## Kết quả chính

Đủ **27/27 checkpoint**, gồm 12 loss và 15 fusion, ba seed 42/123/3407.
Mỗi checkpoint chạy trên toàn bộ **26.446 ảnh test**, reference-v1 GC,
batch inference 128. Audit CPU đã kiểm hash checkpoint, frozen plan, cache,
evaluator, từng report và tính lại mean/sample std. Kết quả **PASS**.

**Adjusted loss là cải tiến rõ nhất:** cùng KG-off, mR@50 tăng **6,479 pp**
ở cả ba seed, đổi lại R@50 giảm **2,235 pp**. Pair fusion KG thật có mR@50
trung bình cao nhất, nhưng lợi ích riêng của mapping KG đúng vẫn chưa ổn định
so với shuffled-KG. Ba seed không đủ để khẳng định ý nghĩa thống kê.

## So sánh 12 checkpoint loss

Đơn vị %, mean ± sample standard deviation. CE/adjusted và KG-off/on dùng
cùng nhóm thí nghiệm; đây là các đối chứng trực tiếp cho tác động loss và KG.

| Loss | KG | R@50 | R@100 | mR@50 | mR@100 |
|---|---|---:|---:|---:|---:|
| CE | Off | 64,42 ± 0,10 | 66,85 ± 0,09 | 18,11 ± 0,16 | 20,56 ± 0,19 |
| CE | On | 63,63 ± 0,14 | 65,93 ± 0,13 | 19,27 ± 0,75 | 21,51 ± 0,91 |
| Adjusted | Off | 62,18 ± 0,16 | 64,81 ± 0,13 | 24,59 ± 0,93 | 27,89 ± 1,02 |
| Adjusted | On | 62,01 ± 0,31 | 64,71 ± 0,46 | 24,67 ± 1,07 | 28,00 ± 1,19 |

Logit adjustment dùng CE(logits + 0,25 × log(prior_train), target) khi train;
prior chỉ từ train, inference dùng logits gốc. Không thay loss hoặc trọng số trong lượt test.

| Đối chứng | ΔR@50 (pp) | ΔmR@50 (pp) | ΔmR@50 seed 42 / 123 / 3407 |
|---|---:|---:|---|
| Adjusted − CE, KG-off | −2,235 | +6,479 | +6,081 / +7,710 / +5,645 |
| Adjusted − CE, KG-on | −1,612 | +5,399 | +6,384 / +3,771 / +6,042 |
| KG-on − off, CE | −0,792 | +1,166 | +0,254 / +1,848 / +1,395 |
| KG-on − off, adjusted | −0,169 | +0,086 | +0,557 / −2,090 / +1,793 |

Loss cải thiện mean Recall rõ trong cả hai trạng thái KG. Với CE, KG theo cặp
tăng mR@50 ở cả ba seed; sau adjusted loss, mức tăng thêm chỉ 0,086 pp và
đảo chiều mạnh giữa seed. Vì vậy không nên viết chung rằng KG hoàn toàn vô ích:
hiệu quả phụ thuộc nền loss và cách ghép. Đối chứng on/off còn bao gồm thay đổi
dung lượng nhánh pair MLP, nên chưa tách riêng lợi ích ngữ nghĩa của KG.

## So sánh 15 checkpoint fusion

Tất cả dùng adjusted loss, cùng backbone initialization theo seed. Baseline
ở đây là baseline được train trong ma trận fusion, khác checkpoint GAT ban đầu
và khác `loss_adjusted_off`; không gộp các nhóm để tính paired ablation.

| Cấu hình | R@50 | R@100 | mR@50 | mR@100 |
|---|---:|---:|---:|---:|
| Baseline | 62,06 ± 0,39 | 64,83 ± 0,21 | 24,32 ± 0,91 | 27,72 ± 1,08 |
| Pair KG thật | 62,37 ± 0,19 | 65,01 ± 0,20 | **25,24 ± 0,39** | **28,65 ± 0,27** |
| Pair KG xáo trộn | 62,34 ± 0,18 | 65,06 ± 0,24 | 25,00 ± 0,71 | 28,33 ± 0,74 |
| Visual KG thật | 62,34 ± 0,21 | 65,05 ± 0,22 | 24,52 ± 0,64 | 27,98 ± 0,73 |
| Visual KG xáo trộn | 62,20 ± 0,23 | 64,96 ± 0,22 | 24,82 ± 1,25 | 27,94 ± 1,10 |

| Đối chứng | ΔR@50 (pp) | ΔmR@50 (pp) | ΔmR@50 seed 42 / 123 / 3407 |
|---|---:|---:|---|
| Pair thật − baseline | +0,306 | +0,926 | +1,783 / −0,560 / +1,556 |
| Pair thật − shuffled | +0,025 | +0,238 | +0,400 / −0,823 / +1,135 |
| Visual thật − baseline | +0,278 | +0,206 | +0,297 / −1,120 / +1,439 |
| Visual thật − shuffled | +0,142 | −0,300 | −0,917 / +0,752 / −0,735 |

Pair thật cao hơn baseline trung bình 0,926 pp, nhưng shuffled cũng đạt 25,00%.
Phần chênh do mapping đúng chỉ 0,238 pp và giảm ở seed 123. Visual thật thấp
hơn shuffled ở mR@50 trung bình. Kết quả chưa chứng minh liên kết tri thức đúng
mang lại cải thiện ổn định so với cả baseline lẫn đối chứng cùng kiến trúc.
Đối chứng chỉ có một permutation xáo trộn; không suy rộng sang mọi cách fusion.

## Đặt cạnh 27 baseline test đã hoàn tất

Bảng sau dùng cùng reference-v1 GC/test; các giá trị baseline lấy từ
[báo cáo 27 run gốc](../reference-test27-v1/README.md).

| Nhóm/cấu hình | R@50 | mR@50 | Ý nghĩa |
|---|---:|---:|---|
| MLP gốc | 63,56 | 16,95 | Baseline phi đồ thị |
| GAT gốc | 63,87 | 18,61 | Message passing tăng mR |
| GAT + RotatE ở node | 64,21 | 18,57 | Gần hòa GAT về mR |
| CE KG-off, ma trận loss | 64,42 | 18,11 | Control trực tiếp của adjusted loss |
| Adjusted KG-off | 62,18 | 24,59 | Tăng mR, giảm R |
| Adjusted KG-on | 62,01 | 24,67 | KG tăng thêm ít |
| Fusion pair thật | 62,37 | 25,24 | mR@50 cao nhất trong 18 cấu hình đã chấm |

Hiện có **54 report test cho 18 cấu hình × 3 seed**. Bảng tổng quan giúp thấy
trade-off; kết luận nhân quả về loss/fusion phải dựa vào bảng đối chứng cùng
ma trận ở trên. Không chọn lại checkpoint theo bảng test. Các phân tích NG,
zero-shot và head/body/tail trước đó vẫn thuộc validation; lượt này chỉ chấm GC.

## Cách trình bày trong báo cáo kỹ thuật

Có thể dùng câu kết luận:

> Trên VG150 test, logit-adjusted loss tăng mR@50 6,479 điểm phần trăm so với
> CE khi cùng tắt KG, với cải thiện ở cả ba seed, đổi lại R@50 giảm 2,235 điểm.
> Pair fusion với KG thật đạt mR@50 25,24%, nhưng chỉ hơn shuffled-KG 0,238
> điểm và không ổn định qua seed. Kết quả ủng hộ điều chỉnh loss để cải thiện
> recall cân bằng, trong khi lợi ích riêng của tri thức nền còn phụ thuộc cấu hình.

Checkpoint được chọn trên validation bằng metric lịch sử, rồi chấm test bằng
reference-v1 mà không chọn lại epoch. Test cũng đã được dùng đánh giá baseline
trước đây; lịch sử này tự nó không chứng minh thiết kế cải tiến được tune bằng test.
Không gọi đây là tập test mới chưa từng được xem. Sáu checkpoint relation-prior
có bất nhất L1/L2 được loại từ trước và không nằm trong 54 report chính.

## Artifact và tái lập

- [summary.json](summary.json): bốn metric mean/std và hash 27 report inference.
- [frozen-checkpoints.json](frozen-checkpoints.json): 27 checkpoint và nguồn plan đã khóa.
- [analysis.json](analysis.json): audit PASS, paired delta đủ bốn metric theo seed.
- [tables.md](tables.md): bảng sinh tự động từ report, giữ số liệu kiểm tra được.
- Các `*-test26446.json`: metric từng seed, per-predicate và provenance evaluator.
- Launcher commit `d899eb1e`; mã phân tích `tools/analyze_loss_fusion_test27_v1.py`.

Chạy lại audit và thống kê, CPU-only, không inference:

```bash
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m tools.analyze_loss_fusion_test27_v1
```

Artifact cũ được kiểm nội dung, không ghi đè nếu thay đổi. Mean/std mô tả biến động
ba lượt train trên cùng test, không phải khoảng tin cậy về toàn bộ phân bố ảnh.
