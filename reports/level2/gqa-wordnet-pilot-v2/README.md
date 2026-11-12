# GQA WordNet v2 — pilot kiểm kỹ thuật

**PASS**, 512 train / 128 validation, seed 42, 5 epoch, batch 512; mỗi model
chỉ có 5 bước optimizer. Ba cấu hình GAT, KG-off và KG-on nhận cùng ảnh/cặp/
soft target. KG-off/KG-on có khởi tạo giống hệt.

| Cấu hình | Epoch chọn | GC mR@50 (%) |
| --- | ---: | ---: |
| GAT | 3 | 0,092535 |
| GAT, nhánh KG tắt | 1 | 0,331969 |
| GAT + WordNet v2 | 1 | 0,332490 |

Bảng này là kiểm pipeline, chưa đủ để kết luận KG cải thiện dự đoán. Chọn
checkpoint bằng GC mR@50 trên đủ 310 predicate; các lớp vắng vẫn tính 0.
Không so chất lượng các điểm pilot này với những lượt full train đã hội tụ.

Ngắt KG-on sau bước đầu rồi resume đã khớp tuyệt đối với bản chạy liền về
history, metrics, model, optimizer, CPU/CUDA/sampler RNG và best model state.
Peak CUDA allocated của pilot cao nhất khoảng **4,153 GiB**. Không chấm test GQA.

[Summary](summary.json), [experiment](experiment.json), [GAT](gat.json),
[KG-off](gat-kg-off.json), [KG-on](gat-kg.json). Full launcher yêu cầu pilot
PASS và feature-manifest hash khớp trước khi cho chạy thực nghiệm chính.
