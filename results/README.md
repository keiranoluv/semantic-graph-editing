# Kết quả thực nghiệm

Các tệp dưới đây được chép nguyên byte từ kết quả đã lưu. [Manifest](manifest.json) ghi commit nguồn, đường dẫn gốc và SHA-256. Đơn vị, protocol và phạm vi so sánh nằm trong từng tệp JSON.

| Phần | Kết quả |
|---|---|
| VG150 | [Baseline](vg150/baselines.json), [fusion tri thức](vg150/knowledge_fusion.json), [hoán vị KG](vg150/knowledge_permutations.json), [long-tail](vg150/long_tail.json), [zero-shot](vg150/zero_shot.json) |
| GQA | [Các mô hình](gqa/models.json), [đối chứng adjusted loss](gqa/adjusted_loss.json) |
| MagicBrush | [Hai chính sách nhãn và ba seed trên test](magicbrush/test.json) |
| He và cộng sự | [MLP/GNN trên 530 mẫu test](he2020/test.json) |
| VN100 | [Điểm đồ thị, baseline ảnh và đối chiếu nhãn tay](vn100/consistency.json) |
| PVSG | [Các mô hình trên 62 video test](pvsg/test.json) |

Ảnh minh họa và dự đoán từng ca nằm trong [notebook tổng hợp](../submission/notebooks/project_submission_demo.ipynb). Những đường dẫn lịch sử bên trong JSON trỏ đến cây thực nghiệm tại [snapshot gốc](../archived/README.md); các log, phiếu review và kết quả pilot được giữ ở snapshot đó.
