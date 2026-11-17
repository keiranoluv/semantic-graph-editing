# Edge-only và PE-CE + AlignG: sẵn sàng train chính

Code commit trước GPU: `5cc246c3`. Không khởi chạy train chính.

- 132 CPU tests PASS: đúng loss/gradient, fast-vs-slow logits, resume,
  queue 6 lượt, skip completed, stop-on-error và bắt buộc W&B online.
- GPU RTX 5090 32 GiB: đủ 6 case batch512 (2 model × 3 batch lớn đã khóa).
- Edge-only peak allocated 4,88 GiB; PE-CE+AlignG 8,58 GiB.
- Peak reserved lớn nhất lần lượt 6,21 GiB và 10,01 GiB.
- Probe chỉ disposable optimizer updates, không W&B, không train checkpoint chính.
- Đã kiểm 9 control lịch sử PE-CE / PE-PLPR / PE-PLPR+AlignG: recipe,
  identity và checkpoint hash. Không chạy lại các baseline này.
- Source/config lịch sử và thư mục src giữ nguyên.

## Lệnh train chính dành cho người dùng

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_aligng_ablation_v1.sh
```

Queue: Edge42 → CE+AlignG42 → Edge123 → CE+AlignG123 → Edge3407 → CE+AlignG3407.
Batch train/val/effective 512; tối đa 200 epoch, patience 15; chọn validation GC mR@50.
Tự chạy tiếp đủ 6 lượt tuần tự; dừng khi lỗi, không retry vô hạn. Chạy lại cùng lệnh
để resume unfinished và skip completed đã xác minh.

W&B online bắt buộc ở project `keiranoluv-hcmus/semantic-graph-editing`.
Output `runs/level2/aligng-ablation-v1/full/{model}/{seed}/`.
Log `logs/aligng-ablation-v1.log`; tmux `aligng-ablation-v1`.

[Recipe và ý nghĩa đối chứng](../../../experiments/aligng_ablation_v1/README.md)
· [Plan + hash control](plan.json) · [GPU capacity](capacity.json).

Chưa có kết quả accuracy cho hai biến thể. Probe không dùng để kết luận model nào tốt hơn.
PE-CE lịch sử dùng engine khác, cùng dữ liệu/recipe nhưng không đồng nhất dropout trajectory.
