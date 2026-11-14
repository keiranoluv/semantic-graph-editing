# Pilot MagicBrush compact trên toàn bộ tập train

Pilot kiểm tra khả năng tổng quát hóa sau khi MLP/GNN đã học thuộc 16 mẫu compact. Huấn luyện trên **301 train**, chọn checkpoint trên **64 validation có thao tác**, lưu dự đoán cho toàn bộ **67 validation**. Ba mẫu mất biểu diễn chỉnh sửa sau rút gọn được ghi riêng; không coi dự đoán rỗng ở các mẫu đó là thực hiện đúng yêu cầu.

## Chạy một lệnh

```bash
bash /workspace/workspace/semantic-graph-editing/scripts/run_magicbrush_compact_pilot_v1.sh
```

Launcher tự chạy **CLIP đóng băng → MLP seed 42 → GNN seed 42**, tuần tự một model mỗi thời điểm, rồi dừng. Không có hàng đợi seed khác hoặc bước test. Cache CLIP mới chỉ mã hóa nguồn/câu yêu cầu train và validation cùng prototype từ vựng train, không mã hóa graph đích validation. Batch CLIP **128**.

Hai model khởi tạo mới. Giữ cấu hình huấn luyện v3 để hạn chế thay đổi đồng thời: batch train **16**, batch validation **32**, learning rate `0.0003`, dropout `0.1`, weight decay `0.01`, hidden 192, hai lớp truyền thông tin cho GNN. Từ vựng lấy toàn bộ train compact: 682 nhãn đối tượng, 298 thuộc tính, 81 quan hệ. Không nạp trọng số đã học thuộc 16 mẫu.

Tối đa **200 epoch/model**. Có **early stopping**: tối thiểu 30 epoch, patience 30 epoch kể từ lần cải thiện; đánh giá mỗi hai epoch và ở các mốc đầu/cuối bắt buộc. Chọn checkpoint theo micro F1 thao tác validation, sau đó exact graph, rồi loss thấp hơn; không cải thiện ở mọi tiêu chí thì giữ checkpoint cũ. Patience tính theo epoch, không phải số lần đánh giá.

Cả hai model đều chạy kể cả khi MLP không đạt quality gate. Gate giữ ngưỡng v3 (valid transaction ≥50%, micro F1 ≥5%) để báo cáo tình trạng vận hành, không phải ngưỡng chất lượng do đề bài quy định và không tự mở khóa seed/test khác.

## Báo cáo validation

- **Primary PRF1 theo sáu loại và exact match:** 64 mẫu có thao tác compact, giữ tất cả trường hợp ngoài từ vựng train trong mẫu số. Cạnh replace_edge không có support trên validation, được ghi rõ.
- **Schema coverage:** 64/67; ba mẫu `453731:1`, `456569:3`, `503943:1` được dự đoán và ghi riêng, không đánh giá độ đúng yêu cầu bằng nhãn rỗng.
- **Conservative exact /67:** số graph đúng trong 64 mẫu chia cho toàn bộ 67 mẫu; ba trường hợp ngoài schema không nhận điểm thành công. Đây không phải exact của nhãn đầy đủ v3.
- Output thô và có ràng buộc đều được chấm. Có baseline không chỉnh sửa trên 64 mẫu cùng tập chấm.

Không so trực tiếp F1 pilot này với v3 đầy đủ để quy toàn bộ chênh lệch cho mô hình: nhãn, từ vựng, graph nguồn và tập có thể chấm đều khác. Đây là pilot một seed, chưa thay thế thực nghiệm chính ba seed của nhiệm vụ 2 trong `Project01_Graph_Editing.pdf`.

## Kết quả và tiếp tục khi bị ngắt

- Tổng hợp: `runs/level2/magicbrush-compact-pilot-v1/summary.json`.
- Mỗi model: `<output>/<model>/42/result.json`, `validation-report.json`, `predictions-val.json` (64), `predictions-val-all.json` (67), `best.pt`, `last.pt`.
- Log: `logs/magicbrush-compact-pilot-v1/launcher.log`.
- Cache: `data/processed/level2/magicbrush-compact-pilot-v1/clip-trainval/`.

Chạy lại cùng lệnh để tiếp tục. Cache được kiểm hash rồi dùng lại; checkpoint lưu mỗi epoch, epoch bị ngắt có thể chạy lại. Model hoàn thành được kiểm receipt và bỏ qua. Khóa thiết bị dùng chung bao trùm cả cache và huấn luyện. Không sửa source/config/output của các thực nghiệm cũ.

`preflight.json` ghi kiểm tra CPU trên artifact thật. `verification.json` ghi kiểm thử adapter, cách tính mẫu số và chạy/resume bằng dữ liệu giả lập CPU. Assistant chưa chạy tác vụ GPU. Xác nhận tay 20/20 trước đây thuộc nhãn gốc; pilot không tạo xác nhận tay mới cho nhãn compact.
