# Phép thử học thuộc 16 mẫu MagicBrush compact

Launcher kiểm tra MLP/GNN có học thuộc được 16 mẫu train với nhãn compact hay không. Giữ đúng 16 ID và nhóm ảnh của phép thử trước, cùng seed 42, kiến trúc, learning rate và ngân sách. Graph nguồn, graph đích và từ vựng thuộc tính dùng phiên bản compact đã khóa; số thao tác đích giảm từ 50 xuống 47. Vì đáp án thay đổi, không dùng chênh lệch F1 này để tuyên bố mô hình tổng quát hóa tốt hơn.

## Lệnh dành cho người dùng

```bash
bash /workspace/workspace/semantic-graph-editing/scripts/run_magicbrush_compact_overfit16_v1.sh
```

Lệnh tự chạy tuần tự:

1. Tạo cache **CLIP ViT-B/32 đóng băng**, batch **128**, cho 1.160 chuỗi: đầu vào nguồn/câu yêu cầu của 16 mẫu và prototype từ vựng của toàn bộ 301 mẫu train compact. Dùng weights cục bộ đã kiểm hash, không huấn luyện CLIP. Không dùng feature cũ cho chuỗi đã thay đổi.
2. Huấn luyện **MLP**, seed **42**, batch train/eval **16**.
3. Huấn luyện **GNN**, seed **42**, batch train/eval **16**, rồi dừng launcher.

Mỗi model khởi tạo mới, không nạp checkpoint đã học của v3. Từ vựng vẫn dùng toàn bộ train: **682 labels, 298 attributes, 81 predicates**. Tối đa **3.000 bước cập nhật/model**, đánh giá mỗi **25 bước**; learning rate `0.0003`, dropout `0`, weight decay `0`.

Dừng sớm khi cả output thô lẫn output có ràng buộc đạt **16/16 exact operation set và exact target graph**, qua **hai lần kiểm tra liên tiếp**. Không dùng patience. GNN vẫn chạy nếu MLP hết ngân sách mà chưa học thuộc. Không tự chạy tiếp toàn bộ dataset, ba seed hay test.

## Output và chạy tiếp

- Kết quả chung: `runs/level2/magicbrush-compact-overfit16-v1/summary.json`.
- Từng model: `<output>/<model>/42/result.json`, `predictions-train.json`, `best.pt`, `last.pt`.
- Log: `logs/magicbrush-compact-overfit16-v1/launcher.log`.
- Cache mới: `data/processed/level2/magicbrush-compact-overfit16-v1/clip-train/`.

Chạy lại cùng lệnh để tiếp tục. Cache hoàn thành được kiểm hash và dùng lại; model hoàn thành được kiểm receipt rồi bỏ qua. Model đang chạy tiếp tục từ checkpoint đánh giá gần nhất, có thể tính lại tối đa 24 bước. Khóa thiết bị bao trùm cả cache và huấn luyện để tránh chạy đồng thời với launcher khác dùng cùng cơ chế khóa.

`MEMORIZED` là đạt tiêu chí học thuộc; `NOT_MEMORIZED_WITHIN_BUDGET` là chưa đạt trong 3.000 bước, không tự động chứng minh lỗi code. Sau khi cache được tạo, launcher kiểm tra các tensor đầu vào không trùng nhau; chưa thể xác nhận phần này bằng preflight CPU trước lần encode CLIP mới.

## Phạm vi kiểm tra

16 mẫu đều thuộc train, có đủ sáu loại thao tác và 47/47 payload nằm trong từ vựng train compact. Ba mẫu validation bị mất thao tác ở bước rút gọn không tham gia phép thử này. Parent preflight đọc/kiểm tra artifact trainval; validation không dùng để học, chọn checkpoint hoặc chấm probe. Không đọc nhãn test.

`preflight.json` ghi ID, coverage và phạm vi chạy. `cpu-shape-check.json` kiểm forward/backward trên kích thước thật của 16 nhãn với embedding giả lập 512 chiều. Các test học và resume dùng dữ liệu giả lập CPU; đây không phải kết quả học trên CLIP thật. Assistant chưa chạy tác vụ GPU.

Theo `Project01_Graph_Editing.pdf`, đây là chẩn đoán cho nhiệm vụ 2. Nó không thay thế thực nghiệm chính ≥300 cặp, kiểm tra tay, precision/recall/F1 từng loại và ba seed. Toàn bộ code, config, manifest, output cũ và notebook cá nhân được giữ nguyên.
