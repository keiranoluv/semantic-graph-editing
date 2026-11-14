# MagicBrush v3: dự đoán thao tác có cấu trúc

V3 phục vụ nhiệm vụ 2 trong `Project01_Graph_Editing.pdf`: graph nguồn và câu yêu cầu → thao tác thêm/xóa/thay đỉnh, cạnh. Thay bộ sinh JSON từng byte của v1/v2 bằng các đầu phân lớp thao tác, chọn ID và sao chép nội dung nguồn. CLIP ViT-B/32 đóng băng. MLP không truyền thông điệp; GNN dùng hai lớp truyền thông điệp. Hai mô hình dùng cùng dữ liệu, đầu dự đoán và cách chấm.

Đây là một thực nghiệm với từ vựng giới hạn, chưa có kết quả GPU v3. Nó không thay thế yêu cầu phương pháp mới từ năm 2023 hoặc các thực nghiệm nhiệm vụ 1 của cấp độ 2.

## Chạy một lệnh

```bash
bash /workspace/workspace/semantic-graph-editing/scripts/run_magicbrush_manual_full602_v3.sh
```

Launcher dùng `myenv/bin/python`, W&B offline và khóa GPU dùng chung của dự án. Có thể truyền `--wandb-mode disabled`. Không cài thêm thư viện. Các lượt học từ đầu, không tiếp tục trọng số của v1/v2.

| Thiết lập | Giá trị |
| --- | --- |
| Dữ liệu | 301 train, 67 validation, 69 test; cùng split đã khóa |
| Mô hình | MLP, GNN; hidden 192; GNN 2 lớp; dropout 0,1 |
| Batch train / validation-test / CLIP | 16 / 32 / 128 |
| Seed | 42, 123, 3407 |
| Ngân sách | Tối đa 200 epoch; không dừng sớm trước 30 |
| Validation | Epoch 1, mỗi 2 epoch, mốc tối thiểu và epoch cuối |
| Patience | 30 epoch không cải thiện thứ tự chọn checkpoint |
| Optimizer | AdamW; LR 0,0003; weight decay 0,01; gradient clip 5 |
| Chọn checkpoint | F1 thao tác → exact target graph → loss cấu trúc validation thấp hơn |
| Điều kiện chạy tiếp | F1 validation ≥ 0,05 và tỷ lệ transaction hợp lệ ≥ 0,50 |

Thứ tự: **MLP/42 → GNN/42 → MLP/123 → MLP/3407 → GNN/123 → GNN/3407**, một model mỗi thời điểm. Cả hai pilot seed 42 đều chạy, kể cả khi pilot đầu không đạt. Sau hai pilot, cả hai phải đạt điều kiện mới chạy tiếp bốn lượt còn lại; mỗi lượt sau cũng phải đạt trước khi tiếp tục. Pilot được tính vào sáu lượt chính, không huấn luyện lại.

Các ngưỡng này là điều kiện phát triển tự đặt, không phải yêu cầu đạt điểm của đề. Nếu không đạt, ghi `quality_gate_failed`, trả mã lỗi, giữ toàn bộ kết quả đã có và chưa mở test. Không báo cáo trung bình ba seed từ một tập con các seed tốt. Chạy lại cùng lệnh không bỏ qua điều kiện thất bại.

Sau khi sáu lượt hoàn tất và vượt ngưỡng, chương trình khóa checkpoint rồi mới mở nhãn test, tạo cache đầu vào, đánh giá và tổng hợp. Mất kết nối hoặc ngắt tiến trình: chạy lại cùng lệnh để khôi phục epoch đã lưu; lượt hoàn tất được kiểm tra hash rồi bỏ qua. Model, optimizer, RNG và trọng số được chọn lưu cùng một checkpoint nguyên tử.

## Đầu ra và cách giải thích

- Output: `runs/level2/magicbrush-manual-experiment-v3/`.
- Log: `logs/magicbrush-manual-experiment-v3/launcher.log`.
- Trạng thái: `progress.json`.
- Mỗi lượt: `<model>/<seed>/{best.pt,last.pt,result.json,predictions-val.json,completed-receipt.json}`.
- Kết quả đầy đủ: `summary.json`, `SUMMARY.txt`, các `test-result.json` và `predictions-test.json`.
- Đối chiếu quyết định thô/ràng buộc: `raw-and-constraint-diagnostics.json`; từng dòng dự đoán có `raw_tokens`, `raw_error`, `constraints`.
- Độ bao phủ từ vựng validation: `preflight.json`; test: `test-vocabulary-coverage.json`, chỉ tạo sau khi khóa các checkpoint.
- Cache riêng: `data/processed/level2/magicbrush-manual-experiment-v3/{clip-trainval,clip-test}`.

Báo cáo giữ per-operation precision/recall/F1, exact match, tỷ lệ lỗi, nhóm chỉnh sửa và baseline không chỉnh sửa. Chỉ số chính dùng đầu ra có ràng buộc; báo cáo riêng chấm quyết định thô với cùng bộ chấm. “Yêu cầu đơn giản” vẫn là quy ước một thao tác đích, không phải nhãn đánh giá độ phức tạp ngôn ngữ.

## Cách hoạt động và giới hạn

Đỉnh/cạnh nguồn được gán lớp giữ/xóa/thay. Mỗi loại thực thể mới có 12 vị trí dự đoán hiện diện. ID của thực thể cũ được lấy trực tiếp từ graph nguồn; đầu mút cạnh chọn bằng attention trên các vị trí đỉnh cũ/mới hoặc sao chép đầu mút cũ. Nhãn và quan hệ chọn từ các giá trị train hoặc sao chép giá trị cũ. Thuộc tính được biểu diễn bằng quyết định bỏ từng thuộc tính nguồn, số thuộc tính thêm và điểm cho từng thuộc tính trong từ vựng; giữ nguyên các thuộc tính còn lại, kể cả thuộc tính nguồn ngoài từ vựng. Tổ hợp thuộc tính mới không cần từng xuất hiện nguyên bộ trong train.

Các giá trị ngữ nghĩa được nhúng bằng CLIP đóng băng và các đầu dự đoán học cách so khớp với chúng. Từ vựng chỉ dùng graph nguồn/đích **của train**: 682 nhãn đối tượng, 1.448 thuộc tính riêng lẻ, 81 quan hệ. Cache ghi rõ có nhúng từ vựng ngữ nghĩa rút từ nhãn train; không dùng nội dung nhãn đích validation/test để tạo từ vựng hay đặc trưng suy luận.

Hạn chế quan trọng: trên validation, từ vựng biểu diễn được nội dung của **154/219 thao tác**, và **15/67 mẫu** có toàn bộ nội dung đích biểu diễn được. Có 92 giá trị trường đích ngoài khả năng biểu diễn này. Đây là kiểm tra khả năng biểu diễn, không phải độ chính xác mô hình. **Vẫn đánh giá đủ 67 mẫu**, gồm cả những mẫu ngoài từ vựng; không loại chúng để nâng điểm. Các trường nhãn không có lớp tương ứng được bỏ qua trong loss phân lớp; với thuộc tính, vẫn tính loss trên các giá trị biết được và số thuộc tính cần thêm. Tất cả lỗi ngoài từ vựng vẫn nằm trong mẫu số đánh giá thao tác.

Ràng buộc cấu trúc chỉ dùng nguồn và dự đoán: che đầu mút đã bị xóa, xuất thao tác xóa rõ ràng cho cạnh liên quan tới đỉnh bị xóa, và giữ cạnh đầu tiên khi trùng bộ ba. Thuộc tính là tập nên không lặp. Vì vậy tỷ lệ hợp lệ cao có thể đến từ thiết kế bộ giải mã, không chứng minh mô hình hiểu đúng câu lệnh. Không âm thầm cắt các dự đoán vượt 12 thao tác hoặc gộp các đỉnh mới có danh tính mơ hồ; bộ chấm cũ tính các trường hợp đó là lỗi. JSON chỉ là định dạng lưu/chấm, không phải chuỗi ký tự mô hình phải học sinh.

Loss tăng trọng số thao tác thay đổi và cân bằng các quyết định thuộc tính dương/âm để số lượng lớn trường không đổi không lấn át thay đổi cần học. Các trọng số được khai báo trong config. Mọi lựa chọn checkpoint chỉ dùng validation; test không tham gia chọn từ vựng, chọn epoch hoặc quyết định chạy tiếp.

## Kiểm chứng và provenance

`codec-audit.json` ghi kiểm tra mã hóa/giải mã với đầu ra đúng được cung cấp sẵn: cả 301 mẫu train và 15 mẫu validation đủ từ vựng khôi phục đúng graph đích. Kiểm tra này không huấn luyện mô hình và không phải kết quả dự đoán. Các kiểm thử CPU còn kiểm tra đủ sáu thao tác, sao chép giá trị nguồn ngoài từ vựng, tổ hợp thuộc tính mới, chặn rò rỉ nhãn đích, học ví dụ giả lập bằng cả MLP/GNN, khôi phục sau ngắt, kiểm tra hash và chặn test trước khi hoàn tất hàng đợi.

`preflight.json` ghi kiểm tra CPU của dữ liệu, môi trường và khóa phiên bản. `verification.json` ghi kết quả kiểm thử. Code nằm ở `tools/magicbrush_experiment_v3/`; config và khóa tại `configs/level2/magicbrush-manual-experiment-v3{,.lock}.json`. Giữ nguyên `src/`, các phiên bản cũ, manifest và notebook cá nhân.

V1 đã đánh giá cùng tập test; v3 không được mô tả như sử dụng một tập test mới chưa từng xem. V3 dựa trên chẩn đoán validation của v2. Nhãn là AI-assisted với 20 mẫu train được người dùng xác nhận; 20/20 không chứng minh tất cả 437 nhãn đều đúng. Chưa có bằng chứng chất lượng thực tế v3 trước khi người dùng chạy GPU; việc phân tích 5 dự đoán đúng/5 sai và demo kết quả v3 cần kết quả thực nghiệm thật.
