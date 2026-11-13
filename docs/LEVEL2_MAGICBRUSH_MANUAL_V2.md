# MagicBrush v2 — xem hai ảnh, zero-shot và nhập JSON

V2 thay luồng source-first của v1 bằng **hai ảnh + câu yêu cầu hiện ngay khi
mở mẫu**. Không cần điền hoặc xác nhận graph nguồn để xem đích. Cả hai bảng
graph sửa được trực tiếp; có thể điền cả cặp bằng JSON do ChatGPT gợi ý.
Không gọi API, VLM hay tác vụ GPU; việc gửi ảnh đến ChatGPT do người dùng thực hiện.

## Chạy

Dừng server v1 bằng Ctrl+C trong terminal đang chạy nó, rồi chạy:

```bash
myenv/bin/python -m tools.magicbrush_manual_v2 --port 8776
```

Mở hoặc tải lại `http://127.0.0.1:8776/`. Có thể chạy v2 ở cổng khác bằng
`--port 8777`. Nếu chạy trên máy chủ khác, forward cổng tương ứng.

V2 dùng output riêng `data/processed/level2/magicbrush-manual-v2/`. Lần đầu
tự sao lưu lịch sử và nhãn đã lưu ở v1, giữ graph, ID, điểm đánh dấu, tên người
gán và trạng thái hoàn thành. Không sửa file v1. Các lượt sửa tiếp chỉ ghi vào
v2, có parent hash và provenance. Sau khi chuyển, tiếp tục làm trên v2 để
tránh hai workspace phát triển tách nhau. Bản nháp v1 chưa lưu server vẫn ở
browser v1; hãy lưu hoặc tải backup trước khi dừng server đó.

## Nhờ ChatGPT gợi ý graph

1. **Tải 2 ảnh + prompt**: ZIP gồm `1-source.png` và `2-target.png` (đuôi
   đúng theo file gốc), `prompt.txt`, `provenance.json`. Ảnh giữ nguyên bytes
   và hash; không ghép/crop hoặc dùng ảnh minh họa.
2. Giải nén, đính kèm hai ảnh vào ChatGPT theo thứ tự nguồn trước, đích sau.
   Nhấn **Sao chép prompt**, dán vào ChatGPT rồi gửi. Nếu clipboard bị chặn,
   giao diện mở ô prompt để chọn và Ctrl/Cmd+C.
3. Prompt yêu cầu đọc từng ảnh, mô tả cả phần không thay đổi và căn chỉnh
   identity. Không đưa graph đã nhập hoặc ví dụ gán nhãn vào prompt: đây là
   zero-shot, chỉ có chỉ dẫn schema, sample ID, câu yêu cầu và hai ảnh.
4. **Dán / nhập JSON**: dán câu trả lời hoặc chọn file JSON. Nhận cả JSON
   thuần và một khối Markdown ` ```json ... ``` `.
5. Nhấn **Kiểm tra / xem trước** để xem số đối tượng, số thao tác và các lỗi.
   Nhấn **Áp dụng vào hai graph** để lưu một bản nháp. Bản graph cũ vẫn trong
   lịch sử; điểm đánh dấu được xóa vì ID trong proposal có thể mang nghĩa khác.
6. Sửa graph nếu cần; **Kiểm tra thao tác**, đối chiếu ảnh, đánh dấu hai mục
   và hoàn thành. Không coi việc parse JSON/transaction đúng là xác nhận ngữ
   nghĩa. Nhãn hoàn thành vẫn cần review độc lập theo protocol dự án.

Schema nhận:

```json
{
  "sample_id": "magicbrush:334954:1",
  "source_graph": {"nodes": [], "edges": []},
  "target_graph": {"nodes": [], "edges": []},
  "notes": ""
}
```

Các mảng rỗng trên chỉ mô tả khung schema, không phải nhãn hợp lệ. Mỗi node
có đúng `id`, `label`, `attributes` (mảng chuỗi). Mỗi edge có đúng `id`,
`subject`, `predicate`, `object`. Source dùng `n001...`, `e001...`; target giữ
ID tương ứng, node/cạnh mới dùng `newn001...`, `newe001...`. Import chuẩn hóa
ID nguồn chỉ theo nguồn, giữ correspondence mà JSON khai báo; không tự đoán
identity đúng hoặc thêm nội dung để khớp câu yêu cầu. Người kiểm phải xác nhận
correspondence. Không cần ChatGPT xuất `operations`; tool tự diff.

Sai sample ID, khóa JSON lặp, graph có endpoint thiếu, schema lạ hoặc identity
node mới không chuẩn hóa được bị từ chối **trước khi ghi**. No-op/vượt budget
thao tác có thể giữ ở dạng nháp để đối chiếu, không được hoàn thành. Budget
và semantics Task B giữ nguyên: ≤20 node /60 edge mỗi graph, 1–12 operations,
≤2048 action tokens, transaction/codec tái dựng đúng target.

## Dữ liệu, provenance và giới hạn

- Giữ đúng 20 ID khởi động và 415 train +97 validation của manifest gốc;
  không mở 90 mẫu test. Không sửa source/manifest/config hoặc report v1.
- Snapshot v1 dưới `legacy-v1/` có checksum từng file. V2 tham chiếu SHA của
  event v1 cuối trước khi ghi event v2 mới; không gán lại tác giả lịch sử.
- Import JSON ghi raw text/hash, `annotation_origin=external_json_proposal`,
  `annotation_context=both_images_and_instruction_visible`, trạng thái draft,
  reset hai mục xác nhận. Không tự tạo reviewer hoặc verdict.
- V2 không tuyên bố graph nguồn được gán mù với đích. Prompt yêu cầu graph
  nguồn chỉ phản ánh ảnh nguồn; ID source và feature model vẫn chuẩn hóa chỉ
  từ graph nguồn, target/operations chỉ làm nhãn offline.
- Các nhãn xuất giữ origin theo từng record, không gọi proposal model là
  nhãn hoàn toàn do người tự gán. Không biết model/revision ChatGPT từ JSON
  dán thì không tự đoán hoặc ghi phiên bản model giả.
- Autosave, undo/redo, phục hồi bản nháp, kiểm revision hai tab và export
  snapshot vẫn có. JSON import không ghi đè file annotation cũ.
- Chưa tự gửi dữ liệu ra ngoài, chưa sinh nhãn thật, chưa train. ≥300 usable
  và review ≥20 vẫn là các gate riêng, không được coi hoàn thành bởi import.

```bash
myenv/bin/python -m tools.magicbrush_manual_v2 --stage plan
myenv/bin/python -m tools.magicbrush_manual_v2 --stage compile
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 myenv/bin/python -m pytest -q tests/editing/test_magicbrush_manual_v2.py
```
