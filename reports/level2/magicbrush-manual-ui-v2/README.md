# MagicBrush v2 — hai ảnh ngay từ đầu và nhập JSON

Đã thay luồng source-first bằng xem cả nguồn, đích và instruction khi mở mẫu.
Cả hai graph sửa được. Có gói ZIP chứa ảnh gốc theo thứ tự + prompt zero-shot,
sao chép prompt, dán/file JSON, xem trước và áp dụng vào cả hai bảng.
[Cách chạy và schema](../../../docs/LEVEL2_MAGICBRUSH_MANUAL_V2.md).

**296 tests CPU PASS**, gồm 16 tests v2: source/target hiển thị trước annotation,
prompt không chứa nhãn cũ, ZIP giữ đúng bytes ảnh, parser/ID/schema/transaction,
JSON sai không ghi dữ liệu, CAS conflict, provenance proposal, loader và browser
luồng từ dán JSON tới đối chiếu/hoàn thành. Có kiểm file JSON và mobile.
Kiểm browser dùng fixture synthetic, không tạo nhãn/reviewer giả cho dữ liệu thật.

Smoke thật chỉ đọc: 512 train/val, 20 ID pilot không đổi, hai ảnh tải được cùng
lúc. Snapshot giữ nguyên **15 event v1 của một cặp đã gán**, graph, ID, điểm và
người gán; kiểm SHA từng file gốc không thay đổi. Không tạo event annotation
v2 trên dữ liệu thật. Ảnh preview giữ cục bộ tại
`data/processed/level2/magicbrush-manual-v2/pair-json-preview-final.png`.

JSON nhập được ghi `external_json_proposal`, ngữ cảnh cả hai ảnh/instruction,
raw/hash và draft; chỉ hoàn thành sau thao tác đối chiếu của người dùng, vẫn
chờ review độc lập. Không gọi ChatGPT API, model local hoặc GPU. Không đổi
operation semantics, evaluator hoặc các file code/report của v1 đã khóa.

Đổi server bằng Ctrl+C ở terminal v1 rồi chạy:

```bash
myenv/bin/python -m tools.magicbrush_manual_v2 --port 8776
```

Tải lại `http://127.0.0.1:8776/`. V2 ghi output riêng; v1 vẫn nguyên trạng.
