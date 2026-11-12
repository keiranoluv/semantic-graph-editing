# MagicBrush v1 — bằng chứng chuẩn bị, chưa phải kết quả model

- 602 cặp ảnh thật, 912 ảnh riêng biệt; 310 nhóm nguồn.
- Split nội bộ từ official train: 415 train / 97 val / 90 test; không trùng
  RGB pixel hash giữa split. Không tải official dev/test annotations.
- Manifest SHA256: `4e815988bb39c130fb912ce5aea0686317d6a0c1647840cbc204b6317834f776`.
- 20 cặp train review được chọn trước khi sinh graph, không lọc theo chất lượng.
- **0 graph-edit labels đã sinh; 0 human reviews hoàn thành. Yêu cầu ≥300
  cặp graph-edit và ≥20 kiểm tay chưa được đánh dấu đạt.**

Các file:

- `preparation.json`: quy mô, nguồn/revision, SHA256 bốn shard train.
- `pair-manifest.csv`: ID gốc, turn, nhóm, split, image hash và câu yêu cầu.
- `split-ids.json`: split IDs và group mapping có thể audit độc lập.
- `review-cohort.json`: 20 ID train cố định.
- `verification.json`: kiểm chứng CPU và bảo toàn thực nghiệm lịch sử.

Full manifest, original image bytes và graph cache sinh sau này nằm dưới
`data/`, không commit binaries. Xem [CLI/protocol](../../../docs/LEVEL2_MAGICBRUSH.md)
và [viewer](../../../reviews/magicbrush-human-v1/README.md).
