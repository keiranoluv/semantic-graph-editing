# VG150 PE-Net v1 — chuẩn bị, chưa chạy training GPU

Đã có pipeline mới cho MLP/GAT/GAT+WordNet-KG/PE-CE/PE-PL/PE-PLPR, supervision
đa nhãn matched, ba seed 42/123/3407 và evaluator reference 50 predicate.
Giữ split 57.723 train / 5.000 val / 26.446 test, tái sử dụng CLIP cache và
WordNet bundle đã duyệt. Chưa có pilot hoặc model chính được train ở version này.

- `preparation.json`: hash, audit train, taxonomy/GloVe mapping và policy.
- `pilot-cohort.json`: cố định 512 train / 128 val, chung cho sáu model.
- `cpu-wiring-smoke.json`: input thật, forward/backward CPU; 0 optimizer update.
- `verification.json`: tests CPU, nguồn/asset lịch sử giữ nguyên, không chạy GPU.

GloVe: 150 object exact; 30 predicate exact + 20 word mean, không có foreground
OOV ngẫu nhiên. Các GT đa predicate được giữ trong train; GT trùng không tăng
trọng số train, vẫn giữ trong evaluation. Không thay đổi kết quả Level 1 cũ.

Code/cấu hình/lệnh chạy: [LEVEL2_VG150_PENET.md](../../../docs/LEVEL2_VG150_PENET.md).
Hiện ưu tiên chạy MagicBrush; lệnh GPU VG150 dành cho người dùng ở lượt sau.
