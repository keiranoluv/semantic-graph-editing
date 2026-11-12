# Grounded anchor v1 — đã chuẩn bị CPU, chờ người dùng chạy pilot

Thay gán nhãn bằng hai graph Qwen độc lập bằng BLIP source caption → Grounding
DINO object boxes → CLIP/crop/pixel checks → thao tác cục bộ giữ ID nguồn.
Graph chỉ gồm object và thứ tự tâm box, chưa có semantic support/action edges.

602 cặp, ảnh, instruction và split gốc được giữ nguyên. Grammar phân tích trên
415 project-train requests, hỗ trợ 64 và báo unsupported cho 351; đây chưa phải
64 nhãn đúng. Chọn cố định 20 request được hỗ trợ trước inference, giữ nguyên
cohort review cũ trong manifest để truy vết.

`verification.json` ghi 88 kiểm thử CPU (16 kiểm thử mới), processor probes,
hash và kiểm tra bảo toàn. Viewer pending đã có 20 cặp ảnh thật, chưa có graph
hay quyết định giả. Chưa chạy GPU, chưa train, chưa đo chất lượng model.

Code chỉ có pilot, không tự chạy full hoặc mở khóa trainer. Nếu pilot đạt chất
lượng mà thiếu số lượng, mở rộng bằng official-train pairs có instruction thuộc
phạm vi hỗ trợ trong manifest mới, thay vì ép đáp án cho yêu cầu ngoài phạm vi.

Lệnh và tiêu chí kiểm được ghi trong
[runbook](../../../docs/LEVEL2_MAGICBRUSH_GROUNDED_ANCHOR_V1.md).
