# Qwen 2.1 — 100 nguồn / 100 đích, lượt đầu

Người dùng yêu cầu assistant xem ảnh, viết một yêu cầu chỉnh sửa đơn giản cho mỗi ảnh, commit code rồi trực tiếp chạy GPU cho 100 ảnh đích. Đây là quyền chạy riêng cho lượt này; không tự chạy seed khác hoặc mở rộng ngoài 100 task.

## Chọn ảnh và ý định trước inference

Dùng đúng hai nguồn/version của pilot: Sidewalk `2b8b043de9604a41e9366a80d24ea8c54c67c6f4`, VietTravelVQA-v2 `1aa7c4f3c3bd179b94c3093678515af7f6f3da48`. Công cụ fetch cũ chỉ đọc config, tải 150 ứng viên mỗi nguồn vào root mới `data/raw/level3/qwen21-batch100-v1/`. Không đổi raw/locked/config cũ.

Assistant xem 130 ứng viên đạt kiểm tra kỹ thuật trên contact sheet, chọn 50 mỗi nguồn; loại minh họa, các cảnh quá tối/khó chọn đối tượng, nhiều ảnh mờ và một cặp cùng cảnh gần trùng rõ rệt. Đã chọn 100 SHA256 khác nhau, không cặp dHash Hamming ≤4. Bộ lọc này không chứng minh mọi cảnh độc lập: cùng địa điểm có thể có nhiều góc nhìn. Khi chia dev/test sau này phải nhóm theo địa điểm/ảnh gần nhau, giữ các ảnh pilot đã xem ở development. Bộ 100 này chưa phải test set độc lập.

`configs/level3/qwen100-instructions-v1.json` lưu 100 câu tiếng Anh do assistant viết sau khi xem ảnh, đối tượng/vị trí, thao tác add/remove/replace, bằng chứng nhìn thấy và pinned URL/hash. Phân bố: 66 thêm, 28 xóa, 6 thay thế. Đây không phải thiết kế cân bằng theo loại thao tác; ảnh Travel nhiều cảnh quan nên thiên về thêm. Không chỉnh thuộc tính vì REACT++ hiện không có attribute head. Ý định này chưa phải dự đoán mô hình Task B hoặc graph GT. `author=assistant_visual_review`, nhãn chất lượng người dùng để null; không giả reviewer người thật.

Ảnh chỉ dùng đánh giá local với provenance; chưa xác minh quyền phát hành từng ảnh. Không upload ảnh vào W&B/Git. Bộ 100 đáp ứng lượt sinh đầu, đề bài vẫn cần 150–200 yêu cầu và nhãn tay ở bản hoàn chỉnh. Không sinh thêm 50 yêu cầu trong lượt này.

## Giao thức sinh ảnh

Checkpoint và Diffusers được kế thừa nguyên bản đã pin của `level3_qwen21_v1`: official Qwen/Qwen-Image-2.1, BF16, literal seed 42 cho từng task, 40 bước, batch 1, output_resolution 1024, true_cfg_scale 1, KV cache bật, model CPU offload và VAE tiling. Một model tại một thời điểm, có inference lock dùng chung với SGG.

**Khác pilot 6 mẫu:** input lấy từ ảnh gốc, EXIF transpose → RGB PNG, không hạ xuống 512 trước. Qwen vẫn resize conditioning nội bộ theo cùng output_resolution. Lưu hash ảnh gốc và hash PNG input riêng. Đây là phiên bản preprocessing mới, không gộp với pilot cũ để tuyên bố so sánh cùng độ phân giải input. Mỗi yêu cầu độc lập từ ảnh nguồn, không edit nối tiếp ảnh đích.

Không chọn lại seed hoặc sinh lại vì ảnh xấu. Kết quả đúng/sai đều được giữ. Lỗi kỹ thuật/OOM dừng lượt và giữ artifacts. Sau khi kết thúc có PNG nguồn/đích, prompt, intent, model/config/code/asset hashes, env, elapsed time, PyTorch peak VRAM và nvidia-smi samples, per-case receipt và run receipt.

## CPU trước GPU

```bash
data/cache/level3-qwen21-v1/venv/bin/pip install -r requirements-level3-qwen100-v1.txt
CUDA_VISIBLE_DEVICES='' data/cache/level3-qwen21-v1/venv/bin/python -m pytest \
  tests/level3/test_qwen100_v1.py tests/level3/test_qwen21_v1.py -q
CUDA_VISIBLE_DEVICES='' data/cache/level3-qwen21-v1/venv/bin/python -m tools.level3_qwen100_v1 preflight
```

Env kế thừa env Qwen riêng và thêm W&B 0.23.0, không thay môi trường SGG/Task 3 cũ. Preflight kiểm tra đúng 100 nguồn, quota 50/50, ảnh/hash, revisions/receipts, manifest đã khóa, checkpoint, API Qwen/Diffusers và CUDA 12.8 wheel. Tests CPU gồm quota/trùng ảnh/nhãn giả, revision, output bất biến, chuẩn hóa input, resume, và receipt không phụ thuộc file transport W&B.

Trang xem trước câu lệnh: `runs/level3/qwen100-v1/authoring-v1/instructions.html`. File này không phải kết quả sinh ảnh, mọi nhãn vẫn trống.

## Chạy chính đã được người dùng cho phép

Sau CPU checks và commit:

```bash
bash scripts/run_level3_qwen100_v1.sh
```

Chạy đúng 100 task đã khóa, batch 1, seed 42, 40 bước, output `runs/level3/qwen100-v1/seed42-v1/`. Có online W&B project `level3-qwen100-v1`, chỉ log thông số/trạng thái/timing/VRAM; không log ảnh. Tự đi qua 100 task trong manifest này rồi dừng, không có hàng đợi lượt/seed khác. Mẫu pilot đo khoảng 26–51 s/ảnh chỉ là tham khảo; input/shape mới có thể mất thời gian hơn. Không hứa thời gian dựa trên phép nhân số mẫu trước khi có đo thật.

Mỗi case hoàn tất được đóng receipt ngay. Nếu bị ngắt, dùng output phiên bản mới để tiếp tục:

```bash
RESUME_FROM=runs/level3/qwen100-v1/seed42-v1 \
OUTPUT=runs/level3/qwen100-v1/seed42-v2 \
  bash scripts/run_level3_qwen100_v1.sh
```

Resume kiểm tra manifest/config/asset/code hashes và receipt từng case, tái sử dụng kết quả thành công, chỉ sinh phần thiếu. Không ghi đè run cũ. Nếu code/protocol đã đổi thì từ chối reuse, cần protocol mới có giải thích. Case bị gián đoạn chưa có receipt không được coi là hoàn tất.

## Kiểm tra sau chạy

```bash
python3 -m http.server 8004 --bind 127.0.0.1 \
  --directory runs/level3/qwen100-v1/seed42-v1
```

Mở http://localhost:8004/comparison.html. Trang chia 10 case/trang, xem nguồn/đích và câu lệnh, chấm Edit Fulfillment/Preservation độc lập, xuất `qwen100-reviewed.json`. Nhãn chưa chấm giữ null/pending. Preview RGBA ghép nền trắng, native PNG giữ nguyên alpha. Không tự tạo PASS, không tính F1/Spearman khi chưa có nhãn. `status=ok` chỉ xác nhận inference/PNG/40 bước hoàn tất.

Bước sau lượt này là REACT++ trên nguồn/đích và kiểm tra alignment/Graph Consistency. Không chạy thêm SGG, CLIP hoặc LPIPS trong launcher sinh 100 ảnh này.
