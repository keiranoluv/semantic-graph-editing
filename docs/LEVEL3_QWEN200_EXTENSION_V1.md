# Thêm 100 yêu cầu: bộ 200 cặp ảnh/yêu cầu Level 3

Đã duyệt lại đủ 100 ảnh nguồn trên 13 contact sheets, viết thêm một câu cho mỗi ảnh và giữ nguyên 100 câu cũ. Mỗi yêu cầu mới chạy **độc lập từ ảnh nguồn gốc**, không dùng ảnh Qwen của lượt trước làm đầu vào. Tổng bộ đánh giá gồm 100 ảnh nguồn, mỗi ảnh hai yêu cầu, đúng quy mô 100 ảnh / 150–200 yêu cầu ở trang 3–4 của đề. Phần này chưa thay thế việc gán nhãn tay toàn bộ kết quả, tính tương quan và các baseline ảnh của Task 3.

| Thao tác | 100 cũ | 100 mới | Tổng 200 |
|---|---:|---:|---:|
| Thêm đối tượng | 66 | 40 | 106 |
| Xóa đối tượng | 28 | 38 | 66 |
| Thay đối tượng | 6 | 12 | 18 |
| Đổi màu | 0 | 10 | 10 |

Lượt mới vẫn có 50 ảnh sidewalk và 50 ảnh travel; 5 câu màu trên mỗi nguồn. 100 câu mới không trùng chuỗi instruction với nhau hoặc với câu cũ. Tổng 200 cặp có 199 chuỗi instruction khác nhau vì câu cũ số 69 và 89 trùng lời nhưng dùng hai ảnh nguồn khác nhau; hai mục cũ được giữ nguyên. Task ID mới kết thúc bằng `-q200-2`, khác hoàn toàn ID `-q100-1` của lượt đầu.

## Manifest và xem trước

- `configs/level3/qwen200-extension-instructions-v1.json`: chỉ 100 câu mới, số thứ tự 101–200, đối tượng/vị trí, quan sát nguồn, tác giả assistant và liên kết câu cũ.
- `configs/level3/qwen200-cohort-v1.json`: 200 mục; 100 mục đầu giữ nguyên nội dung JSON của manifest cũ, 100 mục sau đúng bằng extension.
- `configs/level3/qwen200-extension-v1.json`: config riêng, pin hash cả ba manifest/config liên quan, checkpoint và receipt của lượt cũ đã hoàn tất.
- `runs/level3/qwen200-extension-v1/authoring-v1/instructions.html`: ảnh nguồn với câu cũ và câu mới cạnh nội dung ảnh.
- `runs/level3/qwen200-extension-v1/authoring-v1/instructions.csv`: 100 câu mới và câu cũ tương ứng, mở được trong bảng tính.

Source bytes, revision, URL, license claims và trạng thái kiểm chứng quyền ảnh giữ nguyên theo manifest nguồn. Không sửa dataset, graph, output hoặc nhãn tay cũ. Instruction do assistant soạn, không gọi là annotation người dùng. Đây là phần mở rộng phát triển sau khi đã xem kết quả lượt đầu, không phải bộ test mù độc lập. 100 ảnh đều được dùng lại, không chọn lọc ảnh dựa trên kết quả inference mới.

Xem trước nếu cần:

```bash
python -m http.server 8008 --bind 127.0.0.1 \
  --directory runs/level3/qwen200-extension-v1/authoring-v1
```

Mở `http://localhost:8008/instructions.html`; máy remote cần forward cổng 8008. Đây là trang instruction, chưa có ảnh đích mới. Không thay trang review graph cũ ở 8007.

## 10 câu màu và giới hạn của SGG

Các request màu đã cố định trước inference: **109, 115, 116, 125, 141, 154, 156, 164, 170, 180**. Ví dụ: đổi ghế vàng thành xanh; đổi phần thân xe hồng thành xanh; đổi cabin xe tải trắng thành xanh; đổi các sọc xanh trên hải đăng thành đỏ. Chỉ yêu cầu đổi màu một đối tượng hoặc một phần được mô tả rõ, giữ phần cảnh còn lại.

Mỗi mục có `operation.kind=attribute`, `attribute=color`, màu nguồn/đích, `evaluation_group=color_attribute_probe` và `sgg_edit_fulfillment_support=unsupported_color_attribute`. Đây là **giả thuyết kiểm tra giới hạn biểu diễn**, không phải nhãn edit thất bại hoặc thành công. REACT++ hiện không xuất attribute màu; graph giống nhau không chứng minh đã đổi đúng màu. Ngược lại, việc đổi màu có thể làm detector/relations dao động nên cũng không mặc định graph phải giữ nguyên.

Bộ chấm graph 100 mẫu v1 vẫn khóa manifest cũ và chỉ hỗ trợ add/remove/replace; chưa đưa 100 yêu cầu mới vào scorer đó. Khi mở rộng chấm, các ca màu cần giữ trạng thái không hỗ trợ/không đủ bằng chứng cho Edit Fulfillment thuần graph và được kiểm tra từ ảnh thật, nhãn tay hoặc baseline ảnh. Không gán E=1 chỉ vì node/edge còn nguyên, cũng không gán E=0 vì extractor thiếu color head. Không chỉnh ngưỡng để ép ra kết luận mong muốn trên nhóm này.

## Lệnh inference chỉ 100 yêu cầu mới

Chạy từ root repo sau commit:

```bash
bash scripts/run_level3_qwen200_extension_v1.sh
```

Launcher dùng môi trường riêng đã chạy được lượt đầu: `data/cache/level3-qwen21-v1/venv`. Mô hình **Qwen/Qwen-Image-2.1**, revision `d26bb61231c349cf6b7896fa83353113880e1ba3`; Diffusers pin theo config Qwen cũ; PyTorch CUDA 12.8 có `sm_120` cho RTX 5090. BF16, **batch size 1**, **seed 42 đặt lại cho từng yêu cầu**, **40 steps**, `output_resolution=1024`, model CPU offload và VAE tiling. Không nâng dependency, không huấn luyện, không tự thử seed khác hoặc lọc ảnh theo chất lượng.

Output mặc định: `runs/level3/qwen200-extension-v1/seed42-v1/`. Lưu từng câu và provenance, ảnh nguồn/đích, số bước thực tế, thời gian, peak VRAM allocator và mẫu đo tổng VRAM, config/môi trường/code hashes, receipts. W&B online project **`level3-qwen200-extension-v1`**. Sau 100 yêu cầu mới, lệnh dừng; **không chạy lại 100 cũ, không tự chạy graph, không có lượt tiếp theo trong hàng đợi**. Dùng cùng GPU lock với launcher cũ để tránh hai mô hình cùng chạy.

Code/config/manifest phụ thuộc phải sạch và đã commit trước khi runner dùng CUDA. Output đã tồn tại sẽ bị từ chối ghi đè. Nếu muốn một lượt mới độc lập:

```bash
OUTPUT=runs/level3/qwen200-extension-v1/seed42-rerun-01 \
  bash scripts/run_level3_qwen200_extension_v1.sh
```

Nếu bị ngắt, tiếp tục vào **thư mục mới**, giữ các case thành công của lần trước:

```bash
RESUME_FROM=runs/level3/qwen200-extension-v1/seed42-v1 \
OUTPUT=runs/level3/qwen200-extension-v1/seed42-resume-01 \
  bash scripts/run_level3_qwen200_extension_v1.sh
```

Resume kiểm tra protocol/code hashes và receipt từng case, chỉ tái sử dụng case đã hoàn tất đủ 40 bước. Case lỗi hoặc chưa chạy được thực hiện tiếp; không sửa thư mục cũ. SIGTERM/Ctrl+C được ghi lỗi và kết quả dở dang; mất điện/SIGKILL vẫn có thể tái dùng các case đã đóng receipt. Nếu OOM, runner dừng và ghi `failure.json`, không tự đổi resolution hoặc steps. Log OOM trong CPU tests là mô phỏng, chưa có OOM thực tế của lượt mới.

## Xem kết quả sau inference

```bash
python -m http.server 8008 --bind 127.0.0.1 \
  --directory runs/level3/qwen200-extension-v1/seed42-v1
```

Mở `http://localhost:8008/comparison.html`. Nếu đang dùng 8008 xem instruction, dừng server đó rồi mở server kết quả. Trang Qwen chỉ có **Đúng yêu cầu** và **Bảo toàn**, không notes hoặc Pix2Pix; lưu lựa chọn tại trình duyệt, tải `qwen200-extension-reviewed-v1.json`. Nhãn mặc định đều trống; ảnh chưa sinh được không thể chấm. File tải về chỉ chứa 100 yêu cầu mới, độc lập với nhãn 100 cũ.

## Kiểm tra trước bàn giao

Đã chạy **45 tests CPU đạt**, gồm contracts mới và hồi quy Qwen100/Qwen21. Có mô phỏng pipeline CPU cho đủ 100 request, interruption/OOM, 39/40 bước và resume chỉ chạy case còn thiếu; đây không phải GPU smoke test. Full preflight CPU đã kiểm tra hash checkpoint, Diffusers API và wheel CUDA 12.8/Blackwell với `CUDA_VISIBLE_DEVICES=''`. Bash syntax, formatting/lint và trình duyệt đều đạt: 100 ảnh source/100 câu mới, hai trường review, lưu/tải nhãn và khóa case chưa có target.

```bash
CUDA_VISIBLE_DEVICES='' data/cache/level3-sgg-pretrained-v2/venv/bin/python -m pytest \
  tests/level3/test_qwen200_v1.py tests/level3/test_qwen100_v1.py tests/level3/test_qwen21_v1.py -q
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  data/cache/level3-qwen21-v1/venv/bin/python -m tools.level3_qwen200_v1 preflight
```

Artifacts kiểm tra nằm trong `runs/level3/qwen200-extension-v1/validation-v1/`. **Chưa chạy GPU hoặc tạo target thật cho 100 câu mới**; lượt này bàn giao lệnh cho người dùng chạy. Không tự tạo điểm chất lượng, F1 hoặc nhãn tay.
