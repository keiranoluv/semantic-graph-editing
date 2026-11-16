# Nhiệm vụ 3 — Vietnam Scene Graph Editing Benchmark v1 (P0/P1)

**Trạng thái:** công cụ tiền xử lý/thu thập/kiểm định CPU đã viết; **chưa tải ảnh thật, chưa có ảnh nào qua duyệt tay, chưa chạy image editor, chưa có consistency score.** Không dùng số liệu synthetic trong báo cáo. Công cụ này không train model và không khởi chạy GPU.

## Phạm vi theo đề

- Bộ đánh giá: 100 ảnh nguồn **tự thu thập có giấy phép** trong bối cảnh Việt Nam, 150–200 câu chỉnh sửa, ảnh đích sinh từ mô hình chỉnh sửa tiền huấn luyện, mọi kết quả được người duyệt đánh dấu đạt/không đạt.
- Đánh giá sau này: graph edit correctness + preservation, Spearman với nhãn tay; đối chứng CLIPScore/LPIPS và 10 ca bất đồng. **Chưa triển khai** các phần này.
- Việc đổi dataset Task 2 sang EMNLP là thay đổi nguồn lớn, vẫn cần giảng viên đồng ý; bộ VN100 không mặc nhiên hợp thức hóa nguồn EMNLP.

## Nguồn & quota

| Key | Dataset release | Quota | License claim | Quan trọng |
|---|---|---:|---|---|
| `sidewalk` | [Vietnamese Sidewalk Violation VQA](https://huggingface.co/datasets/manhdungcr7/dataset_viahe_vqa) | 40 | CC BY-NC 4.0 | Dataset ghi ảnh chụp đường phố chưa làm mờ mặt/biển số; PHẢI review quyền riêng tư từng ảnh. Chỉ dùng nghiên cứu phi thương mại nếu đáp ứng điều khoản. |
| `streetfood` | [Vietnamese Street Food](https://huggingface.co/datasets/HoagMin/Vietnamese_StreetFood_Dataset) | 30 | CC BY 4.0 | ZIP ~222 MB, tải một lần, giải nén duy nhất ảnh ứng viên. Cần kiểm chứng tác giả/quyền của ảnh gốc trước khi chấp nhận. |
| `travel` | [VietTravelVQA-v2 HF mirror](https://huggingface.co/datasets/phunguyendinhkh/VietTravelVQA-v2), [Mendeley canonical v2](https://doi.org/10.17632/fvxtpnh8mh.2) | 30 | Dataset CC BY 4.0 | Ảnh từ Wikimedia Commons, license **riêng cho từng ảnh**. Phải điền URL file Wikimedia, tác giả, giấy phép ảnh gốc. |

**Không giả định dataset-level CC BY đồng nghĩa ảnh gốc được phép phân phối.** Với CC BY-SA hoặc CC BY-NC-SA phải tuân theo điều kiện gốc khi phát hành ảnh chỉnh sửa. Nên chỉ commit hash/metadata thống kê, không commit JPG/PNG, crop hay contact sheet chứa ảnh.

## Cài và chạy pilot: 10 + 10 + 10 ảnh

Chạy từ project root trên máy có internet; môi trường Python >=3.10 đã có Pillow:

```bash
python -m pip install -r requirements-level3-vietnam-v1.txt
# Optional: huggingface-cli login if HF access requires auth, but public sources normally do not.

# Ba lệnh dưới dùng internet và tải tài nguyên. KHÔNG train hay sử dụng GPU.
python -m tools.level3_vietnam_benchmark_v1 fetch --source sidewalk --target pilot
python -m tools.level3_vietnam_benchmark_v1 fetch --source streetfood --target pilot
python -m tools.level3_vietnam_benchmark_v1 fetch --source travel --target pilot

# Offline: kiểm kích thước, decode, SHA256, sinh contact sheets và review.csv
python -m tools.level3_vietnam_benchmark_v1 scan --target pilot
```

Kết quả nằm trong `data/raw/level3/vietnam-benchmark-v1/` và `data/processed/level3/vietnam-benchmark-v1/pilot/` (đã được `.gitignore` bao phủ). Mỗi source có `source-receipt.json` chứa commit SHA upstream được khóa tại lúc tải. Repo HF có thể thay đổi: KHÔNG dùng nội dung `main` mới cho kết quả cũ.

**Review:** mở các `contact_sheets/sheet-*.jpg` để xem nhanh, sau đó đối chiếu ảnh gốc (thực tế) và điền `review.csv`: chỉ đúng 10 ảnh mỗi source được `decision=accept`, các ảnh còn lại để pending hoặc reject. Các cột bắt buộc của ảnh chấp nhận: `creator`, `image_license` (ví dụ `CC BY 4.0`, `CC BY-SA 4.0`, `CC BY-NC 4.0`, `CC0`), `image_license_url`, `license_checked=yes`, `privacy_clear=yes`, `objects_clear=yes`, `edit_feasible=yes`, `reviewer` có thật, nguồn asset. Với `travel`, bắt buộc điền `original_image_url` Wikimedia Commons.

Cách review nội dung: tránh ảnh có mặt/biển số nhận diện được, ảnh mờ, ảnh chỉ có 1 vật thể, ảnh biển quảng cáo/biển hiệu chữ chiếm phần lớn; ưu tiên 2–8 đối tượng có thể gán node và quan hệ rõ ràng. `objects_clear` và `edit_feasible` phải được người duyệt đánh giá thực, không tự set bằng code.

Sau **khi con người đã review**:

```bash
python -m tools.level3_vietnam_benchmark_v1 finalize --target pilot
```

Lệnh từ chối export nếu thiếu quota, thiếu quyền, thiếu người duyệt, dữ liệu nguồn thay đổi, ảnh trùng/nổi bật gần trùng (dHash Hamming <=4), hoặc không qua gate. Kết quả `pilot/locked/`: 30 ảnh có hash/provenance, `manifest.json`, `data-card-draft.json`, `edit_slots.csv` gồm 45 dòng **instruction để trống**, không tự suy đoán thao tác từ ảnh.

## Mở rộng full 100 ảnh

**Sử dụng raw/processed path version khác để giữ nguyên pilot.** Pin `--revision <SHA>` theo `source-receipt.json` tương ứng của pilot, một revision cho mỗi nguồn. Lệnh mẫu:

```bash
RAW=data/raw/level3/vietnam-benchmark-full-v1
OUT=data/processed/level3/vietnam-benchmark-full-v1
python -m tools.level3_vietnam_benchmark_v1 fetch --source sidewalk --target full --raw-root "$RAW" --revision SHA_SIDEWALK
python -m tools.level3_vietnam_benchmark_v1 fetch --source streetfood --target full --raw-root "$RAW" --revision SHA_STREETFOOD
python -m tools.level3_vietnam_benchmark_v1 fetch --source travel --target full --raw-root "$RAW" --revision SHA_TRAVEL
python -m tools.level3_vietnam_benchmark_v1 scan --target full --raw-root "$RAW" --processed-root "$OUT"
# Human review /edit the CSV and check original images
python -m tools.level3_vietnam_benchmark_v1 finalize --target full --raw-root "$RAW" --processed-root "$OUT"
```

Đích full khi qua gate: 100 ảnh (40+30+30), **150 dòng yêu cầu cần con người viết** (`50×1 + 50×2`) với `status=needs_authoring`. Chưa có edited images hoặc labels consistency. Cùng ảnh có hai instruction vẫn phải nằm cùng group nếu chia dev/test. Đề khuyến nghị toàn bộ nhãn tay consistency chỉ dùng để đánh giá; nếu tinh chỉnh score phải tách dev/test theo image_id.

## Thiết kế bước tiếp (chưa chạy)

1. Người duyệt viết 150 câu chỉnh sửa căn cứ ảnh, ghi `edit_type`, đối tượng mục tiêu, các đối tượng/quan hệ phải bảo toàn và nguồn bằng chứng.
2. Chạy editor **suy luận** trên 150 yêu cầu, lưu ảnh đích và seed/weights/config/hash (GPU phải commit code + CPU tests trước, người dùng tự chạy theo AGENTS.md).
3. Trích graph của ảnh nguồn/đích, giữ object identity bằng grounding và alignment giữa hai ảnh; so score edit và preserve; so với oracle graph đã kiểm tay và model Task 2 sau adaptation.
4. Gán nhãn tay đạt/không đạt cho **tất cả** output, chấm Spearman và baseline CLIPScore/LPIPS, phân tích >=10 bất đồng. Không tune trên test.

## Kỹ thuật & hạn chế

- Bước fetch tải tối đa 3× quota để dự phòng ảnh hỏng/kích thước không đạt; bước scan chọn 2× quota làm ứng viên kiểm tay. Bộ scan kiểm file decode, size, SHA và dHash; **không** xác nhận ngữ nghĩa, quyền tác giả hoặc quyền riêng tư. Human reviewer là bắt buộc; dấu `license_checked=yes` nghĩa là người dùng đã xác minh, không phải công cụ xác minh giấy phép.
- `fetch` dùng HF revision SHA tại thời điểm chạy. Download image repo ở mức **chọn lọc**; riêng Street Food ZIP nguyên 222 MB vì upstream không phân phối từng ảnh. Không thực hiện mạng ở `scan`/`finalize`.
- Không copy `data/`, `reviews/`, checkpoint/model hoặc ảnh lên Git. Lưu thêm hash của script/config/protocol khi nộp báo cáo.
- Để đảm bảo độ lặp lại giữa các lần chạy, lựa chọn ứng viên được sắp theo SHA256(seed, source, image path). Nếu dataset upstream thay đổi, phải giữ SHA commit cũ, không dùng `main` mới.
- Mẫu pilot 30 ảnh chỉ để kiểm phương pháp lấy ảnh và review; **không** được báo cáo là đã hoàn thành VN100. Không có VLM/model chỉnh ảnh trong phiên bản này.

## Kiểm thử CPU

```bash
python -m pytest -q tests/level3/test_vietnam_benchmark_v1.py
python -m tools.level3_vietnam_benchmark_v1 --help
```
