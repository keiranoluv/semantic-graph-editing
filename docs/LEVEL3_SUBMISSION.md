# Bàn giao Task 3 — archive, notebook và báo cáo

Gói local lưu đầy đủ bằng chứng của cohort seed42: **100 nguồn, 200 ảnh edit, 200 instruction và 300 prediction SGG**, cùng nhãn và điểm. Tên file ZIP chứa commit code để xác định phiên bản. Dùng gói này để sao lưu hoặc chuyển máy; chưa có thao tác upload lên Kaggle/Hugging Face.

## Tạo và kiểm tra gói

Từ root repo, sau khi commit code/tài liệu:

```bash
bash scripts/package_level3.sh
```

Lệnh chạy CPU, đọc các index/receipt đã khóa, xác minh ảnh và graph, rồi tạo:

```text
runs/level3/submission-v1/
  portable-manifest.json
  level3-seed42-<commit>.zip
  level3-seed42-<commit>.zip.sha256
  level3-seed42-<commit>.verification.json
```

ZIP gồm code **đã commit tại HEAD**, không lấy các sửa đổi cá nhân chưa commit. Artifacts ngoài Git được chọn từ cohort và các receipt, không thu gom toàn bộ workspace. Mọi thành viên có checksum trong `BACKUP_INDEX.json`; sau nén, công cụ mở lại và xác minh từng file. Lặp lại trên cùng commit kiểm tra archive hiện có. Khi nguồn thay đổi phải tạo output version mới; archive cũ không bị ghi đè. Nếu nén bị ngắt, file `.partial` không phải bản sao lưu hợp lệ; dùng `--output` khác để tạo lại.

Kiểm tra lại ZIP trước hoặc sau khi chuyển máy:

```bash
bash scripts/package_level3.sh verify --archive /duong/dan/level3-seed42-<commit>.zip
# Hoặc kiểm checksum toàn file trong thư mục chứa ZIP:
sha256sum -c level3-seed42-<commit>.zip.sha256
```

Archive và file `.sha256` nên được sao chép cùng nhau. Checksum xác nhận tính toàn vẹn, không phải chữ ký xác thực tác giả.

## Giải nén và xem demo ở thư mục bất kỳ

Giải nén vào **thư mục mới, trống** để tránh ghi đè workspace đang dùng:

Môi trường demo đã kiểm tra với Python 3.12; không cần PyTorch hoặc CUDA.

```bash
mkdir -p /duong/dan/level3-restored
python -m zipfile -e /duong/dan/level3-seed42-<commit>.zip /duong/dan/level3-restored
cd /duong/dan/level3-restored
python -m venv .venv-demo
.venv-demo/bin/pip install -r requirements-level3-demo-v1.txt
.venv-demo/bin/jupyter lab notebooks/level3_task3_demo.ipynb
```

Notebook tìm `portable-manifest.json` ở root đã giải nén. Nếu mở trực tiếp trong workspace gốc, chuẩn bị manifest bằng `bash scripts/package_level3.sh prepare`. Chọn kernel từ môi trường vừa cài, chạy Run All. Có thể mở `DEMO_PREVIEW.html` nếu bản đóng gói có kèm preview đã chạy; preview không thay thế việc thực thi notebook.

Notebook xem ảnh/bbox/graph thật, **tính lại toàn bộ 200 điểm graph và tương quan trên CPU**, đối chiếu CLIPScore/LPIPS cùng nhãn đã lưu. Phần Task 2 replay predicted operations của GNN/seed42 trên EMNLP, tách riêng vì chưa có adapter được kiểm chứng sang graph REACT++. Không chạy model thị giác, không train, không tự khởi chạy hàng đợi.

HTML phân tích 10 ca có sẵn tại:

`runs/level3/evaluation-v1/seed42/reports/0fc28934ad71812a/analysis-final-v1/index.html`

Có thể mở trực tiếp hoặc dùng `python -m http.server` tại thư mục chứa HTML. Tất cả media của trang này nằm trong gói.

## Tái tạo ảnh và SGG

Hai bước vẫn độc lập, theo [LEVEL3_PIPELINE.md](LEVEL3_PIPELINE.md):

```bash
RUN_ID=reproduce01 bash scripts/run_level3_qwen.sh
RUN_ID=reproduce01 bash scripts/run_level3_sgg.sh
```

Đây là **lệnh GPU do người dùng chạy** sau khi chuẩn bị môi trường và checkpoint chính thức. Qwen-Image-2.1: BF16, batch 1, seed42, 40 steps, CPU offload. REACT++: batch 1, seed42, ONNX đã khóa. Mỗi lệnh chỉ chạy phần còn thiếu của lượt tương ứng, không tự gọi bước kia. Output mới ở `runs/level3/qwen200/reproduce01` và `runs/level3/qwen-sgg/reproduce01`.

Gói sao lưu không chứa weights/virtualenv. Dùng requirements, asset locks và hướng dẫn [Qwen](LEVEL3_QWEN.md), [SGG](LEVEL3_SGG.md), [evaluation](LEVEL3_EVALUATION.md) để chuẩn bị lại các môi trường. Hai launcher gốc và manifests lịch sử có một số đường dẫn tuyệt đối tới `/workspace/workspace/semantic-graph-editing`: để resume/tái lập bằng nguyên launcher đã khóa, khôi phục repo và artifacts tại vị trí này (và dùng Git checkout đúng commit). **Notebook portable không có ràng buộc vị trí đó.** Chuyển đường dẫn cho một lượt inference mới phải có metadata/version mới; không sửa trực tiếp manifest lịch sử để vượt kiểm hash.

## Tài liệu cần mở

- [Data card và nguồn/giấy phép công bố](LEVEL3_DATA_CARD.md).
- [Báo cáo Task 3 bằng Markdown](LEVEL3_TASK3_REPORT.md).
- [Bảng số liệu và phân tích đã chốt nhãn](LEVEL3_RESULTS_V1.md).
- `notebooks/level3_task3_demo.ipynb`: notebook mới, không thay notebook Level 1 cá nhân.

## Lưu thêm trên dịch vụ ngoài

Đã chọn phương án nén local trong lượt bàn giao này. Có thể tải ZIP và checksum lên kho lưu trữ của nhóm hoặc một repository dataset private trên Hugging Face để có bản sao ngoài máy. Việc upload cần chọn tài khoản/repository và quyền truy cập cụ thể; công cụ đóng gói không tự đăng nhập, tạo repository hay công bố dữ liệu. Nếu phát hành public, giữ bảng nguồn và điều kiện giấy phép trong data card, không gắn một giấy phép mới chung cho mọi ảnh.
