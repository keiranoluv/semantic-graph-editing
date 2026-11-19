# NV3 mở rộng · Đối chứng Qwen3-VL trên VN100

Judge đọc **ảnh nguồn, ảnh đã chỉnh và nguyên văn yêu cầu tiếng Anh** của từng cặp đã duyệt. Hai tiêu chí là thực hiện yêu cầu và bảo toàn phần còn lại của ảnh. Mỗi tiêu chí dùng thang nguyên 0–4. Một câu tiếng Anh ghi lại bằng chứng, và điểm tổng hợp là `J = fulfillment × preservation / 16`. Hai điểm gốc được giữ trong kết quả. [Prompt và rubric nguyên văn](../configs/level3/mllm-judge-prompt-v1.txt) · [Cấu hình và SHA-256](../configs/level3/mllm-judge-v1.json)

## Chuẩn bị trên CPU

Chạy các lệnh từ thư mục gốc dự án. Launcher mặc định dùng môi trường NV3 có sẵn ở `data/cache/level3-evaluation-v1/venv/bin/python`. Để tạo môi trường riêng, dùng Python 3.12, cài `requirements-level3-mllm-judge-v1.txt` và đặt biến `PYTHON` thành đường dẫn đến interpreter đó.

```bash
bash scripts/run_level3_mllm_judge_v1.sh prepare
bash scripts/run_level3_mllm_judge_v1.sh check
```

`prepare` tải và xác minh 14 file gồm 4 shard trọng số, tokenizer, chat template và cấu hình processor của revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`. Workspace hiện tại đã có snapshot trong cache Hugging Face. `check` đối chiếu hash ảnh nguồn/đích, manifest và instruction của 200 cặp thuộc 100 ảnh nguồn, xác minh 5 cặp smoke ngoài VN100 rồi chạy processor trên CPU. Bước GPU đọc manifest chỉ chứa ID, instruction, đường dẫn ảnh và hash. [Manifest VN100 cho judge](../configs/level3/mllm-judge-vn100-v1.json) [Model đã khóa](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/tree/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b)

## Smoke GPU và khóa bản chạy

```bash
bash scripts/run_level3_mllm_judge_v1.sh smoke
```

Lệnh chạy một model **Qwen3-VL-8B-Instruct** đóng băng trên một RTX 5090, BF16, attention SDPA, batch 1 cặp, seed 42, greedy, tối đa 128 token mới. Hai ảnh được chuyển RGB và thu nhỏ theo tỉ lệ bằng LANCZOS tới cạnh dài tối đa 1024 px. Khi kích thước chưa chia hết cho 32, ảnh được đặt giữa khung với viền xám trung tính, tổng độ đệm tối đa 31 px mỗi chiều. Grid sau processor phải khớp khung thực tế. Smoke chỉ chạy 5 cặp MagicBrush rồi dừng. [Danh sách ảnh và yêu cầu smoke](../configs/level3/mllm-judge-smoke-v1.json)

Mở hai ảnh theo thứ tự `source` rồi `target` trong manifest và đọc năm câu evidence in ở terminal. Kiểm tra mỗi câu diễn giải được thay đổi và phần cảnh còn lại. Xem raw output, số token, kích thước từng khung, thời gian và peak allocated/reserved của PyTorch (MiB) trong `runs/level3/mllm-judge-v1/smoke/<hash>/cases/`. Console và `complete.json` có hai peak riêng khi nạp model và khi chấm. Đây là peak bộ nhớ do PyTorch quản lý trong process. W&B chỉ bật ở lượt chấm chính.

Khi đủ 5/5 JSON hợp lệ, evidence đã được đọc và lượt smoke hoàn tất trong bộ nhớ GPU, chạy trên CPU để ghi nhận và commit bản đã kiểm tra. Lệnh commit giới hạn ở tệp release. Bản này ràng buộc hash code, prompt, manifest hai tập, cấu hình, môi trường và receipt smoke.

```bash
bash scripts/run_level3_mllm_judge_v1.sh freeze
git add -- configs/level3/mllm-judge-release-v1.json
git commit --only -m "Freeze VN100 judge protocol after five GPU smoke cases" -- configs/level3/mllm-judge-release-v1.json
```

Nếu smoke gặp OOM, giải phóng VRAM rồi gọi lại `smoke`. Launcher lưu thư mục lỗi riêng và bắt đầu một lượt đủ năm cặp. Khi cần sửa giao thức trước freeze, commit phần sửa rồi chạy lại smoke. Thay đổi sau release dùng một phiên bản thực nghiệm mới.

## Chấm 200 cặp và xuất báo cáo

Sau khi release đã commit và đã đăng nhập W&B trong môi trường chạy, dùng lệnh GPU độc lập sau.

```bash
bash scripts/run_level3_mllm_judge_v1.sh score
```

Cấu hình GPU giống smoke. Launcher chỉ chấm 200 cặp VN100 theo thứ tự manifest, một model, một seed, từng cặp một, sau đó dừng. W&B online ghi cấu hình, tiến độ và bộ nhớ. Ảnh và bằng chứng văn bản nằm cục bộ. Thư mục đầu ra là `runs/level3/mllm-judge-v1/seed42/`. Khi hạ tầng gián đoạn, gọi lại cùng lệnh. Launcher kiểm tra hash từng kết quả và chỉ infer các cặp còn thiếu. Một phản hồi sai định dạng được lưu nguyên văn với trạng thái `invalid_format` và điểm khuyết.

Khi đủ 200 bản ghi, chạy bước CPU sau.

```bash
bash scripts/run_level3_mllm_judge_v1.sh report
```

Trong `runs/level3/mllm-judge-v1/seed42/report/`, `results.csv` có một hàng mỗi cặp, `summary.json` chứa ID sử dụng và khoảng bootstrap, `report.md` có bảng kết quả. Nhãn tay cuối và baseline chỉ được ghép ở bước phân tích, qua `task_id` và binding của đúng cặp ảnh.

| Tập cố định | Câu hỏi |
|---|---|
| 91 cặp, 62 ảnh nguồn, 82 đạt cả hai | So ρ của J, G, CLIPScore và −LPIPS với nhãn tổng hợp trên cùng các ID có đủ bốn điểm. Ước lượng Δρ của J với từng baseline. |
| 200 cặp, 100 ảnh nguồn, 175 đạt cả hai | Báo độ phủ, ρ của từng tiêu chí với nhãn tay tương ứng và ρ của J với nhãn tổng hợp. |
| 109 cặp G khuyết, 71 ảnh nguồn, 93 đạt cả hai | Báo riêng các số đo của Qwen trên các cặp đồ thị đã bỏ qua. |

Nhãn tổng hợp là cả hai tiêu chí cùng đạt. Bootstrap percentile 95% lấy mẫu 2.000 lần theo cụm ảnh nguồn với seed 42, giữ các edit của cùng ảnh trong cụm và dùng chung mỗi lượt lấy mẫu giữa các chỉ số. Vector hằng cho kết quả khuyết. Mỗi bảng ghi mẫu số, số lỗi định dạng và số lượt bootstrap hợp lệ. Bước phân tích kiểm lại ρ trên 91 ID gốc trước khi lấy giao với Qwen. Giá trị đến bốn chữ số thập phân lần lượt là 0,1767 (G), 0,0561 (CLIPScore) và 0,1962 (−LPIPS). [Kết quả NV3 đã chốt](LEVEL3_RESULTS_V1.md)

Nhãn tay đến từ một người duyệt. Có 25/200 ca trượt ít nhất một tiêu chí. Đây là đối chứng một seed trên bộ ảnh đã phân tích. Ghi rõ phạm vi đó khi báo cáo chênh lệch giữa các bộ chấm.

## Kiểm tra tái lập

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  data/cache/level3-evaluation-v1/venv/bin/python -m pytest -q \
  tests/level3/test_mllm_judge_v1.py tests/level3/test_final_analysis.py tests/level3/test_evaluation200.py
```

Đã xác minh SHA-256 của toàn bộ checkpoint và chạy thử model BF16 thật trên CPU với cặp ngoài VN100 `magicbrush:124157:1`. Model trả JSON hợp lệ trong 49 token mới. Kết quả này tách khỏi receipt smoke GPU và bảng thực nghiệm. Test tích hợp dùng đầu ra judge giả lập để kiểm tra các pha, việc tiếp tục sau ngắt, trường hợp thiếu điểm, phép đối chiếu trên cùng ID và các định dạng xuất.
