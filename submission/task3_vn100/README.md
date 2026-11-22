# Nhiệm vụ 3 · VN100

Pipeline dùng Qwen-Image-2.1 và REACT++ pretrained. Qwen tạo 200 ảnh chỉnh sửa từ 100 ảnh nguồn. REACT++ trích 100 đồ thị nguồn và 200 đồ thị đích. Bộ chấm so khớp đồ thị để đo đáp ứng yêu cầu và bảo toàn.

Chạy từ gốc submission/ trong các môi trường tương ứng environments/qwen-edit.txt, react-graph.txt và image-evaluation.txt. Cần bộ ảnh VN100, checkpoint pretrained và các cache được liệt kê trong config.

## Tạo ảnh và đồ thị

    python -m task3_vn100.generate preflight --output runs/level3/qwen200/seed42
    python -m task3_vn100.generate infer --output runs/level3/qwen200/seed42 --no-import
    python -m task3_vn100.extract_graphs preflight --output runs/level3/qwen-sgg/seed42
    python -m task3_vn100.extract_graphs infer --output runs/level3/qwen-sgg/seed42 --no-import --max-new-images 300

Qwen dùng configs/level3/qwen-unified-v1.json, batch 1, seed 42, 40 diffusion steps, resolution 1024, BF16 và model CPU offload. REACT++ dùng configs/level3/qwen-sgg-unified-v1.json. Mỗi lệnh infer kết thúc sau phạm vi của mình. Chỉ chạy một mô hình GPU mỗi thời điểm.

## Đánh giá và demo

    python -m task3_vn100.evaluation check
    python -m task3_vn100.evaluation graph --device cpu
    python -m task3_vn100.evaluation metrics --device cuda

Các bước chấm đọc đầu vào đã khóa trong configs/level3/evaluation200-v1.json và lưu theo DEFAULT_OUTPUT trong evaluation/core.py. Khi đổi thư mục tạo ảnh/đồ thị, cần cấu hình đánh giá có provenance riêng trỏ đến lượt đó.

Notebook tổng hợp tính lại điểm sáu ca từ đồ thị đã trích. demo.py đọc kết quả, generation/ chứa runtime Qwen, graph_extraction/ điều phối REACT++, models/ chuyển dự đoán sang đồ thị và evaluation/ chấm điểm. scoring.py định nghĩa điểm đồ thị; qwen_assets.py và react_assets.py kiểm tra checkpoint pretrained. Data card và nguồn ảnh ở assets/datasets/vn100/.
