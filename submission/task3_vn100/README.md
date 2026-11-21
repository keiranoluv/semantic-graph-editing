# Nhiệm vụ 3 · VN100

qwen_unified/ tạo ảnh theo yêu cầu. qwen_sgg/ trích đồ thị bằng REACT++. evaluation/ chấm đáp ứng chỉnh sửa, bảo toàn và nhất quán. submission/demo.py đọc kết quả đã lưu cho sáu ca notebook.

configs/level3/qwen200-cohort-v1.json khóa 200 yêu cầu, seed 42. Dataset, ảnh, data card và nhãn ở Hugging Face theo README gốc.

Các môi trường tương ứng là environments/qwen-edit.txt, react-graph.txt và image-evaluation.txt. Mỗi CLI có --help và bước check trước khi suy luận. Chỉ chạy một bước GPU mỗi thời điểm. Notebook tính lại điểm từ đồ thị đã trích trên CPU. qwen21/, image_graph/ và shared/ cung cấp runtime và kiểm tra checkpoint.
