# Mở rộng video · PVSG

m2/ chuẩn bị đặc trưng vùng và track. m3/models.py cài MLP, GAT tĩnh và GAT không gian–thời gian. m4/models.py bổ sung mô hình thời gian. m5/run.py chấm checkpoint được chọn trên validation.

manifests/ khóa split theo video. Các thư mục m3_150/, m4_150/, m4_150_bs32/ và calibration chứa cấu hình các lượt so sánh. provenance/ ghi nguồn PVSG và các artifact gốc.

PVSG cần video và cache đặc trưng riêng. Tệp môi trường nằm cùng m2/, m3/, m4/. Chạy module từ thư mục gốc bản nộp. Dùng python -m extension_pvsg.m5.run --help để xem bước kiểm tra và chấm điểm.
