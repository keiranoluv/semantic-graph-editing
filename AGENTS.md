# Quy ước làm việc của dự án

- Giao tiếp với người dùng bằng tiếng Việt.
- Trước mỗi tác vụ tính toán GPU (train, inference, benchmark), hoàn tất code
  và kiểm tra phù hợp trên CPU, commit code, rồi gửi người dùng lệnh chạy cùng
  mô tả phạm vi. Người dùng tự chạy; assistant không tự khởi chạy thêm tác vụ GPU.
- Khi bàn giao launcher, nói rõ mô hình, batch size, seed, output và việc có
  tự chạy tiếp các lượt trong hàng đợi hay không. Huấn luyện tuần tự, một model
  mỗi thời điểm.
- Giữ nguyên source/manifest/config đã khóa của các thực nghiệm cũ; phần mở
  rộng cần version và provenance riêng. Không thêm/sửa `src/` khi launcher cũ
  còn khóa glob nguồn ở thư mục này.
- `notebooks/level1_scene_graph_demo.ipynb` là thay đổi cá nhân của người dùng;
  không ghi đè hoặc đưa vào commit công việc khác.
