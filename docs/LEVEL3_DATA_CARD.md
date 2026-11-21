# Data card — Vietnam Graph Editing 200, seed 42

## Mục đích và phạm vi

Bộ đánh giá nội bộ của Project 1, Nhiệm vụ 3: kiểm tra ảnh sau chỉnh sửa bằng đồ thị cảnh, đối chiếu với nhãn người duyệt và các chỉ số ảnh. Phiên bản này gồm **100 ảnh nguồn, 200 yêu cầu độc lập, 200 ảnh Qwen-Image-2.1 và 300 prediction REACT++**. Mỗi ảnh nguồn có hai yêu cầu; ảnh đích thứ hai cũng được sinh từ ảnh nguồn, không tiếp tục chỉnh ảnh đích thứ nhất.

Ngày đóng gói: **16/11/2026, Asia/Bangkok**. Artifact giữ seed 42 và protocol cũ; đóng gói không thay nhãn, instruction, graph hoặc kết quả.

## Nguồn dữ liệu và ghi công

| Thành phần | Số ảnh | Nguồn | Giấy phép công bố ở nguồn |
|---|---:|---|---|
| Bối cảnh vỉa hè Việt Nam | 50 | [Vietnamese Sidewalk Violation VQA — manhdungcr7/dataset_viahe_vqa](https://huggingface.co/datasets/manhdungcr7/dataset_viahe_vqa) | CC BY-NC 4.0 |
| Du lịch, di sản và phong cảnh Việt Nam | 50 | [VietTravelVQA-v2 — phunguyendinhkh](https://huggingface.co/datasets/phunguyendinhkh/VietTravelVQA-v2); [Mendeley Data, version 2, DOI 10.17632/fvxtpnh8mh.2](https://data.mendeley.com/datasets/fvxtpnh8mh/2) | CC BY 4.0 cho bản phát hành dataset |

Thông tin giấy phép ở bảng được đối chiếu trên các trang nguồn ngày 16/11/2026. Bản Sidewalk có điều kiện phi thương mại. VietTravelVQA dẫn nguồn ảnh Wikimedia Commons và giữ điều kiện ghi công/giấy phép ảnh gốc. Data card này ghi nhận nguồn và điều khoản công bố ở cấp dataset; không tạo xác nhận kiểm tra quyền riêng cho từng ảnh. Metadata lịch sử được giữ nguyên.

Revision được dùng để lấy ảnh, không phải `main` mới nhất:

- Sidewalk: `2b8b043de9604a41e9366a80d24ea8c54c67c6f4`.
- VietTravelVQA-v2: `1aa7c4f3c3bd179b94c3093678515af7f6f3da48`.

Từng ảnh có ID, đường dẫn upstream, URL tại revision, SHA256 và kích thước trong `configs/level3/qwen200-cohort-v1.json`; bảng `sources` của `portable-manifest.json` trỏ đến bản ảnh trong gói. Các tác giả của VietTravelVQA được ghi tại DOI gốc; nhóm đồ án xây tập con/chỉnh sửa/graph/nhãn đánh giá, không nhận là tác giả các ảnh nguồn.

## Chọn ảnh và tạo yêu cầu

- Assistant đã xem 130 ảnh ứng viên qua contact sheet và chọn cố định 50 ảnh mỗi nguồn trước lượt Qwen 100 yêu cầu đầu. Đây là chọn có chủ đích cho bối cảnh Việt Nam và khả năng chỉnh sửa, không phải lấy mẫu ngẫu nhiên đại diện toàn bộ hai dataset.
- 100 yêu cầu bổ sung dùng lại đúng 100 nguồn, tạo thành hai yêu cầu độc lập mỗi ảnh. Các yêu cầu được viết với hỗ trợ của assistant, chủ yếu chỉnh một đối tượng; thêm một nhóm nhỏ đổi màu để khảo sát giới hạn SGG.
- Instruction đưa vào editor bằng tiếng Anh. Manifest lưu cả loại thao tác, nhãn đối tượng, mô tả vị trí, nguồn tác giả và bằng chứng ảnh.

| Loại yêu cầu | Số cặp |
|---|---:|
| Thêm đối tượng | 106 |
| Xóa đối tượng | 66 |
| Thay đối tượng | 18 |
| Đổi màu | 10 |
| Tổng | 200 |

## Cách tạo dữ liệu dẫn xuất

**Ảnh đích:** checkpoint chính thức `Qwen/Qwen-Image-2.1`, revision `d26bb61231c349cf6b7896fa83353113880e1ba3`; BF16, batch 1, seed 42, 40 steps, CPU offload, RTX 5090/CUDA 12.8. Lưu ảnh nguồn đã chuẩn hóa EXIF/RGB, ảnh đích, prompt, cấu hình, thời gian, VRAM và receipt. Ảnh nguyên bản tải từ nguồn cũng được giữ.

**Scene graph:** REACT++ PSG, `maelic/REACTPlusPlus_PSG`, revision `25aaf1838c3f6cee32f69a43871bab0870af4bea`, ONNX `yolo12m/react_pp_yolo12m.onnx`. Lưu output native, decoded JSON, ảnh input chuẩn hóa, EditGraph, bbox, confidence và provenance. Có 300 prediction cho 100 nguồn + 200 đích; 400 graph binding theo cặp vì mỗi nguồn dùng hai lần. Không có attribute màu; không xem graph dự đoán là ground truth.

**Điểm:** E/P/G tính từ graph và grounding nguồn đã khóa. CLIPScore dùng ViT-B/32 pretrained và caption đích mong muốn; LPIPS dùng AlexNet v0.1 trên source/target. Không huấn luyện hoặc fine-tune các mô hình thị giác.

## Nhãn người duyệt

Mỗi kết quả có hai trường boolean: `edit_fulfillment` và `preservation`. Nhãn tổng hợp là AND của hai trường. Người dùng xác nhận mỗi trường để trống có nghĩa là đạt; chỉ chuẩn hóa các null thành true, giữ nguyên mọi true/false được điền rõ.

- 200 cặp có nhãn cuối: **175 đạt cả hai / 25 không đạt ít nhất một tiêu chí**.
- Không đạt thực hiện yêu cầu: 10; không đạt bảo toàn: 18; không đạt cả hai: 3.
- Có **287 trường trống ở 144 cặp** được chuẩn hóa theo chỉ dẫn trên, kèm provenance từng trường.
- Người dùng xem lại nhóm 25 ca không đạt và xác nhận “Không sửa, dùng bộ nhãn hiện có”.
- Raw `qwen200-reviewed.json` được giữ nguyên; SHA256 `3e71a883fad961e7c5fc9a503b97b841eee92789c4c7eca0f5a97e6b50ca3e54`.

Đây là một bộ nhãn của người duyệt; chưa đo độ đồng thuận nhiều annotator. Nhận xét của assistant trong báo cáo không thay thế nhãn này.

## Chia dữ liệu và sử dụng đúng phạm vi

Toàn bộ 200 cặp là bộ đánh giá phát triển; không có tuyên bố held-out test độc lập. Nhãn không dùng để chỉnh scorer, ngưỡng hoặc caption trong phiên bản này. Các phân tích sau khi thấy nhãn phải được mô tả là hậu kiểm. Nếu phát triển scorer v2, chia theo **100 `image_id`**, giữ cả hai yêu cầu của một ảnh trong cùng nhóm, công bố phần dùng để phát triển; không gọi phần đã dùng chỉnh phương pháp là test chưa xem.

Bộ này phù hợp cho demo/đánh giá Task 3 và phân tích lỗi. Nó chưa đại diện mọi bối cảnh Việt Nam hoặc mọi loại edit. Độ phủ G chỉ 91/200; 10 yêu cầu màu nằm ngoài biểu diễn. Không có graph GT, nên chưa có object/relation precision, recall, F1 chuẩn của SGG trên bộ này.

## Đọc một mẫu VN100

Đơn vị đánh giá là một yêu cầu (`task_id`). Dùng `image_id` để nối hai yêu cầu với cùng ảnh nguồn. Với mẫu `travel-a4fe5a686b0396-q200-2`, ảnh nguồn thuộc VietTravelVQA-v2; câu lệnh tiếng Anh yêu cầu thêm chó trên đá ở tiền cảnh cạnh hồ. Ảnh đích là kết quả Qwen. Hai đồ thị nguồn/đích là dự đoán REACT++. Hai nhãn tay về chỉnh sửa và bảo toàn đều đạt. Bộ chấm ghi E = 1 và G ≈ 0,829. Giá trị G trống ở các mẫu thiếu bằng chứng cần được giữ là null khi xuất bảng.

## Phạm vi đề xuất cho Hugging Face

VN100 là tập chỉnh sửa dẫn xuất. Hai bộ VQA cung cấp ảnh nguồn; phần nhóm xây dựng gồm việc chọn ảnh, soạn yêu cầu, chạy chỉnh sửa và trích đồ thị, thu nhãn đánh giá và tính điểm. Ngôn ngữ câu lệnh là tiếng Anh; bối cảnh ảnh là Việt Nam.

Bản manifest khóa ghi `local_evaluation_only: true` và `per_image_rights_verified: false` cho cả 200 yêu cầu trên 100 ảnh. Đây là trạng thái xác minh của nhóm. Khi chuẩn bị bản công khai, cần lưu nguồn ghi công, điều kiện sử dụng và phạm vi tái phân phối của từng ảnh được chọn, kể cả việc phát hành ảnh sau chỉnh sửa. Với VietTravelVQA, liên kết tệp tại revision đã khóa giúp tìm lại ảnh trong bộ VQA; thông tin tác giả và giấy phép ảnh Commons gốc là các trường riêng. [Manifest 200 yêu cầu](../configs/level3/qwen200-cohort-v1.json) · [Nguồn gốc VietTravelVQA](https://data.mendeley.com/datasets/fvxtpnh8mh/2)

Đề xuất bản đầu trên Hub gồm data card và một bảng metadata 200 dòng do nhóm xuất từ manifest và kết quả đã khóa. Mỗi dòng nên có `task_id`, `image_id`, nguồn và URL ảnh tại revision, hash ảnh nguồn, câu lệnh, loại thao tác, hai nhãn tay, điểm E/P/G cùng lý do G trống. Giữ provenance về nhãn trống đã chuẩn hóa. Loại đường dẫn máy cá nhân khi xuất. Bảng dùng một tập đánh giá phát triển; tránh tự tạo split train/test từ 200 dòng.

Khi có bằng chứng ghi công và quyền phân phối tương ứng, có thể bổ sung ảnh và các sản phẩm dẫn xuất vào phiên bản sau, kèm danh sách tệp, checksum và điều kiện sử dụng theo thành phần. Trên Hub, data card đặt ở `README.md` tại gốc dataset repo. [Hướng dẫn dataset card của Hugging Face](https://huggingface.co/docs/hub/datasets-cards)

## Nội dung gói và truy xuất

`portable-manifest.json` cung cấp 100 nguồn, 200 case, 300 prediction và đường dẫn tương đối. `BACKUP_INDEX.json` ghi SHA256/kích thước từng thành viên archive và commit code. Gói có ảnh nguyên bản, ảnh editor dùng, ảnh đích, graph/native output, instruction, nhãn raw/final, điểm/receipts, HTML phân tích, code đã commit, notebook và tài liệu. Tên/thư mục lịch sử được giữ để bảo toàn hash; đó không phải các cohort riêng.

Checkpoint mô hình thị giác, virtualenv và log W&B không nằm trong gói; version, URL và hash tải lại nằm trong các asset lock đã lưu. Phần replay Task 2 đi kèm một nhóm ví dụ EMNLP riêng, không được tính vào 100 ảnh/200 yêu cầu của bộ này.

Hướng dẫn sao lưu/khôi phục: [LEVEL3_SUBMISSION.md](LEVEL3_SUBMISSION.md). Phương pháp và kết quả: [LEVEL3_TASK3_REPORT.md](LEVEL3_TASK3_REPORT.md).
