# Chỉnh sửa ảnh theo yêu cầu bằng đồ thị cảnh

Đồ án nghiên cứu ba câu hỏi: hai đối tượng trong ảnh có quan hệ gì, câu lệnh yêu cầu thay đổi đồ thị thế nào, và ảnh sau chỉnh sửa có đáp ứng yêu cầu trong khi bảo toàn phần còn lại của cảnh hay không.

| Bạn muốn | Mở |
|---|---|
| Xem toàn bộ đồ án ngay sau khi clone | [Notebook tổng hợp đã lưu output](notebooks/project_submission_demo.ipynb). Mở trong JupyterLab hoặc VS Code để xem ảnh, đồ thị và bảng điểm. |
| Chạy demo trực tiếp trên CPU | [Notebook chỉnh sửa đồ thị](notebooks/graph_editing_live_demo.ipynb). Ba ca lần lượt là thêm thuyền buồm, thêm chó trên đá và xóa ghế. [Cài môi trường và chạy](docs/REPRODUCE.md#chay-demo). |
| Xem 100 ảnh Việt Nam và 200 ảnh Qwen chỉnh sửa | [VN100 · image-pairs/dev](https://huggingface.co/datasets/ndttoan/vn100-graph-editing/viewer/image-pairs/dev). Mỗi hàng gồm ảnh nguồn, câu lệnh và ảnh đích. [Nhãn và điểm E/P/G · default/dev](https://huggingface.co/datasets/ndttoan/vn100-graph-editing/viewer/default/dev) · [Data card](docs/LEVEL3_DATA_CARD.md). |
| Tái lập một kết quả, kiểm tra checkpoint hoặc split | [Hướng dẫn tái lập](docs/REPRODUCE.md) · [Danh mục đầu vào ngoài Git của các demo CPU](submission/demo-assets.json). |

## Bản đồ mã nguồn theo nhiệm vụ

| Nhiệm vụ | Dữ liệu và phép kiểm | Mã nguồn | Kết quả |
|---|---|---|---|
| **1. Dự đoán quan hệ** | VG150 PredCls, so MLP, GAT và các biến thể có tri thức nền. Demo tính lại 5 Hit và 5 Miss ở GC top-50. | [Mô hình](src/models) · [Evaluator PredCls](src/evaluation/reference_predcls.py) | [So sánh GC/NG](reports/level1/reference-comparison/comparison.md) |
| **2A. Dự đoán quan hệ trên tập thứ hai** | GQA, so PE-PLPR với PE-PLPR + AlignG trên cùng cặp đối tượng. Demo tái hiện 5 Hit và 5 Miss. | [AlignG](experiments/aligng_fast_v1) · [Chấm test](tools/gqa_test_v1) | [Kết quả GQA](reports/level2/gqa-test-v1/README.md) |
| **2B. Dự đoán thao tác theo câu lệnh** | MagicBrush, từ đồ thị nguồn và câu lệnh dự đoán thao tác trên nút/cạnh. Demo xóa ghế chạy GNN seed 123 trên CPU. | [Dữ liệu, runner và evaluator](experiments/magicbrush_prompt_intent_v1) | [Đánh giá 59 ca test](reports/level2/magicbrush-prompt-intent-test-v1/README.md) |
| **2B mở rộng. Chỉnh sửa đồ thị truy vấn** | Bộ crowdsourced của He và cộng sự. Notebook phát lại một dự đoán đúng và một dự đoán sai rồi áp dụng thao tác lên đồ thị nguồn. | [Trình chấm checkpoint v4](tools/eval_emnlp2020_task_b_wandb_v4_test_v1.py) | [Kết quả trên 530 mẫu](reports/level2/emnlp2020-task-b-wandb-v4-test-v1/README.md) |
| **3. Kiểm tra ảnh sau chỉnh sửa** | VN100, Qwen tạo ảnh và REACT++ trích đồ thị. Notebook tính lại điểm chỉnh sửa, bảo toàn và nhất quán cho sáu cặp ảnh, đối chiếu với nhãn tay. | [Hai bước tạo ảnh và đồ thị](docs/LEVEL3_PIPELINE.md) · [Bộ chấm](tools/level3_evaluation) | [Kết quả VN100](docs/LEVEL3_TASK3_REPORT.md) |
| **Mở rộng video** | PVSG, dự đoán quan hệ theo thời gian giữa hai track chuẩn. | [Pipeline PVSG](experiments/pvsg_trd_v1) | [Kết quả và phân tích lỗi](reports/pvsg-trd-m5-v1/final.md) |

## Chạy thử một ca

Sau khi cài môi trường CPU và khôi phục bộ tệp `chair` theo [hướng dẫn](docs/REPRODUCE.md#chay-demo), chạy từ thư mục gốc:

```bash
.venv/bin/python -m scripts.demo_magicbrush_chair
```

Với yêu cầu *remove the chairs*, checkpoint dự đoán xóa nút ghế và cạnh nối ghế với quầy. Hai dòng cuối kiểm tra tập thao tác và đồ thị đích đều in `True`.

## Cấu trúc bản nộp

- `src/`, `experiments/`, `tools/`, `scripts/`, `configs/` chứa mô hình, mã xử lý dữ liệu, lệnh chạy và cấu hình thí nghiệm.
- `notebooks/`, `reports/`, `reviews/`, `docs/` chứa demo, kết quả theo từng lượt, minh chứng kiểm tra dữ liệu và hướng dẫn.
- `submission/` chứa môi trường demo, manifest để khôi phục artifact và nguồn của bộ dữ liệu phát hành.
- `archived/` chứa bản đóng gói cũ và ghi chú lịch sử, có thể bỏ khỏi bản nộp. Một số module đang dùng có chữ `pilot` trong tên; hãy giữ chúng tại đường dẫn hiện tại.
