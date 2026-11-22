# Chỉnh sửa ảnh theo yêu cầu bằng đồ thị cảnh

Mã nguồn gồm ba nhiệm vụ và phần mở rộng video. [Notebook tổng hợp](notebooks/project_submission_demo.ipynb) có ảnh và kết quả đã lưu. [Notebook demo trực tiếp](notebooks/graph_editing_live_demo.ipynb) chạy GNN trên ba yêu cầu chỉnh sửa.

## Mã nguồn theo nhiệm vụ

| Nhiệm vụ | Tiền xử lý và split | Mô hình | Huấn luyện | Đánh giá và suy luận |
|---|---|---|---|---|
| [1 · VG150](task1_vg150/README.md) | [Split](task1_vg150/preprocess.py), [cache CLIP](task1_vg150/preprocessing/cache_regions.py) | [GAT](common/models/gat.py), [GAT + KG](common/models/gat_kg.py) | [Launcher theo báo cáo](task1_vg150/train.py), [GAT](task1_vg150/training/gat.py), [MLP](task1_vg150/training/mlp.py) | [Chấm checkpoint](task1_vg150/evaluate.py) |
| [2A · GQA](task2_gqa/README.md) | [Chuẩn hóa và chia tập](task2_gqa/preprocess.py) | [PE-PLPR + AlignG](task2_gqa/models/fast.py) | [Runner](task2_gqa/train.py) | [Đánh giá checkpoint](task2_gqa/evaluate.py) |
| [2B · MagicBrush](task2_graph_editing/magicbrush/README.md) | [Cặp và split](task2_graph_editing/magicbrush/data), [đọc cặp](task2_graph_editing/magicbrush/dataset.py) | [GNN](task2_graph_editing/magicbrush/model.py) | [Runner](task2_graph_editing/magicbrush/train.py) | [Dự đoán](task2_graph_editing/magicbrush/predict.py), [chấm test](task2_graph_editing/magicbrush/evaluate.py) |
| [2B mở rộng · He và cộng sự](task2_graph_editing/he2020/README.md) | [Chuẩn hóa và chia tập](task2_graph_editing/he2020/preprocess.py) | [MLP/GNN](task2_graph_editing/he2020/model.py) | [Train](task2_graph_editing/he2020/train.py) | [Dự đoán](task2_graph_editing/he2020/predict.py), [chấm checkpoint](task2_graph_editing/he2020/evaluate.py) |
| [3 · VN100](task3_vn100/README.md) | [200 yêu cầu](configs/level3/qwen200-cohort-v1.json) | [Qwen](task3_vn100/generation/backend.py), [REACT++](task3_vn100/models/react.py) | Mô hình pretrained | [Tạo ảnh](task3_vn100/generate.py), [trích đồ thị](task3_vn100/extract_graphs.py), [chấm điểm](task3_vn100/evaluation) |
| [Mở rộng · PVSG](extension_pvsg/README.md) | [Split video](extension_pvsg/manifests/pvsg_author_release_338_62.lock.json) | [MLP, GAT](extension_pvsg/baselines/models.py), [mô hình thời gian](extension_pvsg/ablations/models.py) | [Train một model](extension_pvsg/train.py) | [Đánh giá checkpoint](extension_pvsg/evaluate.py) |

Mỗi task có lệnh tiền xử lý, huấn luyện và đánh giá trong README riêng. training.py hoặc training/ chứa vòng tối ưu; model.py hoặc models/ định nghĩa mô hình; dataset.py hoặc data.py đọc dữ liệu. Phần dùng chung giữa các task nằm trong common/, gồm encoder và codec chỉnh sửa đồ thị, đặc trưng, các mô hình và metric.

configs/ lưu cấu hình cố định. assets/ chứa ví dụ và danh mục checkpoint cho notebook, data card VN100, manifest trọng số và xuất xứ mã nguồn.

## Chạy demo trên CPU

Dùng Linux và Python 3.12, chạy từ thư mục chứa README này.

    python3.12 -m venv .venv
    .venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1 torchvision==0.24.1
    .venv/bin/pip install -r environments/demo-cpu.txt
    .venv/bin/python -m common.restore_assets chair --from-dir /duong/dan/artifacts
    .venv/bin/python -m task2_graph_editing.magicbrush.demo

Thư mục artifacts có data/, runs/, hoặc nội dung đã giải nén từ chair-assets.tar. Ca xóa ghế cần 7 tệp, khoảng 357 MiB, gồm CLIP và checkpoint GNN seed 123. Hai dòng kiểm tra thao tác và đồ thị đích phải in True. [Đường dẫn, dung lượng và SHA-256](assets/demo-assets.json)

Hai notebook đã lưu output, nên có thể mở xem mà không khôi phục artifact. Để chạy lại các ô, chọn bộ tương ứng:

| Bộ artifact | Dung lượng | Phạm vi |
|---|---|---|
| chair | 357 MiB | Lệnh demo xóa ghế bên trên. |
| live | 412 MiB | graph_editing_live_demo.ipynb: thuyền buồm, chó và ghế. |
| overview | 22,05 GiB | project_submission_demo.ipynb: các nhiệm vụ VG150, GQA, MagicBrush, He và VN100. |

Bộ overview chứa kho ảnh GQA khoảng 20,32 GiB. Bộ live đủ cho notebook demo trực tiếp. Để chạy lại cả hai notebook, khôi phục cả live và overview; phần chung được dùng lại, tổng khoảng 22,08 GiB.

Ví dụ chạy notebook demo trực tiếp:

    .venv/bin/python -m common.restore_assets live --from-dir /duong/dan/artifacts
    .venv/bin/jupyter lab notebooks

Chọn kernel của môi trường vừa cài. Với notebook tổng hợp, thay live bằng overview. Notebook tổng hợp hiển thị 5 Hit và 5 Miss trên validation của mỗi bộ VG150/GQA, kết quả He, demo GNN MagicBrush và sáu ca VN100.

## Dữ liệu và split

- VG150 dùng split theo ID ảnh. GQA chia train chính thức thành 69.942 ảnh train và 5.000 validation. 10.696 ảnh validation chính thức dùng làm test của đồ án. Từ vựng xây trên train.
- MagicBrush có 240/46/59 cặp train/validation/test, tách theo nhóm COCO và hash pixel hai ảnh. Nhãn có hỗ trợ AI, giữ thao tác theo câu lệnh và thay đổi cấu trúc liên quan. Đồ thị đích chấm điểm được dựng bằng cách áp nhãn lên đồ thị nguồn. [Kết quả kiểm tay 20 mẫu](task2_graph_editing/magicbrush/review20-result.json).
- Bộ He có 530 mẫu test sau lọc. Quy trình kiểm tra nhóm nguồn dành cho train/validation. Tính độc lập theo ảnh gốc chưa được xác minh. Notebook phát lại dự đoán đã lưu.
- VN100 có 100 ảnh nguồn và 200 yêu cầu. Hai yêu cầu cùng ảnh có chung image_id. [Data card và nguồn ảnh](assets/datasets/vn100/README.md) · [Ảnh và dataset](https://huggingface.co/datasets/ndttoan/vn100-graph-editing).
- PVSG chia 338 video train thành 287 fit và 51 validation, giữ các cửa sổ cùng video trong cùng tập. Test gồm 62 video theo split phát hành. Bốn nhóm ID nguồn Ego4D xuất hiện hai phía của split phát hành.

## Môi trường và xuất xứ

[environments/](environments) có môi trường riêng cho demo CPU, huấn luyện đồ thị, Qwen, REACT++, chấm ảnh và PVSG. Từng nhiệm vụ có hướng dẫn ngắn về dữ liệu và lệnh chạy.

[Danh mục xuất xứ](assets/source_origins.json) liên kết mã đã tổ chức lại với mã gốc. [Manifest bản nộp](assets/refactor_manifest.json) kiểm tra mã hiện tại. Cấu hình, nhãn và receipt lịch sử giữ thông tin của lượt đã tạo checkpoint. Huấn luyện bằng bản refactor ghi nhận hash mới. Các lượt train trong bản clone ghi Git revision. Bản chép độc lập ghi hash manifest mã nguồn.
