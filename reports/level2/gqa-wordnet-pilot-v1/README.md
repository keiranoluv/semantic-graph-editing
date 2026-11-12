# GQA GAT+WordNet — pilot kiểm pipeline

**PASS** trên 512 ảnh train / 128 validation cố định, seed 42, 5 epoch,
train/validation batch 512. Mỗi epoch chỉ có **một bước optimizer**, tổng cộng
5 bước cho mỗi model; các điểm dưới đây chỉ mô tả trạng thái pilot.

| Model | Epoch chọn | GC R@50 (%) | GC R@100 (%) | GC mR@50 (%) | GC mR@100 (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| GAT | 3 | 11,590 | 20,075 | 0,0925 | 0,1586 |
| GAT, nhánh KG tắt | 1 | 0,911 | 1,799 | 0,3320 | 0,3391 |
| GAT + WordNet | 1 | 0,847 | 1,775 | 0,3316 | 0,3391 |

Chọn best bằng validation GC mR@50 trên đủ **310 predicate**, tie giữ epoch
sớm hơn. Vocabulary/split dự án và evaluator giống các lượt full GQA đã có,
nhưng chỉ dùng subset pilot lấy bằng SHA256 như pilot cũ. Nhiều lớp vắng
trong 128 ảnh vẫn được tính 0. Không dùng bảng này để kết luận KG giúp, hoặc
so sánh chất lượng với các lượt full train đã hội tụ.

GAT thường có 540.087 tham số; hai nhánh KG có 556.727. KG là vector TransE
64D đóng băng + valid bit, ghép trước input projection; đầu vào thị giác/hình
học 517D và embedding nhãn 32D giữ nguyên. Đối chứng KG-off giữ cùng kiến
trúc với KG-on nhưng che vector/valid bit. [Mapping và nguồn KG](../gqa-wordnet-v1/README.md).

Độ phủ theo object trong subset: **5.483/8.350 = 65,66% train** và
**1.359/2.124 = 63,98% validation**. Đây là độ phủ subset, không phải đo trên
toàn bộ 69.942/5.000 ảnh. Độ phủ theo lớp của mapping là 40,79%.

Kiểm chứng:

- Chạy tuần tự trong tmux; không train đồng thời hai model. Peak CUDA allocated
  cao nhất **4,153 GiB** trong các run pilot.
- Trace ảnh/cặp/soft target giống hệt ở cả ba đối chứng, qua cả 5 epoch.
- KG-on/KG-off có initial state giống hệt; buffer KG trong checkpoint khớp
  tensor xuất theo ID lớp GQA.
- Ngắt GAT+KG sau bước đầu rồi resume: history, metrics, model, optimizer,
  CPU/CUDA/sampler RNG và best state khớp tuyệt đối với bản chạy liền.
- Reload best cho điểm chọn đúng; có GC/NG đầy đủ trong [summary.json](summary.json).
- 146 tests liên quan PASS, gồm exact resume CPU/CUDA và kiểm tra sai hash,
  sai ID, che entity chưa train, bỏ ảnh hưởng KG ở đối chứng tắt.
- 15 lượt full cũ và notebook cá nhân giữ nguyên;
  [audit độc lập](../gqa-wordnet-v1/verification.json). Không chấm test GQA.

Code phần mở rộng đặt trong `tools/` vì launcher cũ khóa toàn bộ `src/*.py`
bằng glob. Không thêm/sửa file trong `src/`; context runtime khôi phục trainer
cũ sau khi chạy. [Experiment đã khóa](experiment.json) chứa hash nguồn,
feature bundle và cohort. Checkpoint ở `runs/level2/gqa-wordnet-pilot-v1/`.

[Cách chạy lại và các việc còn lại](../../../docs/LEVEL2_GQA_KG.md).
