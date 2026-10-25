# Thiết lập và tái lập mức 1 cho thành viên nhóm

Mọi lệnh dưới đây chạy từ thư mục gốc repository sau khi clone code.
Kiểm chứng tại máy project ngày 2026-10-24: Linux x86_64, Python 3.12.3.
Môi trường kiểm tra CPU được cài mới trong `myenv`, không dùng package Python global.

## 1. Tạo môi trường

Trên Linux cần có Python 3.12, module venv, `curl`, `unzip` và `sha256sum`.
Ví dụ trên Ubuntu nếu chưa có: `sudo apt install python3.12-venv curl unzip`.

```bash
python3.12 -m venv myenv
source myenv/bin/activate
python -m pip install --upgrade pip
```

Chọn **một** bản PyTorch trước khi cài dependencies còn lại.

CPU để đọc dữ liệu, chạy tests và smoke nhỏ:

```bash
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
python -m pip check
```

GPU NVIDIA hỗ trợ CUDA 12.8, dành cho người dùng chạy train:

```bash
nvidia-smi
python -m pip install --upgrade --force-reinstall torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-dev.txt
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Nếu driver/máy không phù hợp CUDA 12.8, chọn build được hỗ trợ theo
[PyTorch](https://pytorch.org/get-started/locally/), giữ phiên bản torch 2.7.0
để giảm khác biệt với môi trường dự án. Không cần cài torchvision/torchaudio
cho pipeline geometry + nhãn + WordNet hiện tại.
Theo [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html),
các thư viện mở rộng sparse là tùy chọn cho cách dùng cơ bản; chưa cần build
torch-scatter/torch-sparse để chạy GAT hiện tại.

`requirements.txt` cố định dependencies trực tiếp; `requirements-dev.txt` thêm
tests và công cụ kiểm tra. Đây chưa phải lockfile mọi dependency gián tiếp.
Trước thực nghiệm chính lưu `python -m pip freeze` trong thư mục run để ghi
lại chính xác môi trường máy chạy; không giả định tương đương bitwise CPU/GPU.

## 2. Download annotations VG150

```bash
bash scripts/download_vg150.sh
```

Script tải HDF5, vocabulary và metadata ảnh; HDF5 được kiểm tra SHA-256.
File tải sẵn được giữ lại. Nếu phát hiện checksum sai, dừng và kiểm tra nguồn;
không tự xóa dữ liệu hoặc tiếp tục dùng file sai.

Nguồn của script:

- HDF5: `https://huggingface.co/datasets/kb-kim/LLM4SGG/resolve/main/VG-SGG-with-attri.h5`
- Vocabulary: `https://raw.githubusercontent.com/Prk10/VG150_VisualGenome/main/VG-SGG-dicts-with-attri.json`
- Metadata: `https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip`

Các URL có nhánh `main` có thể đổi. Đối chiếu fingerprint của nhóm ở dưới,
không chỉ dựa vào tên file. Đây là fingerprint đo trên máy project, không phải
chứng thực độc lập từ đơn vị phát hành.

## 3. Tạo split dùng chung

Chỉ chạy khi chưa có `image_data.json` và `splits.json` đã tạo:

```bash
python -m scripts.build_vg150_manifest \
  --h5 data/raw/vg150/VG-SGG-with-attri.h5 \
  --metadata data/raw/vg150/image_data_raw.json \
  --output-dir data/raw/vg150
```

Script không shuffle; loại bốn metadata ảnh lỗi, lọc ảnh không có box/quan hệ,
lấy 5.000 ảnh đầu của pool train làm validation và giữ split test HDF5.
Không tự resplit ngẫu nhiên. Script từ chối ghi đè output sẵn có.
Tất cả mô hình phải trỏ cùng thư mục và dùng `box_policy=clip` như baseline.

Kết quả nhóm: train **57.723**, val **5.000**, test **26.446**.
Đã xác minh tái tạo khớp split hiện có; chưa xác minh đầy đủ tương đương mọi
quy ước của benchmark cộng đồng, vì vậy không tự so sánh trực tiếp với bảng
điểm bài báo.

Fingerprint tham chiếu (SHA-256):

| File trong `data/raw/vg150/` | SHA-256 |
|---|---|
| `VG-SGG-with-attri.h5` | `ad52a01f8e6142bb7bc62d91f97dad1567f44772a23eac68d60a5b7766ffe4b1` |
| `VG-SGG-dicts-with-attri.json` | `1255873ae0250555f59d20f8d1e5a4bdf22b331119502c850f765173b6963ca1` |
| `image_data_raw.json` | `93d0053d7e8f451497646fa513e8fb9b1a889df6e757010b88e51e2e21803487` |
| `image_data.json` | `aa6ffcc11edf68d9d5376bc7dac04caa4eafab589d4db0290b6990a616ff00db` |
| `splits.json` | `d9402d67ad0dc22fa67adc558d39502bfb479f9998bc689ffb3d74058bac4bf0` |

Kiểm tra bằng:

```bash
sha256sum data/raw/vg150/VG-SGG-with-attri.h5 \
  data/raw/vg150/VG-SGG-dicts-with-attri.json \
  data/raw/vg150/image_data_raw.json \
  data/raw/vg150/image_data.json data/raw/vg150/splits.json
```

Các run annotation-only trước đây không đọc pixel ảnh. Ma trận CLIP mới cần
đủ JPEG theo hướng dẫn ngay dưới. Để tải riêng một JPEG cho notebook sau khi
biết `image_id`:

```bash
python -m scripts.download_vg150_image --image-id IMAGE_ID
```

Script lấy URL từ `image_data.json` và lưu vào `data/raw/vg150/images/`.
Ảnh đơn lẻ này chỉ phục vụ demo, không đủ cho ma trận CLIP.
Trong môi trường kiểm tra ngày 2026-10-24, host JPEG trả lỗi kết nối; notebook
vẫn chạy với hộp và đồ thị trên nền trống. Nếu máy teammate cũng không truy cập
được, dùng [trang tải Visual Genome chính thức](https://visualgenome.org/api/v0/api_home)
để tải archive ảnh part 1/2, rồi đặt JPEG cần dùng vào thư mục `images/`.

### JPEG đầy đủ cho ma trận CLIP PredCls

Sau khi có `image_data.json` và `splits.json`, teammate chạy từ gốc repo:

```bash
myenv/bin/python -m scripts.download_vg150_images --dry-run
myenv/bin/python -m scripts.download_vg150_images
```

Lệnh tải đúng 89.169 ảnh train/val/test từ hai archive qua mirror đã dùng cho
review, tiếp tục được nếu tải dở. Kết quả gồm `data/raw/vg150/images/*.jpg`,
hai ZIP trong `data/raw/vg150/archives/` và `images_manifest.json` chứa hash
từng ảnh, kích thước JPEG thực và hash archive. Không commit các file dữ liệu.
URL tải JPEG được ghi trực tiếp trong `scripts/download_vg150_images.py`:

- `https://huggingface.co/datasets/pppop7/VisualGenome_VG_100K_1_and_2/resolve/main/images.zip`
- `https://huggingface.co/datasets/pppop7/VisualGenome_VG_100K_1_and_2/resolve/main/images2.zip`

Đây là mirror của hai archive ảnh [Visual Genome](https://visualgenome.org/api/v0/api_home).
Annotation HDF5, từ điển nhãn và metadata dùng các URL riêng đã khai báo ngay
đầu `scripts/download_vg150.sh`.
Một số JPEG có kích thước thực khác metadata; script ghi cả hai kích thước để
trích vùng ảnh quy đổi tọa độ GT box theo từng trục. Cache CLIP và lệnh train
đầy đủ ở [level1_clip_regions.md](experiments/level1_clip_regions.md).
Hai archive nhiều GB là bắt buộc nếu teammate tái lập ma trận CLIP đầy đủ;
giữ nguyên `image_id` và kiểm tra hình khớp metadata.

Để review WordNet với 5 JPEG train cho mỗi class mà không tải toàn bộ
archive, dùng lệnh sau sau khi đã tạo `train_examples_with_boxes.csv`:

```bash
python -m scripts.download_level1_review_images
```

Lệnh đọc từng JPEG cần thiết từ bản lưu trữ ZIP thay thế bằng HTTP Range, lưu
237 ảnh gốc khác nhau vào `data/raw/vg150/images/` và tạo bảng đối chiếu 750
cặp class–ảnh trong
`data/knowledge/level1-human-review-v2/review_images.csv`. Chi tiết tại
`docs/knowledge/LEVEL1_MAPPING_REVIEW.md`.
Để tạo thư mục ảnh có tên class dễ xem bằng file manager, chạy tiếp:

```bash
python -m scripts.prepare_level1_review_images
```

Kết quả nằm ở `data/raw/vg150/images_review/`; ví dụ
`001_airplane_2349975.jpg`, trong đó số cuối là `image_id` gốc. Mở
`data/raw/vg150/images_review/index.html` để xem khung bao màu đỏ trên 5 ảnh
của mỗi class.

## 4. WordNet và mapping

```bash
python -m nltk.downloader wordnet
python -c "from nltk.corpus import wordnet as wn; print(wn.get_version())"
python -m scripts.prepare_knowledge --output-dir data/knowledge/wordnet-team-v1
python -m scripts.prepare_wordnet_review rank \
  --mapping data/knowledge/wordnet-team-v1/wordnet_mapping.json \
  --output-dir data/knowledge/wordnet-team-review-v1
```

Mục đích: tải corpus ngoài, kiểm tra phiên bản, liên kết vocabulary và tạo
CSV ứng viên để **người** kiểm tra. WordNet của artifact cũ là 3.0. Môi trường
mới có thể ghi NLTK khác artifact cũ; output khác phải dùng thư mục mới.
Không ghi đè bản cũ hoặc coi ranking là quyết định kiểm duyệt.

Bản mapping đại diện 150/150 do Toan chọn đã được lưu thành override có
provenance trong `configs/knowledge/wordnet_overrides.reviewed-toan-v1.json`.
Sau khi cài WordNet 3.0, teammate có thể tạo lại mapping:

```bash
python -m scripts.prepare_knowledge \
  --overrides configs/knowledge/wordnet_overrides.reviewed-toan-v1.json \
  --output-dir data/knowledge/wordnet-reviewed-toan-v1
```

Lệnh này chỉ chuẩn bị mapping trên CPU; KG và checkpoint TransE draft cũ chưa
tương thích với mapping reviewed mới.
Toan đã xác nhận 150 lựa chọn này là chính thức. Teammate tiếp tục dựng KG và
tạo manifest reviewed batch bằng lệnh trong
[c2_gat_wordnet.md](knowledge/c2_gat_wordnet.md). Manifest chứa đường dẫn tuyệt
đối và SHA-256, nên phải tạo lại trên từng máy sau khi có KG; không copy
manifest từ máy khác rồi đổi đường dẫn bằng tay.
Chạy trong `myenv` (NLTK 3.9.1). Python hệ thống trên máy kiểm tra dùng NLTK
3.10.3 và sinh metadata `nltk_version` khác, dù 151 dòng mapping giống hệt;
không ghi đè hai artifact cùng thư mục. Nếu đổi môi trường, chọn output mới.

Các artifact đang dùng cho smoke project nằm ở
`data/knowledge/wordnet-ai-draft-v1/`, không được commit. Máy teammate cần nhận
bản artifact chung cùng fingerprint hoặc tái tạo bằng các workflow trong
`docs/knowledge/` (xem [a2_ai_draft_workflow.md](knowledge/a2_ai_draft_workflow.md));
tạo mapping mới không bảo đảm trùng 124 lựa chọn AI cũ. Để chạy cùng thực nghiệm
KG, nhóm phải chia sẻ cả mapping, KG, checkpoint TransE hoặc C.2 export
với hash đã ghi trong manifest. Chỉ download VG150 + WordNet corpus sẽ chưa tạo
ra đúng 124 lựa chọn draft này.
Hướng dẫn review: [HUMAN_REVIEW_REQUIRED.md](knowledge/HUMAN_REVIEW_REQUIRED.md).

ConceptNet là nguồn bổ sung, chưa cần download để chạy WordNet đầu tiên.
Nếu dùng, làm theo [b2_conceptnet_workflow.md](knowledge/b2_conceptnet_workflow.md):
download snapshot 5.7.0, kiểm tra hash chung, giữ attribution và tách không gian KG.

## 5. Kiểm tra trước khi train

```bash
python -m pytest tests/ -q
python -m scripts.smoke_train_mlp --steps 3 --box-policy clip
python -m ruff check src scripts tests
python -m black --check src scripts tests
python -m isort --check-only src scripts tests
git diff --check
```

Tests phần lớn dùng dữ liệu tổng hợp. Smoke MLP dùng annotations VG150 thật,
chạy CPU ba optimizer steps, không ghi checkpoint baseline và không đo benchmark.
Đánh giá test chỉ thực hiện sau khi chốt thiết kế và lựa chọn mô hình bằng validation.

Nếu thành viên có đúng 9 checkpoint baseline của nhóm, tái đánh giá full
validation và tạo bảng 3 seeds:

```bash
mkdir -p reports/level1
for model in mlp mlp_semantic gat; do
  for seed in 42 123 3407; do
    python -m scripts.evaluate_level1 \
      --checkpoint "checkpoints/experiments/${model}_bs512_seed${seed}/run_0001/best.pt" \
      --device cpu --batch-size 64 \
      --output "reports/level1/${model}_seed${seed}_val5000.json"
  done
done
python -m scripts.summarize_level1 reports/level1/*_seed*_val5000.json \
  --output reports/level1/baselines_3seed_val5000_summary.json
```

Mục đích: đọc checkpoint từ cùng split hiện tại, đánh giá toàn bộ 5.000 ảnh
validation, rồi tính mean và sample std. Script từ chối ghi đè output.
Checkpoint chưa được phân phối qua Git; nhóm cần chia sẻ run files hoặc tự
train lại theo `docs/experiments/step1_mlp_baselines.md` và
`docs/experiments/step2_gat_baseline.md`, rồi dùng đường dẫn checkpoint mới.

Để chạy notebook demo:

```bash
python -m pip install -r requirements-demo.txt
python -m jupyter lab
```

Mở `notebooks/level1_scene_graph_demo.ipynb` trong JupyterLab. Lệnh chạy từ
thư mục gốc repository để notebook tìm thấy `README.md` và dữ liệu. Hướng dẫn
cài JupyterLab này theo [Project Jupyter](https://jupyter.org/install).

## 6. Giới hạn đã biết

- Config baseline ghi số ảnh nhưng không ghi hash split tại thời điểm train:
  kiểm tra split hiện tại chưa chứng minh split lịch sử của từng run.
- Chưa tái chạy full validation từ các checkpoint trong đợt kiểm tra này.
- Lịch sử các runs có độ dài khác nhau; không suy ra ngân sách train bằng nhau.
- Lệnh train GPU và kết quả thực nghiệm tiếp theo được ghi riêng trong
  [LEVEL1_PROGRESS.md](LEVEL1_PROGRESS.md).
