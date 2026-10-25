# Dữ liệu thứ hai cho cấp độ 2: GQA và PSG

Ghi nhận ngày 2026-10-25. Đây là bước khảo sát và tải trước dữ liệu, chưa có
adapter, cache CLIP, run huấn luyện hay kết quả đánh giá trên GQA/PSG. Đề bài
`Project01_Graph_Editing.pdf` (trang 3–5) yêu cầu cấp độ 2 thực nghiệm trên ít
nhất hai tập có sẵn, ví dụ VG150 + GQA hoặc VG150 + PSG.

## Chọn tập nào

| | GQA | PSG |
|---|---|---|
| Grounding | Object box, nhãn, thuộc tính, quan hệ có hướng | Segment panoptic, box, nhãn thing/stuff, quan hệ có hướng |
| Đầu ra gốc | Nhãn quan hệ dạng chuỗi; 310 giá trị khác nhau trong hai tệp scene graph đã tải | 56 predicate và 133 object class ở bản đầy đủ |
| Thuận lợi | Giống input box của VG150; có thể dùng crop CLIP theo box | Taxonomy predicate cố định; mirror đã nhúng RGB và mask, không cần tải riêng COCO |
| Việc cần xử lý | Chuẩn hóa vocabulary, đặc biệt quan hệ không gian; chọn split kiểm thử | Loader segment/mask, ID và split; xác minh mirror với bản chính thức |

GQA phù hợp hơn nếu ưu tiên tái dùng cấu trúc box hiện tại. PSG đáng cân nhắc
nếu ưu tiên taxonomy 56 predicate được định nghĩa rõ, nhưng khác VG150 vì quan
hệ nối hai segment. Hai tập đều cần adapter, split cố định và CLIP cache riêng;
không thể đưa trực tiếp vào launcher 15 run VG150 hiện tại. Các bảng nhúng nhãn
và WordNet C.2 `[151,64]` hiện tại gắn với **class ID VG150**, không được dùng
trực tiếp cho class ID GQA/PSG; bản GAT+KG trên tập mới cần mapping và export
riêng hoặc phải báo cáo rõ chính sách không dùng KG.

### Chồng lặp ảnh và protocol

GQA dùng ảnh Visual Genome. Trong hai tệp scene graph đã tải, 71.380/85.638
image ID có mặt trong `data/raw/vg150/images_manifest.json` của máy này
(train GQA 62.506/74.942, val GQA 8.874/10.696). PSG cũng có ảnh từ COCO và
Visual Genome theo [OpenPSG](https://github.com/Jingkang50/OpenPSG). Nếu phối
hai dataset trong cùng một mô hình hoặc tuyên bố kiểm tra chuyển miền, phải
so trùng **ảnh gốc** giữa các split để tránh một ảnh nằm ở train của tập này
và test của tập kia. Kết quả R@K/mR@K khác taxonomy hoặc split không so trực
tiếp như cùng một benchmark.

Nếu chọn một trong hai tập để train, nên giữ official train làm nguồn tạo
train/validation nội bộ theo image ID và giữ split còn lại để kiểm tra cuối.
GQA chỉ có train/val scene graph public trong archive này; PSG mirror đặt tên
`validation` cho official test. Chốt file split/seed trước khi chọn mô hình và
giữ cùng split cho MLP, GNN và phương pháp mới.

## GQA

- [Trang tải chính thức](https://cs.stanford.edu/people/dorarad/gqa/download.html),
  scene graph v1.1. Gói scene graph chỉ cần cho dự đoán quan hệ; chưa cần tải
  gói câu hỏi 1,4 GB.
- Annotation chính thức:
  `https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip`
- Ảnh chính thức:
  `https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip`
  (HEAD trả về 21.817.965.542 bytes).
- URL `nlp.stanford.edu` không kết nối được từ máy này. Tệp annotation được tải
  từ mirror
  `https://huggingface.co/datasets/zrchen03/llava_eval/resolve/72af8107904517f819df370a46a9c73bdac8a2ba/gqa/sceneGraphs.zip`;
  SHA-256 `59f6a3f6ac5227bac6cc615508e542dd546e309c7dbbdb666de05e42d7c51989`,
  44.824.862 bytes. Sau đó tải riêng cùng URL `downloads.cs.stanford.edu`
  chính thức và đối chiếu: hai file có **cùng SHA-256**. `unzip -tq` PASS.
- Đã lưu `data/raw/gqa/archives/sceneGraphs.zip` và giải nén vào
  `data/raw/gqa/sceneGraphs/`: 74.942 train, 10.696 val image graphs;
  1.405.465 object và 4.330.796 relation records. Chỉ riêng `to the left of`
  và `to the right of` đã có 3.859.182 records. Có 1.703 object strings và
  310 predicate strings; các số này là thống kê dữ liệu gốc, chưa phải
  vocabulary huấn luyện.
- Audit CPU: train/val không trùng image ID, mọi relation target đều tồn tại
  trong object list của ảnh. Có 75 box train và 68 box val có chiều rộng hoặc
  chiều cao không dương; adapter phải định nghĩa chính sách loại/sửa và ghi số
  mẫu bị ảnh hưởng trước khi tạo CLIP crop.
- Đã tải `data/raw/gqa/archives/images.zip` từ URL chính thức ở trên và,
  do tốc độ chậm, hoàn tất các phần còn lại bằng URL mirror ghim revision
  `https://huggingface.co/datasets/Feeky929/GQA-images/resolve/a30da776e9311362fe2691163dd74608578d58a1/images.zip`.
  File chính thức và mirror cùng 21.817.965.542 bytes; đã đối chiếu hai vùng
  1 MiB ở đầu và giữa phần tải được, giống hệt. SHA-256 LFS kỳ vọng của
  mirror là `02ce5c49c793accd5305356de9c39a50f80a7aaac193b0203de30dbbc65bde62`;
  **SHA-256 toàn file đã khớp** và `unzip -tq` PASS. ZIP chứa 148.854 JPEG. Mọi image ID trong
  74.942 train và 10.696 val scene graphs đều có JPEG trong archive. Chưa
  giải nén ảnh để tránh tạo bản sao khi các run VG150 đang chạy.

Lệnh tải lại annotation trên máy có kết nối Stanford:

```bash
mkdir -p data/raw/gqa/archives data/raw/gqa/sceneGraphs
curl -fL -C - --retry 5 \
  'https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip' \
  -o data/raw/gqa/archives/sceneGraphs.zip
sha256sum data/raw/gqa/archives/sceneGraphs.zip
unzip -tq data/raw/gqa/archives/sceneGraphs.zip
unzip -nq data/raw/gqa/archives/sceneGraphs.zip -d data/raw/gqa/sceneGraphs
```

Nếu host Stanford không kết nối được, tải annotation từ mirror đã kiểm với
bản chính thức:

```bash
curl -fL -C - --retry 5 \
  'https://huggingface.co/datasets/zrchen03/llava_eval/resolve/72af8107904517f819df370a46a9c73bdac8a2ba/gqa/sceneGraphs.zip' \
  -o data/raw/gqa/archives/sceneGraphs.zip
echo '59f6a3f6ac5227bac6cc615508e542dd546e309c7dbbdb666de05e42d7c51989  data/raw/gqa/archives/sceneGraphs.zip' | sha256sum -c -
```

Tải ảnh riêng:

```bash
curl -fL -C - --retry 5 \
  'https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip' \
  -o data/raw/gqa/archives/images.zip
sha256sum data/raw/gqa/archives/images.zip
unzip -tq data/raw/gqa/archives/images.zip
```

Nếu dùng mirror ảnh thay Stanford:

```bash
curl -fL -C - --retry 5 \
  'https://huggingface.co/datasets/Feeky929/GQA-images/resolve/a30da776e9311362fe2691163dd74608578d58a1/images.zip' \
  -o data/raw/gqa/archives/images.zip
echo '02ce5c49c793accd5305356de9c39a50f80a7aaac193b0203de30dbbc65bde62  data/raw/gqa/archives/images.zip' | sha256sum -c -
unzip -tq data/raw/gqa/archives/images.zip
```

## PSG

- [Trang dataset chính thức](https://psgdataset.org/data_stats.html) và
  [OpenPSG](https://github.com/Jingkang50/OpenPSG). Bản đầy đủ có 48.749 ảnh,
  133 object class, 56 predicate class. Kho dữ liệu chính thức có
  [link thư mục chia sẻ](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/EgQzvsYo3t9BpxgMZ6VHaEMBDAb7v0UgI8iIAExQUJq62Q?e=fIY3zh)
  và [link OneDrive bạn mở được bằng trình duyệt](https://entuedu-my.sharepoint.com/personal/jingkang001_e_ntu_edu_sg/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fjingkang001%5Fe%5Fntu%5Fedu%5Fsg%2FDocuments%2Fopenpsg%2Fdata&ga=1).
  `curl` từ container này nhận HTTP 403 (`403 FORBIDDEN`) với **cả hai** URL,
  nên chưa tải bản chính thức bằng CLI để so sánh với mirror. HTTP 403 ở đây
  không có nghĩa người dùng không mở được bằng trình duyệt.
- Mirror đã chọn để tải trước:
  [wliafe/PSG](https://huggingface.co/datasets/wliafe/PSG), revision
  `c33d624babf8fbc5b4da75ab82ba4cd3f9b387d5`. Repo khai báo 48.749
  records: train 46.563, validation 2.186; `validation` là official test
  được đổi tên trong mirror. Các Parquet chứa sẵn RGB, panoptic mask, box,
  label và relations; tổng 8,56 GB. Dữ liệu mirror chưa được xác thực từng
  mẫu với SharePoint, nên chưa dùng làm kết quả chính thức trước audit.
- Đã tải đủ 17 train shard và 1 validation shard vào
  `data/raw/psg-hf-wliafe-c33d624/data/`. Từng file khớp byte count và
  SHA-256 LFS từ API Hugging Face trước khi đổi `.part` sang `.parquet`.
- Audit CPU toàn bộ 48.749 records: train 46.563, validation 2.186, PSG sample ID
  không trùng giữa hai split; mọi record có mask, số box/label/segment bằng
  nhau, box nằm trong ảnh và có diện tích dương, subject/object index hợp lệ.
  Thấy đủ label ID 1–133 và predicate ID 1–56. Đây là kiểm tra cấu trúc
  mirror, **chưa chứng minh** annotation giống byte-for-byte với SharePoint.
- `id` là PSG sample ID. Có 225 train records và 16 validation records dùng
  lại `image.path` trong cùng split; không có `image.path` trùng giữa hai split.
  Khi tạo validation nội bộ từ train, phải chia theo ảnh gốc, không chỉ theo
  sample ID.
- [Điều khoản PSG](https://psgdataset.org/term.html): annotation CC BY 4.0;
  ảnh gốc có quyền sử dụng riêng. Giữ provenance và không phân phối lại ảnh
  trong Git.

URL mẫu của một shard, các shard khác thay `00000` thành `00001`–`00016`;
validation là `validation-00000-of-00001.parquet`:

```text
https://huggingface.co/datasets/wliafe/PSG/resolve/c33d624babf8fbc5b4da75ab82ba4cd3f9b387d5/data/train-00000-of-00017.parquet
https://huggingface.co/datasets/wliafe/PSG/resolve/c33d624babf8fbc5b4da75ab82ba4cd3f9b387d5/data/validation-00000-of-00001.parquet
```

Có thể tải lại toàn bộ revision bằng `huggingface_hub`:

```bash
python -m pip install huggingface_hub
```

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="wliafe/PSG",
    repo_type="dataset",
    revision="c33d624babf8fbc5b4da75ab82ba4cd3f9b387d5",
    allow_patterns=["README.md", "data/*.parquet"],
    local_dir="data/raw/psg-hf-wliafe-c33d624",
)
```

Sau tải lại, đối chiếu byte count/SHA-256 của từng shard với LFS metadata ở
revision trên. Bản [PSG-coco-format](https://huggingface.co/datasets/maelic/PSG-coco-format)
chỉ có box, không có mask gốc, nên không được coi là bản PSG panoptic đầy đủ.

Không commit archive, JPEG, Parquet, mask hoặc cache đặc trưng vào Git.
