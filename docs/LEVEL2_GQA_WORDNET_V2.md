# GQA WordNet v2 — quy trình và bàn giao GPU

Đã hoàn thành mapping v2, audit toàn bộ train, train embedding TransE mới và
pilot có bằng chứng exact resume. Full GAT+KG ba seed đã được khởi chạy trước
quy ước vận hành mới. **Từ các lượt GPU tiếp theo: assistant chuẩn bị/kiểm code,
commit, rồi gửi lệnh; người dùng tự chạy.**

- [Mapping, review và audit train](../reports/level2/gqa-wordnet-v2/README.md).
- [Pilot v2](../reports/level2/gqa-wordnet-pilot-v2/README.md).
- [Full recipe, trạng thái và W&B](../reports/level2/gqa-wordnet-full-v2/README.md).

## Theo dõi job đã khởi chạy

```bash
tmux attach -t gqa-wordnet-v2
```

Hoặc đọc log mà không attach:

```bash
tail -f logs/gqa-wordnet-v2-pipeline.log
cat runs/level2/gqa-wordnet-v2-pipeline/status.json
```

Job này có lịch chạy tuần tự cả ba seed. `Ctrl+B`, rồi `D` detach khỏi tmux.
`Ctrl+C` khi đang tail chỉ dừng xem log. File kết quả chính chỉ được xuất sau
khi run hoàn thành; các metric sớm trong history chưa phải kết quả cuối.

## Người dùng tự chạy tiếp sau khi job hiện tại đã dừng

Từ thư mục repo, tạo phiên tmux rồi chạy lệnh sau. Pipeline khóa thư mục để
tránh hai launcher ghi cùng lúc. Nó xác minh và bỏ qua seed đã hoàn thành,
resume seed dở từ `last.pt`, rồi tiếp tục các seed còn lại.

```bash
tmux new-session -s gqa-wordnet-v2-resume
```

Trong phiên đó:

```bash
cd /workspace/workspace/semantic-graph-editing
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 myenv/bin/python -u \
  -m tools.run_gqa_wordnet_v2_pipeline --stage full \
  > logs/gqa-wordnet-v2-resume.log 2>&1
```

Lệnh dùng **GPU**, batch **512**, W&B **online**, seeds 42/123/3407 tuần tự,
200 epoch tối đa / patience 15. Output:
`runs/level2/gqa-wordnet-full-v2-bs512/gat-kg/<seed>/`.
Không tự thay source/config đã khóa để resume; dùng phiên bản mới nếu đổi
protocol. Model/optimizer/sampler/CPU/CUDA RNG và offset đều được checkpoint.

## Các giai đoạn đã dùng để tạo artifact

Mapping/audit/build KG là bước CPU:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 myenv/bin/python \
  -m tools.prepare_gqa_wordnet_v2
```

Pipeline có các lựa chọn `--stage prepare`, `embedding`, `pilot`, `full`,
hoặc `all`. **Embedding/pilot/full/all có thể chạy GPU**; người dùng chỉ chạy
sau khi kiểm commit và phạm vi lệnh. Artifact v2 hiện đã có, nên dùng stage
`full` khi tiếp tục thực nghiệm hiện tại để tránh thao tác dư.

Toàn bộ chuỗi có thể tái gọi bằng:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 myenv/bin/python -u \
  -m tools.run_gqa_wordnet_v2_pipeline --stage all
```

Stage embedding dùng TransE 64D, batch 512, CUDA, W&B online, 100 epoch /
patience 15. Trainer TransE cũ không có CLI resume; nếu stage ngắn này bị
ngắt trước `metrics.json`, output được giữ nguyên để điều tra và cần một
phiên bản output mới để phục hồi. Full GAT+KG hỗ trợ resume giữa epoch.

## Khóa nguồn và kiểm chứng

Code mở rộng đặt trong `tools/`; không thêm/sửa `src/` để giữ identity các
thực nghiệm cũ. Mapping v1, KG v1, feature v1, checkpoint và report cũ được
giữ nguyên. V2 tham chiếu CSV human review đã chốt bằng SHA256; những lớp
ngoài mẫu giữ nguyên mapping v1. Bảy defer chỉ che nhánh KG, không xóa object
hay supervision của chúng khỏi GQA.

Trước khi khởi chạy đã PASS 27 tests liên quan, gồm cập nhật mapping theo
review, masking/export, thống kê instance/image, exact resume CPU/CUDA và
bộ tổng hợp ba seed. Các test có sử dụng GPU không được assistant tự chạy
tiếp theo quy ước mới. Kiểm chứng hiện trạng nguồn/artifact trong
`reports/level2/gqa-wordnet-v2/preservation.json`.
