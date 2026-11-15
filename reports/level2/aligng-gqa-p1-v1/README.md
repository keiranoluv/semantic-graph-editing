# AlignG custom GQA P1 — GPU pilot hoàn tất

User cho phép assistant thực hiện bước1–3, gồm GPU test/pilot; bước4 full
training bàn giao cho user. **Không chạy full training, VG150 hoặc held-out test.**

- Code: `886af54d4aa0c6653d3b86982a06bd94a735c8ac`.
- Config pilot batch16/full: commit `0cd6d5b`; code identity vẫn là886af54.
- Upstream AlignG: `ba2dd471fb3c3bb39030660dcb7f6f105c52446c`.
- RTX5090 32GB, float32, deterministic algorithms, không AMP/TF32.
- Hai cặp pilot, cùng cohort512train/128val, seed42, 5epoch. W&B tắt trong
  test/pilot; full bắt buộc online. Mỗi lần chỉ chạy một model.
- Hai biến thể trong mỗi cặp có cùng toàn bộ initial state và sampler trace
  qua cả5epoch. Best checkpoint hash đúng và reload tái tạo đúng GCmR@50.

## Kết quả validation tại checkpoint chọn bằng GC mR@50

Đơn vị metric %. Chỉ một seed/cohort nhỏ, không phải kết quả benchmark.

| Batch/effective | Model | Best epoch | R@50 | mR@50 | R@100 | mR@100 | Peak tensor CUDA MiB |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | pe-plpr-aligng | 1 | 10.391 | 0.380 | 17.136 | 0.444 | 182.7 |
| 1 | pe-plpr | 4 | 21.345 | 0.630 | 29.069 | 0.710 | 262.1 |
| 16 | pe-plpr-aligng | 5 | 17.892 | 0.435 | 26.338 | 0.526 | 428.5 |
| 16 | pe-plpr | 5 | 18.149 | 0.421 | 25.970 | 0.499 | 640.6 |

Batch1 AlignG kém đối chứng. Batch16 AlignG nhỉnh hơn mR nhưng chênh lệch
rất nhỏ; không đủ kết luận cải thiện. Cả hai cặp chỉ chứng minh pipeline hoạt
động và cho phép kiểm nghiệm tiếp. Không chọn batch theo model thắng: batch16
đã được chốt/commit từ capacity probe trước khi chạy pilotbatch16.

## Bằng chứng correctness và khả năng chạy

- `gpu-probe.json`: D256/C311, 126objects/15,750context pairs mỗi ảnh,
  forward/backward với batch1/8/16 hữu hạn. Peak tensor allocation lần lượt
  203.3/580.6/1009.3MiB; bao gồm graph autograd/gradient, chưa có optimizer
  trong capacity step. Synthetic supervision có4query/image, vì vậy đây
  là stress context lớn, không phải worst-case mọi pattern supervision.
- Cũng kiểm full-query15,750 trên một ảnh và image isolation trên GPU.
- Resume GPU có dropout: cùng history/metric, best epoch và toàn bộ final
  parameters **exact** giữa chạy liên tục và ngắt sau một batch rồi tiếp tục.
  Hai lượt kiểm resume dùng dữ liệu tổng hợp, không phải thực nghiệm chính.
- Pilot thực tế batch16 peak428.5MiB (AlignG),640.6MiB (baseline); có AdamW
  updates. Các loss component/gradient finite. Không có OOM hoặc failure.
- `cache-equivalence.json`: cả640ảnh pilot có feature values và normalized
  record hashes khớp chính xác giữa cache pilot cũ và cache full tái sử dụng.
- CPU: **112passed,0failed,7deselected CUDA**; gồm P0, P1, frozen PE-Net,
  GQA checkpoint/resume và tracker regressions. Kiểm config bổ sung11passed.
- Black/isort/Ruff và commit hooks PASS. Toàn bộ tệp tracked có sẵn trướcP1
  giữ nguyên bytes, gồm P0 source/receipt, notebook và AGENTS cá nhân.

Chi tiết per-epoch loss/gradient/sampling trace, GC/NG, checkpoint hashes nằm
trong `pilot-bs1-summary.json` và `pilot-bs16-summary.json`. `verification.json`
lưu lệnh, artifacts và phạm vi đã chạy. Tensor allocation không bao gồm toàn
bộ CUDA driver/context/reserved memory; không coi đó là tổngVRAM hoặc đảm
bảo cho mọi batch của full dataset.

## Bàn giao bước4

Config `configs/level2/aligng-gqa-full-v1-bs16.json`: full69,942train/5,000val,
batch/effective16, valbatch1, seeds42/123/3407, max200epoch/patience15,
AdamW0.001/decay0.01/clip5, loss và evaluator giữ nguyên. W&B online bắt buộc.
Main output `runs/level2/aligng-gqa-v1/full-bs16/{model}/{seed}/`.
Không queue hoặc tự nối job. Full đã qua **CPU plan**, chưa được chạy.

[Lệnh train/resume và mô tả đầy đủ](../../../docs/LEVEL2_ALIGNG_GQA_P1.md).
P1 chưa kiểm live W&B online hoặc full-dataset train; full W&B wiring và
fail-closed guard đã có CPU tests. Đối chứng chính phải cùng batch16 và3seed,
không dùng kết quả batch512 lịch sử thay thế. VG150 được để lại cho bước sau.
