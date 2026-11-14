# AlignG custom GQA — P1 và launcher full v1

Namespace riêng, giữ nguyên toàn bộ P0, `src/`, config/cache/checkpoint lịch sử.
User cho phép assistant chạy GPU test/pilot trong lượt P1; full training chỉ
bàn giao lệnh cho user. Không chạy VG150, không mở annotation test.

## Kiến trúc và reproducibility

`engine.py` là bản riêng của control flow `src/training/gqa_full.py` tại
4e149ce, thay wiring model/batch/evaluation. Không monkeypatch module gốc.
Adapter/loss/sampler P0 không thay đổi. Cả `pe-plpr-aligng` và `pe-plpr`
dựng cùng model với feedback bật/tắt: cùng initial state toàn bộ; nhánh tắt
chạy forward PE-Net gốc và không cập nhật các parameter feedback không dùng.
Reset dropout RNG sau constructor; sampling dùng Generator riêng cùng seed.

Checkpoint atomic `last.pt`: model, AdamW, CPU/CUDA RNG, sampler RNG, permutation,
index giữa epoch, best state và history. SIGINT/SIGTERM dừng tại ranh giới batch,
lưu rồi trả 130; chạy lại cùng lệnh để tiếp tục. Không đổi config/source để resume.
Best validation GC mR@50, hòa giữ epoch sớm; full max200/patience15/min_delta0.
Evaluator GC/NG giữ duplicate GT và denominator310. Đây là custom GQA, không GQA200.

`runner.prepare` xác minh metadata pin P0, toàn bộ HDF5 shard SHA256, GloVe
vectors, train/val split/cohort và source hashes. Dùng cache full lịch sử cho
cả pilot/full; pilot lấy đúng 512/128 image IDs của cohort lịch sử. Không
load encoder, tạo feature hay sửa cache. Full dùng RAM loader gốc (4 CPU
workers); pilot load 640 graph CPU. Có lock GPU dùng chung với GQA/MagicBrush.

`experiment.json` khóa source commit và hashes (gồm P0, P1 và dependency
lịch sử), upstream AlignG, config, splits, cohort, asset identities, versions,
evaluator/selection/sampling. Run identity khóa model+seed. Checkpoint/result
hash được kiểm trước khi skip run hoàn tất. Commit chỉ sửa báo cáo không đổi
code identity; sửa source cần version/output mới. Không đưa thay đổi cá nhân
AGENTS/notebook vào source identity.

## Lệnh pilot được user cho phép assistant chạy

Chạy từ repo root. Mỗi lệnh chỉ một job, không tự queue.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 myenv/bin/python -m pytest -q tests/aligng_gqa_v1 tests/aligng_adapted_v1 tests/training/test_gqa_full.py tests/training/test_gqa_tracking.py tests/training/test_penet.py -k 'not cuda'
myenv/bin/python -m experiments.aligng_gqa_v1.runner --stage plan
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 myenv/bin/python -u -m experiments.aligng_gqa_v1.probe --device cuda --output runs/level2/aligng-gqa-v1/gpu-probe
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 myenv/bin/python -u -m experiments.aligng_gqa_v1.runner --stage pilot --model pe-plpr-aligng --seed 42
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 myenv/bin/python -u -m experiments.aligng_gqa_v1.runner --stage pilot --model pe-plpr --seed 42
myenv/bin/python -m experiments.aligng_gqa_v1.runner --stage summary
```

Pilot: config `configs/level2/aligng-gqa-pilot-v1.json`, batch/effective1,
val batch1, seed42, 512train/128val, 5epochs, W&B disabled. Output:
`runs/level2/aligng-gqa-v1/pilot-bs1/{pe-plpr-aligng|pe-plpr}/42/`.
Probe: synthetic worst-case126objects/image, batch1/8/16, một forward/backward
mỗi batch, full-query15750, exact interrupted resume, isolation. Không train
chính và không dùng dữ liệu test. Bằng chứng kết quả sẽ nằm ở báo cáo P1 riêng.

## Full training do user chạy sau khi hai pilot PASS

CLI `--stage train --config <full-config> --model <variant> --seed <seed>`
yêu cầu model/seed tường minh (42/123/3407), một job/lần. Full config được
chốt sau GPU probe/pilot. Gate yêu cầu hai pilot hoàn tất, initial weights/
sampling khớp, best checkpoint đúng hash; không đòi AlignG phải thắng baseline.
Full bắt buộc W&B online `keiranoluv-hcmus/semantic-graph-editing`; fail trước
optimizer nếu tracker không online. Logging thực hiện sau durable checkpoint.
Không fallback offline. Không tự nối tiếp seed/model, không tự mở test split.

Giữ D256, heads2, GRU, dropout0.2, pair_chunk1024, attention query_chunk64,
checkpoint activation bật, float32 (không AMP/TF32), AdamW lr0.001/decay0.01,
clip norm5; loss k1=10/k2=1/gamma1=1/gamma2=7/unit weights. Full batch sẽ
được chọn từ kiểm tra kỹ thuật, cùng batch cho đối chứng. Không so pilot
batch1 trực tiếp với kết quả full batch512 lịch sử để kết luận cải thiện.
