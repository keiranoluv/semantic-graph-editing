# MagicBrush joint-VLM: entrypoint server v2

Bản sửa ngày 13/11/2026 cho lỗi `ModuleNotFoundError: No module named 'h5py'`
khi chạy `--stage runtime` trong môi trường vLLM riêng.

CLI đa stage của v1 import review ngay lúc khởi động. Review dùng module GQA
có phụ thuộc h5py. Server không cần review/GQA; cài thêm h5py không giải quyết
việc CLI nạp các phần không liên quan. Entry point mới
`tools.magicbrush_joint_server_v2` chỉ import các thành phần runtime/server.
`--help` chỉ cần thư viện chuẩn Python. Runtime/server-plan không khởi tạo GPU.

Giữ nguyên toàn bộ `tools/magicbrush_joint_v1/`, prompt, schema, config, output,
manifest và `run.json` đã khóa ở commit `e631993`. Đây là bản sửa đường vào
server, không đổi annotation protocol, cohort hoặc cache identity. Launcher v2
ghi provenance riêng tại
`data/processed/level2/magicbrush-joint-qwen36-nvfp4-v1/launchers/server-entrypoint-v2.json`
khi người dùng chủ động gọi serve-model: hash entrypoint, commit mới, hash và
commit annotation run gốc, model, batch và seed. `server.json` vẫn do server
v1 đã khóa tạo, với runtime/model receipt như trước.

Dùng các lệnh dưới đây thay cho lệnh **runtime/server-plan/serve-model** của
[tài liệu v1](LEVEL2_MAGICBRUSH_JOINT_VLM_V1.md). Các lệnh annotation, compile
và review tiếp tục dùng `tools.magicbrush_joint_v1` trong `myenv` của repo.
Không cần cài h5py vào môi trường server cho bản sửa này.

CPU kiểm tra thực tế bằng `/workspace/venvs/magicbrush-vllm028/bin/python` đã
qua: vLLM 0.28.0, torch 2.13.0, transformers 5.22.2, compressed-tensors 0.17.0,
flashinfer-python 0.6.16.post3, safetensors 0.8.0; bốn file runtime/registry
được tìm thấy và hash thành công. Đây là kiểm tra metadata/source trên CPU,
chưa xác nhận GPU kernel, load weights hoặc inference thành công.

## Terminal 1: kiểm tra CPU, rồi người dùng khởi động server

Code phải được commit và test CPU trước khi người dùng gọi GPU. Chạy từ repo:

```bash
cd /workspace/workspace/semantic-graph-editing
CUDA_VISIBLE_DEVICES='' /workspace/venvs/magicbrush-vllm028/bin/python -m tools.magicbrush_joint_server_v2 --stage runtime
CUDA_VISIBLE_DEVICES='' /workspace/venvs/magicbrush-vllm028/bin/python -m tools.magicbrush_joint_server_v2 --stage server-plan
```

Khi GPU dành cho bạn rảnh, tự chạy lệnh sau; giữ terminal này hoạt động:

```bash
mkdir -p logs
CUDA_VISIBLE_DEVICES=0 /workspace/venvs/magicbrush-vllm028/bin/python -u -m tools.magicbrush_joint_server_v2 --stage serve-model 2>&1 | tee logs/magicbrush-joint-server-v2.log
```

Model vẫn là `inferRouter/Qwen3.6-35B-A3B-NVFP4`, revision
`0c24da2038a5d9c373112bc59b90bd0e019fe069`, batch/max_num_seqs 1, seed 42,
frozen inference, TP 1, context 16384. Output vẫn ở
`data/processed/level2/magicbrush-joint-qwen36-nvfp4-v1/`.
Server không tự gửi request, train hoặc nối tiếp lượt khác. Lần đầu có thể
tải weights khi người dùng chủ động chạy. Không sửa hoặc dừng process GPU khác.

## Terminal 2: readiness và smoke một cặp

```bash
cd /workspace/workspace/semantic-graph-editing
curl --fail http://127.0.0.1:8017/v1/models
```

Chỉ khi có model `magicbrush-joint-v1`, chạy riêng smoke và CPU compile:

```bash
myenv/bin/python -u -m tools.magicbrush_joint_v1 --stage infer --scope pilot --limit 1
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m tools.magicbrush_joint_v1 --stage compile --scope pilot
```

Phạm vi một cặp hai ảnh của cohort 20 train đã khóa; batch 1, seed 42. Không
tự chạy tiếp 19 cặp còn lại hoặc mở rộng/train. Nếu server không lên, lấy lỗi:

```bash
tail -n 60 logs/magicbrush-joint-server-v2.log
```

## Regression tests CPU

```bash
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m pytest tests/editing -q
```

Tests mới chặn import h5py/pyarrow/wandb/torch_geometric, GQA, review, runner và
CLI đa stage cũ trong subprocess; runtime/server-plan vẫn phải chạy được và
không được gọi serve. Chúng cũng kiểm help không cần site-packages, provenance
riêng và việc entrypoint server từ chối các stage annotation.
