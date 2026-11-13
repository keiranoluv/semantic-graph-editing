# MagicBrush joint-VLM: server environment v3

Bản sửa ngày 13/11/2026 cho traceback FlashInfer sampling:

```text
FileNotFoundError: [Errno 2] No such file or directory: 'ninja'
```

`ninja` 1.13.2 đã có ở venv, nhưng terminal đang activate `myenv` khác. Chạy
Python bằng đường dẫn tuyệt đối không tự thêm `bin/` của venv đó vào PATH.
FlashInfer gọi subprocess bằng tên `ninja`, nên bị lỗi dù Python package đã cài.

Kiểm tra CPU cũng phát hiện `nvcc` không có trong PATH, FlashInfer mặc định
trỏ `/usr/local/cuda` không chứa compiler. Compiler CUDA thật nằm trong pip
package `nvidia/cu13`. Runtime/headers đang 13.0, còn NVCC/CRT ban đầu là 13.5;
compiler tìm được vẫn không build được vì headers/compiler không tương thích.
Không dùng lại lượt GPU để dò tiếp từng dependency.

## Đã sửa trong môi trường server

Đã cài pin compiler/CRT/NVVM từ extra nvcc của toolkit 13.0.3.0:

```bash
/workspace/venvs/magicbrush-vllm028/bin/python -m pip install 'cuda-toolkit[nvcc]==13.0.3.0'
```

Kết quả: `nvidia-cuda-nvcc`, `nvidia-cuda-crt`, `nvidia-nvvm` cùng 13.0.88;
runtime giữ 13.0.96. Không thay torch, vLLM, transformers, FlashInfer, model
weights, config hoặc prompt. Không đổi driver CUDA của host. Bộ GCC/headers
CPU đã được cài ở [bước toolchain v1](LEVEL2_MAGICBRUSH_JOINT_TOOLCHAIN_V1.md).

Entry point mới `tools.magicbrush_joint_server_v3` thực hiện:

- Thêm `bin` của đúng Python venv và thư mục NVCC vào đầu PATH.
- Đặt CC/CXX bằng executable thật và giới hạn build CPU `MAX_JOBS=2`.
- Tạo một CUDA toolkit view riêng dưới output mới, gồm bin/include/nvvm và
  lib64; alias libcudart.so/libcuda.so trỏ thư viện đã có. Không sửa `/usr/local/cuda`.
- Đặt CUDA_HOME về view này và khóa version compiler/CRT/NVVM 13.0.88.
- Ghi provenance environment/hash entrypoint và bằng chứng CPU build riêng;
  không sửa bất kỳ source/manifest/config/receipt đã khóa của v1 hoặc v2.
- Trước serve-model, yêu cầu preflight CPU sampling đã thành công và artifact
  có đúng hash. Server vẫn dùng hàm serve/model arguments đã khóa của v1.

## Kiểm tra CPU thực tế

```bash
CUDA_VISIBLE_DEVICES='' /workspace/venvs/magicbrush-vllm028/bin/python -m tools.magicbrush_joint_server_v3 --stage preflight
CUDA_VISIBLE_DEVICES='' /workspace/venvs/magicbrush-vllm028/bin/python -m tools.magicbrush_joint_server_v3 --stage runtime
CUDA_VISIBLE_DEVICES='' /workspace/venvs/magicbrush-vllm028/bin/python -m tools.magicbrush_joint_server_v3 --stage server-plan
```

Preflight đã biên dịch/link đủ ba source của sampling FlashInfer cho kiến trúc
`sm_120f`, bằng đúng Ninja/NVCC/GCC và recipe sinh ra ở lượt khởi động bị lỗi.
Kết quả sampling.so **PASS**; chỉ build trên CPU, không load module, không gọi
CUDA hoặc chạy GPU kernel. Đây chưa phải xác nhận inference GPU thành công.

Tool preflight này dành cho run RTX 5090 đã có recipe sampling trong cache từ
lượt lỗi. Nó không sinh recipe bằng GPU khi recipe thiếu; sẽ báo lỗi rõ. Bản
kiểm đầy đủ nằm tại `preflights/flashinfer-cpu-build-v1.json`. Compiler toolchain
và recipe build riêng nằm dưới `toolchains/`, đều trong output MagicBrush mới.
Các snapshot kiểm CPU lỗi trước đó được giữ lại để chẩn đoán.

Sau khi kiểm recipe riêng thành công, sampling cache thực tế của FlashInfer
cũng đã được build trên CPU để tránh phải build lại toàn bộ khi nạp model.
Recipe trước sửa được lưu riêng, và hash cache sau build nằm tại
`preflights/flashinfer-cache-cpu-build-v1.json`. Đây là cache kernel phát sinh,
không phải dữ liệu nhãn, review hoặc kết quả thực nghiệm.

## Người dùng chạy lại server bằng entrypoint v3

Code/test phải commit trước khi chạy GPU. Từ repo root:

```bash
cd /workspace/workspace/semantic-graph-editing
CUDA_VISIBLE_DEVICES=0 /workspace/venvs/magicbrush-vllm028/bin/python -u -m tools.magicbrush_joint_server_v3 --stage serve-model 2>&1 | tee logs/magicbrush-joint-server-v3.log
```

Model giữ `inferRouter/Qwen3.6-35B-A3B-NVFP4`, revision
`0c24da2038a5d9c373112bc59b90bd0e019fe069`, batch/max_num_seqs 1, seed 42,
frozen inference, TP 1/context 16384. Không đổi MoE Marlin, quantization,
FlashInfer sampler hoặc cách constrained decoding. Output vẫn ở
`data/processed/level2/magicbrush-joint-qwen36-nvfp4-v1/`.
Server không tự gửi annotation request, train hoặc nối tiếp queue. Weights và
các kernel cache đã có được tái sử dụng. Không chạy thêm server thứ hai khi
một server của run còn sống.

Terminal khác kiểm readiness rồi mới smoke một cặp:

```bash
curl --fail http://127.0.0.1:8017/v1/models
```

Khi trả model `magicbrush-joint-v1`:

```bash
myenv/bin/python -u -m tools.magicbrush_joint_v1 --stage infer --scope pilot --limit 1
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m tools.magicbrush_joint_v1 --stage compile --scope pilot
```

Vẫn một cặp hai ảnh, batch 1/seed 42, không tự chạy tiếp 19 cặp, mở rộng hoặc
training. Nếu còn lỗi GPU, đọc traceback trong log mới; không tự đổi recipe.

## Regression tests

```bash
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m pytest tests/editing -q
```

Tests mới kiểm đúng PATH venv, alias layout CUDA, Ninja thiếu, compiler minor
mismatch, symlink/evidence không bị ghi đè, Ninja build không load .so, recipe
thiếu và help chỉ cần thư viện chuẩn. Source v1/v2, config và manifest cũ giữ
nguyên nên preflight provenance của annotation run vẫn phải khớp.
