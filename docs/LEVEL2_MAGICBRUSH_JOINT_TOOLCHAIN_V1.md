# MagicBrush joint-VLM: compiler preflight v1

Ngày sửa 13/11/2026. Server đã tải xong weights trong khoảng 313 giây, nạp
model thành công (21.88 GiB VRAM), rồi dừng khi profile vision encoder:

```text
RuntimeError: Failed to find C compiler. Please specify via CC environment variable or set triton.knobs.build.impl.
```

Triton cần compiler C để build CUDA **host helper**. Môi trường container
Ubuntu 24.04 thiếu cả gcc/clang và `/usr/include/python3.12/Python.h`.
Warning Marlin FP4 không phải exception làm server dừng trong log này; không
đổi checkpoint, quantization, model flags hoặc pixel budget để chữa lỗi compiler.
Log này chưa có CUDA OOM; GPU inference hai ảnh vẫn chưa được xác nhận.

Đã cài trên CPU trong container hiện tại:

```bash
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends build-essential python3.12-dev
```

Đây là dependency hệ điều hành; dựng lại container mới cần cài lại hoặc đưa
vào image riêng. Không thay pip packages ở `myenv`/môi trường vLLM, không cài
CUDA driver, không khởi động hay kết thúc process GPU. Apt đã cài các dependency
CPU và nâng bản vá libc/dpkg/zlib theo repository Ubuntu; log package tại
`/tmp/magicbrush-toolchain-apt-install.log`.

Đã kiểm thực tế bằng Python của môi trường server:

```bash
CUDA_VISIBLE_DEVICES='' /workspace/venvs/magicbrush-vllm028/bin/python -m tools.magicbrush_joint_toolchain_v1
CUDA_VISIBLE_DEVICES='' /workspace/venvs/magicbrush-vllm028/bin/python -m tools.magicbrush_joint_server_v2 --stage runtime
```

Compiler gcc 13.3.0, headers Python 3.12.3, Triton 3.7.1: **PASS**. Tool mới
biên dịch và link đúng `triton/backends/nvidia/driver.c`, với Python/CUDA headers
và `libcuda.so.1`, ra một shared library tạm rồi xóa nó. Không load thư viện,
không gọi CUDA, không tải model hoặc chạy GPU kernel. Report có version/hash
của tool/source driver/CC và đường dẫn library; đây chỉ là xác minh toolchain
CPU, không thay thế smoke GPU.

Giữ nguyên mọi source/config/manifest/cache của annotation v1 và entrypoint
server v2 đã khóa. Tool preflight có version riêng, không làm đổi hash run cũ.
Runtime/source hashes vLLM sau sửa vẫn khớp receipt đã tạo. Tests mới kiểm lỗi
compiler, headers, libcuda, CC override và việc compile không load shared library.

Sau khi code/test đã commit, **người dùng** chạy lại server bằng lệnh riêng.
Dùng log mới để giữ nguyên bằng chứng lỗi trước:

```bash
cd /workspace/workspace/semantic-graph-editing
CUDA_VISIBLE_DEVICES=0 /workspace/venvs/magicbrush-vllm028/bin/python -u -m tools.magicbrush_joint_server_v2 --stage serve-model 2>&1 | tee logs/magicbrush-joint-server-v2-toolchain1.log
```

Model `inferRouter/Qwen3.6-35B-A3B-NVFP4`, revision
`0c24da2038a5d9c373112bc59b90bd0e019fe069`, frozen inference, batch/max_num_seqs
1, seed 42. Output giữ ở
`data/processed/level2/magicbrush-joint-qwen36-nvfp4-v1/`; weights đã cache được
tái sử dụng. Không tự chạy annotation, training hay nối tiếp lượt khác.

Ở terminal khác, chỉ chạy smoke khi endpoint đã trả model `magicbrush-joint-v1`:

```bash
curl --fail http://127.0.0.1:8017/v1/models
myenv/bin/python -u -m tools.magicbrush_joint_v1 --stage infer --scope pilot --limit 1
```

Smoke chỉ một cặp hai ảnh, cùng batch 1/seed 42. Giữ recipe/cohort đã khóa;
không xóa cache hoặc sửa receipt để tránh kiểm provenance. Nếu server còn lỗi,
đọc traceback mới thay vì tự thay model/backend.
