"""Fail early on unsupported native CUDA architecture; explicit tiny kernel probe."""

import re


def validate_cuda_arch(capability, arches):
    major, minor = capability
    native = []
    for arch in arches:
        match = re.fullmatch(r"sm_(\d+)(\d)", arch)
        if match:
            native.append(tuple(map(int, match.groups())))
    # Native cubins can cover later minor revisions of the same major. Do not
    # assume PTX fallback or architecture-specific ('a') kernels are sufficient.
    if not any(m == major and n <= minor for m, n in native):
        raise RuntimeError(
            f"cuda_arch_not_supported: GPU sm_{major}{minor}, wheel arches={arches}. "
            "Use requirements-level3-sgg-robin-cu128-v2.txt and "
            "scripts/run_level3_sgg_robin_cu128_v2.sh; no inference was started."
        )


def cuda_device_evidence():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU fallback")
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    arches = torch.cuda.get_arch_list()
    validate_cuda_arch(capability, arches)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("ROBIN requires BF16 support")
    return {
        "device_index": device,
        "name": torch.cuda.get_device_name(device),
        "capability": list(capability),
        "compiled_cuda_arches": arches,
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
    }


def kernel_probe():
    """Only called by the explicit gpu-smoke command, never CPU preflight."""
    import torch
    from torchvision.ops import nms

    with torch.inference_mode():
        source = torch.arange(24, device="cuda").reshape(2, 3, 4)
        moved = source.permute(2, 0, 1).contiguous()
        assert moved.shape == (4, 2, 3) and moved[3, 1, 2].item() == 23
        values = torch.ones((8, 8), device="cuda", dtype=torch.bfloat16)
        assert torch.equal((values @ values).cpu(), torch.full((8, 8), 8, dtype=torch.bfloat16))
        boxes = torch.tensor([[0, 0, 10, 10], [0, 0, 10, 10]], device="cuda", dtype=torch.float32)
        scores = torch.tensor([0.9, 0.8], device="cuda")
        assert nms(boxes, scores, 0.5).cpu().tolist() == [0]
    torch.cuda.synchronize()
    return {"permute_contiguous": True, "bf16_matmul": True, "torchvision_cuda_nms": True}
