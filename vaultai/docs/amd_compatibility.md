# AMD AI MAX 395+ Compatibility Notes

Hardware: AMD Ryzen AI MAX 395+ (Strix Halo), 128 GB unified memory, ROCm 6.x/7.x

---

## Component status

| Component | Status | Notes |
|---|---|---|
| `non_cuda_equivalents.py` | Compatible | Full Python fallback; tries `libamdhip64.so` |
| `requirements/rocm_core.txt` | Compatible | ROCm deps (cupy-rocm-7-0) already listed |
| `setup.py` HIP build | Needs work | `BUILD_WITH_HIP=1` path exists, untested on device |
| CUDA kernels (`csrc/*.cu`) | Needs work | Hipify infra ready, needs validation |
| vLLM integration | Compatible | Delegates to vLLM multi-device platform detection |
| `cachegen_decoder.py` | Blocked | Hardcoded `.cuda()` calls |
| `cachegen_encoder.py` | Blocked | Hardcoded `.cuda()` calls |
| `usage_context.py` | Partial | No AMD branch — falls through to CPU count |
| TensorRT-LLM adapter | Irrelevant | NVIDIA-only, skip |
| Docker images | Blocked | NVIDIA base images only — needs custom ROCm Dockerfile |

---

## Unified memory advantage

On discrete GPU systems, LMCache's tier boundary (GPU VRAM → CPU RAM) costs PCIe bandwidth.
On Strix Halo, GPU and CPU share the same 128 GB LPDDR5X pool at ~256 GB/s.
There is no copy cost between the "GPU tier" and "CPU tier" — both see the same physical addresses.
CPU-mode LMCache on this device outperforms GPU-mode LMCache on a typical PCIe-limited workstation.

---

## Patches needed for GPU-accelerated mode

### 1. cachegen_decoder.py and cachegen_encoder.py

Replace hardcoded `.cuda()` with device-agnostic `.to(tensor.device)`:

```python
# Before
return ret.cuda()

# After
return ret.to(tensor.device)
```

Files:
- `lmcache/storage_backend/serde/cachegen_decoder.py`
- `lmcache/storage_backend/serde/cachegen_encoder.py`

### 2. usage_context.py — AMD device detection

Add ROCm/HIP branch after the XPU check:

```python
if torch.cuda.is_available():
    # CUDA path
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    # Intel XPU path
elif hasattr(torch, "cuda") and torch.version.hip is not None:
    # AMD ROCm path  ← add this
    gpu_count = torch.cuda.device_count()
else:
    gpu_count = psutil.cpu_count()
```

### 3. HIP kernel compilation

```bash
BUILD_WITH_HIP=1 pip install -e . --no-build-isolation
```

Validate on device before enabling in production. Requires ROCm SDK 7.0+ installed.

---

## Safe deployment path (no patches needed)

```yaml
# vaultai/config/lmcache_cpu.yaml
local_cpu: true
max_local_cpu_size: 20
remote_serde: "naive"
remote_url: "fs://localhost:0/var/vaultai/lmcache"
```

```bash
LMCACHE_CONFIG_FILE=vaultai/config/lmcache_cpu.yaml \
vllm serve <model> --gpu-memory-utilization 0.7 --port 8000
```
