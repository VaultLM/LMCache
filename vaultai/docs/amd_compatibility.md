# AMD AI MAX 395+ Compatibility Notes

**Hardware:** AMD Ryzen AI MAX 395+ (Strix Halo), 128 GB unified memory  
**Architecture:** gfx1151 (RDNA 3.5 APU) — no discrete GPU, ROCm 6.x/7.x  
**Last reviewed:** 2026-05-04

---

## Unified memory advantage

On discrete GPU systems, LMCache's tier boundary (GPU VRAM → CPU RAM) costs PCIe bandwidth (~40–60 GB/s).  
On Strix Halo, GPU and CPU share the same 128 GB LPDDR5X pool at ~256 GB/s.  
There is **no copy cost** between tiers — both CPU and GPU compute units see the same physical addresses.  
CPU-mode LMCache on this device outperforms GPU-mode LMCache on a typical PCIe-limited workstation.

---

## Component status

| Component | Status | Notes |
|---|---|---|
| `non_cuda_equivalents.py` | **Compatible** | Full Python fallback; explicitly tries `libamdhip64.so` |
| `requirements/rocm_core.txt` | **Compatible** | ROCm deps (`cupy-rocm-7-0`) already listed |
| `setup.py` HIP build (`BUILD_WITH_HIP=1`) | **Needs work** | Hipify infra ready; untested on gfx1151 |
| CUDA kernels (`csrc/*.cu`) | **Needs work** | HIP conversion path exists, needs gfx1151 validation |
| `lmcache/v1/` memory management | **Compatible** | Falls back through `c_ops` alias to Python fallback |
| vLLM integration | **Compatible** | Delegates to vLLM's multi-device platform detection |
| `cachegen_decoder.py` | **Blocked (upstream fix incoming)** | Hardcoded `.cuda()` — see PR #3091/#3168 |
| `cachegen_encoder.py` | **Blocked (upstream fix incoming)** | Hardcoded `.cuda()` — see PR #3091/#3168 |
| `usage_context.py` AMD detection | **Partial (upstream fix incoming)** | No ROCm branch; falls to CPU — see PR #3091/#3168 |
| GDS / NVMe Direct Storage | **Incoming** | hipFile shim in PR #3115 (approved) |
| CacheBlend sparse attention | **Needs gfx1151 validation** | PR #3092 works on gfx942/gfx950, Triton may work on gfx1151 |
| TensorRT-LLM adapter | **Irrelevant** | NVIDIA-only, skip entirely |
| Official Docker images | **Blocked** | NVIDIA base images; PR #3101 adds ROCm but targets gfx942/gfx950 |

---

## Upstream PRs to track

These PRs are in review and directly remove work we would otherwise have to do ourselves.  
**Do not patch the affected files in our fork — wait for these to merge then pull.**

### PR #3091 — Global device abstraction
**URL:** https://github.com/LMCache/LMCache/pull/3091  
**Author:** hlin99 (Tony Lin)  
**Status:** Open, many reviewers requested, labeled "full"  
**What it does:** Replaces all `torch.cuda.*` hardcoding across the entire codebase with a unified `torch_dev` / `torch_device_type` layer. Single entry point in `lmcache/__init__.py`. Covers vLLM, SGLang, TensorRT-LLM integrations, cache engine, memory management, storage backends, multiprocess servers.  
**Impact for VaultAI:** Eliminates the `cachegen_decoder/encoder.py` `.cuda()` blockers and `usage_context.py` AMD detection gap in one shot.

### PR #3168 — torch dev for CPU only (experimental)
**URL:** https://github.com/LMCache/LMCache/pull/3168  
**Author:** hlin99 (Tony Lin)  
**Status:** Open, experimental  
**What it does:** CPU-only variant of the #3091 device abstraction work. More conservative scope. Adds runtime guards for CUDA-only features (IPC events, cudart pinning) so they fail cleanly instead of crashing.  
**Impact for VaultAI:** Directly improves CPU-mode reliability on our hardware. Likely to merge before #3091.

### PR #3082 — GPU kernel stubs + non-CUDA fallback parity
**URL:** https://github.com/LMCache/LMCache/pull/3082  
**Author:** maobaolong  
**Status:** Open  
**What it does:** Adds pure-Python fallback for block-level KV transfer (`multi_layer_block_kv_transfer`). Adds no-op stubs (`PageBufferShapeDesc`, `record_event_on_stream`, `drain_recorded_events`) to keep CPU-only platforms API-compatible. Enforces 1:1 symbol parity between `c_ops` and `non_cuda_equivalents`.  
**Impact for VaultAI:** Strengthens the CPU fallback path we currently depend on.

### PR #3115 — hipFile shim layer (AMD GDS) — **Approved**
**URL:** https://github.com/LMCache/LMCache/pull/3115  
**Author:** riley-dixon  
**Status:** Open, approved by ApostaC (core maintainer)  
**What it does:** Wraps AMD's `hipFile` package behind the `cufile-python` API. Enables GPU Direct Storage on AMD — KV cache reads/writes directly between GPU and NVMe without CPU involvement.  
**Impact for VaultAI:** On Strix Halo's unified memory, the GPU and NVMe can exchange KV tensors at full NVMe bandwidth with no CPU overhead. High-priority pull once merged.

### PR #3092 — ROCm Triton sparse attention for CacheBlend
**URL:** https://github.com/LMCache/LMCache/pull/3092  
**Author:** andyluo7  
**Status:** Open, approved by sammshen  
**What it does:** Adds `LMCTritonSparseBackend` as a drop-in for `LMCFlashInferSparseBackend`. Enables CacheBlend (non-prefix KV reuse) on AMD without flashinfer. Auto-detects ROCm and routes accordingly. Tested on MI300X (gfx942) and MI355X (gfx950).  
**Impact for VaultAI:** CacheBlend on AMD. Triton kernels are architecture-agnostic but **need gfx1151 validation** — confirmed working on Instinct, not yet on Strix Halo APU.

### PR #3101 — ROCm Dockerfiles for AMD Instinct
**URL:** https://github.com/LMCache/LMCache/pull/3101  
**Author:** andyluo7  
**Status:** Open, approved by Shaoting-Feng  
**What it does:** Adds `Dockerfile.rocm` and `Dockerfile.rocm-lightweight` using `rocm/dev-ubuntu-24.04:7.0-complete` base. Builds HIP extensions with `BUILD_WITH_HIP=1`. Default GPU targets: `gfx942`, `gfx950`.  
**Impact for VaultAI:** Template for our own ROCm Dockerfile. We need to change `PYTORCH_ROCM_ARCH=gfx1151` for Strix Halo. Cannot use these images directly — they target datacenter Instinct GPUs, not the APU.

---

## The gfx1151 gap

All upstream AMD work targets **Instinct datacenter GPUs** (gfx942 / gfx950 — CDNA architecture).  
Our device is **gfx1151** — Strix Halo RDNA 3.5 APU. Different chip family, different ISA extensions.

What this means practically:
- ROCm 6.2+ officially supports RDNA 3 (gfx1100 family); gfx1151 support varies by ROCm version
- HIP kernel compilation needs explicit `PYTORCH_ROCM_ARCH=gfx1151`
- Triton kernels (PR #3092) are architecture-agnostic but need runtime validation
- GPU Direct Storage (PR #3115) behavior on unified memory needs testing — there is no physical NVMe-to-VRAM DMA boundary on APU

**This gap is VaultAI's primary contribution target.** Upstream won't close it. We validate and document gfx1151 behavior.

---

## Safe deployment path today (no patches needed)

```yaml
# vaultai/config/lmcache_cpu.yaml
chunk_size: 256
local_cpu: true
max_local_cpu_size: 20      # GB of unified RAM for hot KV cache
save_unfull_chunk: false
remote_serde: "naive"       # bypasses cachegen .cuda() calls entirely
remote_url: "fs://localhost:0/var/vaultai/lmcache"
```

```bash
LMCACHE_CONFIG_FILE=vaultai/config/lmcache_cpu.yaml \
vllm serve <model> --gpu-memory-utilization 0.7 --port 8000
```

This path:
- Does not trigger `cachegen_decoder/encoder.py` (avoided by `remote_serde: naive`)
- Does not require HIP kernel compilation
- KV tensors live and stay in unified pool — no copy overhead
- Disk overflow goes to NVMe at full throughput

---

## Patches to write ourselves (post upstream merges)

After PRs #3091/#3168 merge and we pull them, these are the remaining gfx1151-specific items:

1. **Custom ROCm Dockerfile** — fork PR #3101's Dockerfile, change `PYTORCH_ROCM_ARCH=gfx1151`
2. **gfx1151 HIP kernel validation** — run `BUILD_WITH_HIP=1` build on device, report results upstream
3. **CacheBlend gfx1151 test** — validate PR #3092 Triton kernels on RDNA 3.5, report upstream
4. **Unified memory GDS behavior** — test PR #3115 hipFile on APU (no discrete NVMe-GPU DMA path)
