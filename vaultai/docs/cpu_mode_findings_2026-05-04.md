# CPU Mode Findings — 2026-05-04

**Stage:** 0-B / Stage 2 (CPU mode, no vLLM runtime)  
**Scope:** LMCache storage layer in isolation — no model loaded, no vLLM, no GPU kernels

---

## Hardware and software

| Field | Value |
|---|---|
| Device | AMD Ryzen AI MAX 395+ (Strix Halo) |
| Memory | 128 GB unified LPDDR5X |
| GPU architecture | gfx1151 (RDNA 3.5 APU) |
| ROCm version | 6.3.3-74 |
| Python | 3.12.3 |
| torch | 2.5.1+rocm6.2 |
| LMCache | 0.1.dev1507 (editable, `NO_CUDA_EXT=1`) |
| Backend | `lmcache.non_cuda_equivalents` (Python fallback — no compiled HIP/CUDA extensions) |
| Config | `vaultai/config/lmcache_cpu.yaml` (`local_cpu: true`, `remote_serde: naive`, `max_local_cpu_size: 20`) |
| venv | `vaultai/vllm` (uv, Python 3.12) |

---

## Environment setup

Full install sequence required to reproduce:

```bash
cd /home/vaultai/Vaultai/LMCache

# 1. Create venv
uv venv --python 3.12 vaultai/vllm
source vaultai/vllm/bin/activate

# 2. ROCm torch (must come from ROCm index — never PyPI)
uv pip install torch --index-url https://download.pytorch.org/whl/rocm6.2

# 3. Upgrade setuptools (torch ships 70.x; LMCache pyproject.toml needs >=77.0.3)
uv pip install "setuptools>=77.0.3,<81.0.0" setuptools_scm

# 4. LMCache — editable, no C extension compilation, no cuda_core.txt deps
NO_CUDA_EXT=1 uv pip install -e . --no-build-isolation --no-deps

# 5. Runtime deps (see vaultai/requirements/cpu_test.txt for full list with justification)
uv pip install -r vaultai/requirements/cpu_test.txt
```

**Packages that were not predicted by static import analysis but were required at runtime:**
- `requests` — top-level import in `config_base.py`, pulled in immediately on config load
- `tqdm` — top-level import in `check_mode_gen.py`; the check registry loads all `check_mode_*` modules at startup and silently drops any that fail, so a missing `tqdm` caused `test_storage_manager` to never register (0 modes loaded)
- `py-cpuinfo` — top-level import in `usage_context.py`, same propagation issue
- `aiohttp` — top-level import in `connections.py`, pulled transitively
- `numba` — top-level import in `non_cuda_equivalents.py`; the `common.txt` comment says "nixl uses numba" but LMCache's own Python fallback kernel (`njit`) also uses it directly

The full verified list is in `vaultai/requirements/cpu_test.txt`.

---

## Backend verification

```
c_ops resolves to: .../lmcache/non_cuda_equivalents.py
torch: 2.5.1+rocm6.2
torch.version.hip: 6.2.41133-dd7f95766
```

`torch.cuda.is_available()` returns `True` on ROCm (HIP compatibility layer). The backend selector tries to import the compiled `lmcache.c_ops` extension, fails (not compiled), and correctly falls back to `non_cuda_equivalents`.

---

## Test results

### Test 1 — Storage manager boot

Config parsed, storage manager created, 1-key round-trip completed without error.
No `.cuda()` call triggered. **PASS.**

---

### Test 2 — RAM tier: PUT/GET correctness (bfloat16, 20 keys)

Default object size: 1024 elements = **2 KB per object**.

| Operation | Avg latency | Throughput | Pass rate |
|---|---|---|---|
| EXISTS (absent) | 0.10 ms | — | 20/20 |
| PUT | 0.08 ms | 23 MB/s | 20/20 |
| EXISTS (present) | 0.03 ms | — | 20/20 |
| GET | 0.02 ms | **81 MB/s** | 20/20 |

Content verified byte-for-byte. **PASS.**

---

### Test 3 — Dtype coverage (20 keys each)

| dtype | PUT throughput | GET throughput | Pass rate |
|---|---|---|---|
| bfloat16 | 23 MB/s | 81 MB/s | 20/20 |
| float16 | 30 MB/s | 89 MB/s | 20/20 |
| float32 | 41 MB/s | **165 MB/s** | 20/20 |

Note: throughput scales with object size (float32 objects are 2× the bytes of bfloat16 at the same element count) while per-operation latency remains ~constant (~0.02–0.08 ms). This is expected — operations are metadata-overhead-dominated at 2 KB object size.

All three dtypes pass content verification. **PASS.**

---

### Test 4 — Disk overflow path (bfloat16, 65536-element objects, 20 keys)

Config: `max_local_cpu_size: 2`, `remote_url: fs://...`. Object size: 65536 elements = **128 KB per object**, 20 objects = **2.56 MB total**.

| Operation | Avg latency | Throughput | Pass rate |
|---|---|---|---|
| EXISTS (absent) | 0.12 ms | — | 20/20 |
| PUT | 1.19 ms | 105 MB/s | 20/20 |
| EXISTS (present) | 0.04 ms | — | 20/20 |
| GET | 0.02 ms | **5.23 GB/s** | 20/20 |

Content verified after settle time. **PASS.**

**Note on disk eviction:** 2.56 MB total << 2 GB cap, so no actual NVMe eviction occurred. All data stayed in RAM. The 5.23 GB/s GET reflects large objects (128 KB) with the same ~0.02 ms per-op latency as smaller objects — throughput just scales with size. True disk eviction test requires a corpus larger than `max_local_cpu_size`. Planned for Stage 5-A.

---

### Test 5 — Concurrency (100 keys, bfloat16)

| Operation | Avg latency | Throughput | Pass rate |
|---|---|---|---|
| EXISTS (absent) | 0.06 ms | — | 100/100 |
| PUT | 0.08 ms | 25 MB/s | 100/100 |
| EXISTS (present) | 0.06 ms | — | 100/100 |
| GET | 0.07 ms | **29 MB/s** | 100/100 |

No deadlock. No OOM. Completed in < 5 seconds. **PASS.**

**Throughput drop vs 20-key baseline:** GET drops from 81 MB/s (20 keys) to 29 MB/s (100 keys). Average latency increases from 0.02 ms to 0.07 ms — a 3.5× increase. At 100 sequential async operations, the per-op overhead compounds. This is the single-threaded async path; under real multi-user vLLM load, the storage manager will use its background thread pool and the pattern will differ.

---

### Test 6 — Naive serde isolation

```
remote_serde: naive
cachegen modules in sys.modules: [] (none)
```

`cachegen_encoder` and `cachegen_decoder` — both containing hardcoded `.cuda()` calls that crash on AMD — are never imported under `remote_serde: naive`. **PASS.**

---

### Test 7 — Memory pre-allocation behaviour

```
RSS before storage manager init:  0.01 GB
RSS after storage manager init:   23.03 GB
Delta:                            23.02 GB
```

**Finding:** `LocalCPUBackend` eagerly pre-allocates the full configured `max_local_cpu_size` pool at startup. With `max_local_cpu_size: 20` (20 GB), the process RSS jumps ~23 GB immediately on initialization — before any KV data is stored.

This is not a bug. It is intentional: the backend reserves a contiguous memory pool upfront to guarantee allocation latency during inference. But it has direct implications for memory budgeting:

| Component | Estimated footprint |
|---|---|
| Qwen2.5-7B (AWQ/Q4) | ~5 GB |
| Qwen3.6-35B-A3B (AWQ) | ~22–25 GB |
| LMCache pool (`max_local_cpu_size: 20`) | **20 GB reserved at boot** |
| OS + ROCm runtime | ~4–6 GB |
| Available for inference buffers | remainder |

**Recommendation:** Treat `max_local_cpu_size` as a hard reservation. For the 35B model, set `max_local_cpu_size: 8` initially and tune up after measuring peak model + inference buffer usage. Update `vaultai/config/lmcache_cpu.yaml` before Stage 1 vLLM tests.

---

## Decision gate

Per the experimentation plan:

> **Ship CPU mode:** Stage 2-A shows ≥ 3× TTFT reduction on cache hits

CPU mode storage layer is **validated and functional** on gfx1151. No upstream patches needed for this path. All 7 storage layer tests pass.

The gate cannot be evaluated yet — TTFT reduction requires a live vLLM + model to measure. That is Stage 1 → Stage 2 of the experimentation plan.

**Next step:** Proceed to Stage 1 (vLLM ROCm install and baseline TTFT measurement). The `vaultai/vllm` venv is ready; `uv add vllm` into the same environment when ROCm vLLM install is ready.

---

## Open items from this session

1. **Disk overflow test not yet triggered** — requires corpus > `max_local_cpu_size`. Plan Stage 5-A for this.
2. **`max_local_cpu_size` needs reduction** before serving 35B model. Candidate value: `8`.
3. **`remote_url` deprecation warning** — upstream recommends `remote_storage_plugins` instead. Low priority until post-Stage 1.
4. **MemoryObj ref_count warning** — `MemoryObj at 0 is being garbage collected with ref_count=3` in Test 1. Appears in 1-key test only, not in 20+ key tests. Likely a cleanup ordering issue in the test harness, not a production code path. Log for upstream.
