# VaultAI Experimentation Plan — AMD AI MAX 395+

**Device:** AMD Ryzen AI MAX 395+ (Strix Halo), 128 GB unified memory, gfx1151  
**Goal:** Validate LMCache on this hardware, quantify TTFT gains, and close the gfx1151 gap that upstream does not cover.

Experiments are ordered by dependency — each stage unlocks the next.

---

## Stage 0 — Environment validation (no model, no vLLM)

Before touching inference, confirm the software stack is functional.

### 0-A: ROCm device visibility

```bash
rocm-smi
# Expected: shows Strix Halo iGPU, gfx1151, memory size

python -c "import torch; print(torch.version.hip); print(torch.cuda.get_device_name(0))"
# Expected: ROCm version string, device name like "AMD Radeon Graphics"
```

### 0-B: LMCache storage sanity (CPU mode, no GPU)

```bash
pip install -e . --no-build-isolation   # CPU-only, no BUILD_WITH_HIP

python -m lmcache.v1.basic_check --mode test_storage_manager
```

**Pass criteria:** No errors. KV tensors written and read back correctly.  
**Record:** Pass/fail, any warnings about device detection.

### 0-C: Unified memory bandwidth baseline

```bash
python - <<'EOF'
import torch, time
size = 4 * 1024**3  # 4 GB tensor
a = torch.randn(size // 4, dtype=torch.float32)
b = torch.empty_like(a)
t0 = time.perf_counter()
b.copy_(a)
t1 = time.perf_counter()
gb = size / 1e9
print(f"CPU->CPU copy: {gb/(t1-t0):.1f} GB/s")

a_gpu = a.cuda()
b_gpu = torch.empty_like(a_gpu)
torch.cuda.synchronize()
t0 = time.perf_counter()
b_gpu.copy_(a_gpu)
torch.cuda.synchronize()
t1 = time.perf_counter()
print(f"GPU->GPU copy: {gb/(t1-t0):.1f} GB/s")

t0 = time.perf_counter()
b.copy_(a_gpu)
t1 = time.perf_counter()
print(f"GPU->CPU copy: {gb/(t1-t0):.1f} GB/s")
EOF
```

**Expected:** All three should be similar (~200–256 GB/s) — confirms unified memory, no PCIe penalty.  
**Record:** All three bandwidth numbers.

---

## Stage 1 — vLLM ROCm baseline (no LMCache)

Establish raw inference performance before adding the cache layer.

### 1-A: vLLM ROCm install

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
pip install vllm --index-url https://download.pytorch.org/whl/rocm6.2

# Sanity check
python -c "import vllm; print(vllm.__version__)"
vllm serve Qwen/Qwen2.5-7B-Instruct --gpu-memory-utilization 0.5 --max-model-len 4096 --port 8000 &
sleep 30
curl -s http://localhost:8000/v1/models | python -m json.tool
```

### 1-B: Baseline TTFT — 7B model

```bash
python benchmarks/long_doc_qa/long_doc_qa.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --num-documents 4 \
  --document-length 4000 \
  --repeat-count 1 \
  --port 8000
```

**Record per run:**

| Metric | Value |
|---|---|
| Model | |
| Document length (tokens) | |
| TTFT ms | |
| Total latency ms | |
| Tokens/sec | |
| RAM used (GB) | |
| GPU util % (`rocm-smi`) | |
| Power draw (W) | |

### 1-C: Baseline TTFT — 35B model

Repeat 1-B with main model:

```bash
vllm serve <path-to-Qwen3.6-35B-A3B> \
  --gpu-memory-utilization 0.85 \
  --max-model-len 16384 \
  --port 8000
```

Test document lengths: 2K, 4K, 8K, 16K tokens.

---

## Stage 2 — LMCache CPU mode (current safe path)

### 2-A: Single-user TTFT gain

Start vLLM with LMCache CPU config:

```bash
LMCACHE_CONFIG_FILE=vaultai/config/lmcache_cpu.yaml \
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --gpu-memory-utilization 0.5 \
  --max-model-len 8192 \
  --port 8000
```

Run benchmark:

```bash
python benchmarks/long_doc_qa/long_doc_qa.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --num-documents 4 \
  --document-length 4000 \
  --repeat-count 3 \
  --port 8000
```

**Record:**

| Round | TTFT ms | Latency ms | Cache hit |
|---|---|---|---|
| 1 (cold miss) | | | No |
| 2 (hit) | | | Yes |
| 3 (hit) | | | Yes |
| **Speedup (R1/R2)** | | | |

**Target:** 3–8x TTFT reduction on cache hits.

### 2-B: Multi-user concurrency

```bash
python benchmarks/long_doc_qa/long_doc_qa.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --num-documents 4 \
  --document-length 4000 \
  --repeat-count 3 \
  --max-inflight-requests 5 \
  --port 8000
```

**Record:** TTFT and throughput at 1, 3, 5 concurrent users.  
**Watch for:** CPU thread contention between inference and KV compression — shows up as TTFT degradation scaling with concurrency.

### 2-C: Hit/miss ratio sweep

```bash
# 3 hits : 1 miss
python benchmarks/long_doc_qa/long_doc_qa.py ... --hit-miss-ratio 3:1

# 1 hit : 1 miss
python benchmarks/long_doc_qa/long_doc_qa.py ... --hit-miss-ratio 1:1
```

**Record:** Cache hit rate vs effective TTFT reduction.

### 2-D: Disk overflow behavior

Set `max_local_cpu_size: 2` (force early eviction to disk), repeat 2-A.  
**Record:** TTFT when KV comes from NVMe vs unified RAM. Quantifies the RAM vs disk tier gap.

---

## Stage 3 — HIP kernel build (gfx1151 validation)

**Prerequisite:** ROCm SDK 7.0 installed (`/opt/rocm` present, `hipcc` on PATH).  
**Risk:** May fail to compile on gfx1151 — document any errors verbatim.

```bash
export PYTORCH_ROCM_ARCH=gfx1151
BUILD_WITH_HIP=1 pip install -e . --no-build-isolation 2>&1 | tee vaultai/logs/hip_build_gfx1151.log
```

### 3-A: Verify HIP extension loaded

```python
import lmcache
import lmcache.c_ops as ops
print(type(ops))
# If HIP compiled successfully: <module 'lmcache.c_ops' from '...'>
# If fallback used:             <module 'lmcache.non_cuda_equivalents' from '...'>
```

### 3-B: KV transfer microbenchmark — HIP vs Python fallback

```bash
python benchmarks/microbenchmark/<relevant script> \
  --backend hip \
  --compare-fallback
```

**Record:** Throughput (GB/s) for multi_layer_kv_transfer, HIP vs Python fallback.

### 3-C: Full LMCache benchmark with HIP kernels

Repeat Stage 2-A with HIP build active.  
**Record:** TTFT gain vs Stage 2-A (Python fallback). Quantify if HIP compilation was worth the effort.

---

## Stage 4 — Upstream PR validation (as PRs merge)

Track and test each PR against gfx1151 as it lands in upstream.

### 4-A: PR #3168 / #3091 — Device abstraction

After pulling the merged commit:

```bash
# Switch remote_serde from "naive" to "cachegen" (previously broken on AMD)
# Update vaultai/config/lmcache_cachegen.yaml
remote_serde: "cachegen"
```

Repeat Stage 2-A. **Expected:** cachegen compression now works on AMD, higher compression ratio, smaller disk cache.  
**Record:** Compression ratio naive vs cachegen, TTFT with cachegen enabled.

### 4-B: PR #3115 — hipFile GDS

After pulling:

```bash
# Test GDS backend on unified memory
python -m lmcache.v1.basic_check --mode test_storage_manager \
  --backend gds --gds-backend hipfile
```

**Open question:** On unified memory APU, there is no separate NVMe-to-GPU DMA path. Document what hipFile does on Strix Halo — does it fall back gracefully or error?

### 4-C: PR #3092 — CacheBlend Triton on gfx1151

After pulling:

```bash
# Run CacheBlend e2e test
cd examples/blend_kv_v1
python blend_kv.py --model Qwen/Qwen2.5-7B-Instruct
```

**Record:** Cold vs blend TTFT. Note any gfx1151-specific errors.  
If it works, report back to upstream PR as gfx1151 validation.

### 4-D: PR #3101 — Custom ROCm Dockerfile for gfx1151

Fork the Dockerfile from PR #3101:

```dockerfile
# Change in Dockerfile.rocm:
ARG ROCM_ARCH=gfx1151   # was gfx942,gfx950
```

Build and verify LMCache imports correctly inside container.

---

## Stage 5 — Production workload simulation

Simulate the actual VaultAI Business Node workload: repeated company documents, multi-user, mixed hit/miss.

### 5-A: RAG document corpus test

Build a test corpus of 20 documents (contracts, policy docs, technical specs) averaging 8K tokens each.  
Query each document 5 times across 3 simulated users.  
**Record:** Cache hit rate, average TTFT, p95 TTFT, throughput.

### 5-B: Thermal + power under sustained load

Run Stage 2-B (5 concurrent users) for 30 minutes continuously.

```bash
# In separate terminal
watch -n 2 rocm-smi --showtemp --showpower
```

**Record:** Temperature plateau, power draw, any thermal throttle events.  
**Threshold:** Throttle above 95°C sustained is a product risk.

### 5-C: Memory pressure test

Load 35B model + 20 GB LMCache RAM + active RAG embeddings simultaneously.  
**Record:** Total unified memory used, OOM behavior if any, fallback-to-disk trigger point.

---

## Results log

All benchmark results go in `vaultai/logs/`. Filename format:

```
YYYY-MM-DD_stageX_description.txt
```

Example: `2026-05-10_stage2a_single_user_ttft_7b.txt`

Each log should include:
- Date and hardware state (ROCm version, driver version, system temps before run)
- Exact command run
- Full output
- Summary metrics extracted at the bottom

---

## Decision gates

| Gate | Condition | Decision |
|---|---|---|
| Ship CPU mode | Stage 2-A shows ≥3x TTFT gain | Include LMCache in Business Node v1 |
| Enable HIP kernels | Stage 3-C shows measurable gain AND no build failures | Switch default to HIP build |
| Enable cachegen serde | Stage 4-A passes after PR #3091/#3168 merge | Switch `remote_serde` to `cachegen` |
| Ship CacheBlend | Stage 4-C passes on gfx1151 | Enable CacheBlend feature in VaultAI |
| Report upstream | Any stage reveals gfx1151-specific behavior | Open issue / comment on relevant PR |
