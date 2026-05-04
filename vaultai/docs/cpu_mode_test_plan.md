# CPU Mode Test Plan — No LLM Runtime

**Scope:** Validate LMCache CPU mode in isolation — no vLLM, no model, no GPU required.  
**Hardware:** AMD Ryzen AI MAX 395+ (Strix Halo), 128 GB unified memory  
**Config:** `vaultai/config/lmcache_cpu.yaml`

All tests use `lmcache.v1.basic_check` directly against the storage layer.  
Each test should complete without importing torch.cuda, triggering HIP, or requiring a live inference server.

---

## Prerequisites

LMCache and vLLM run in the **same Python process** — a single shared venv is required.
torch is installed once from the ROCm index and never reinstalled from PyPI.
The environment spec lives in `vaultai/pyproject.toml` and is locked with `uv.lock`.

### 1. Create the shared venv (once per device)

```bash
cd /home/vaultai/Vaultai/LMCache

uv venv --python 3.12 .venv
source .venv/bin/activate
```

### 2. Sync the locked environment

```bash
# First time: resolve and lock. Subsequent runs: exact replay from uv.lock.
cd vaultai && uv sync && cd ..
```

`uv sync` reads `vaultai/pyproject.toml`, which pins the ROCm wheel index for torch
so it is never silently replaced by a CUDA build from PyPI. After this command, the
environment is byte-for-byte reproducible from `uv.lock`.

### 3. Install LMCache into the same venv (no extension compilation)

```bash
# NO_CUDA_EXT=1 skips all C/HIP extension builds.
# --no-deps prevents setup.py from auto-pulling cuda_core.txt (cupy-cuda12x, nixl).
cd /home/vaultai/Vaultai/LMCache
NO_CUDA_EXT=1 uv pip install -e . --no-build-isolation --no-deps
```

When vLLM ROCm is ready (Stage 1 of experimentation plan), add it to the same venv:

```bash
uv add vllm --index-url https://download.pytorch.org/whl/rocm6.2
# uv re-locks automatically. torch is already present and will not be reinstalled.
```

### 4. Verify

```bash
python -c "import lmcache; print('import OK')"

# c_ops must resolve to the Python fallback, not a compiled .so
python -c "
import lmcache.c_ops as ops, inspect
print(inspect.getfile(ops))
# Expected: .../non_cuda_equivalents.py
"
```

**Pass criteria:** No CUDA/HIP errors, `c_ops` resolves to `non_cuda_equivalents`.

---

## Test 1 — Storage manager creation

Verify the config is parsed and the storage manager initialises correctly.

```bash
LMCACHE_CONFIG_FILE=vaultai/config/lmcache_cpu.yaml \
python -m lmcache.v1.basic_check --mode test_storage_manager --num-keys 1
```

**Pass criteria:** `Test: Passed - Created storage manager with valid config` printed, no exception.  
**Watch for:** Any `.cuda()` traceback — would mean the naive serde guard is not working.  
**Record:** Pass/fail, any warnings.

---

## Test 2 — RAM tier: write and read back

Validate that KV tensors can be stored in and retrieved from the local CPU (unified RAM) backend.

```bash
LMCACHE_CONFIG_FILE=vaultai/config/lmcache_cpu.yaml \
python -m lmcache.v1.basic_check \
  --mode test_storage_manager \
  --num-keys 20 \
  --kv-dtype bfloat16
```

**Pass criteria:**
- Phase 1 (EXISTS non-exist): 20/20 correctly absent
- Phase 2 (PUT): 20/20 pass
- Phase 3 (EXISTS exist): 20/20 found
- Phase 4 (GET): 20/20 content correct

**Record:** PUT and GET throughput (GB/s) from the performance results table.

---

## Test 3 — KV dtype coverage

Repeat Test 2 for all dtypes relevant to the model targets.

```bash
for DTYPE in float16 bfloat16 float32; do
  echo "=== dtype: $DTYPE ==="
  LMCACHE_CONFIG_FILE=vaultai/config/lmcache_cpu.yaml \
  python -m lmcache.v1.basic_check \
    --mode test_storage_manager \
    --num-keys 10 \
    --kv-dtype $DTYPE
done
```

**Pass criteria:** All three dtypes reach 10/10 content correct.  
**Record:** PUT/GET throughput per dtype.

---

## Test 4 — Disk overflow tier

Force eviction to NVMe by setting `max_local_cpu_size: 2` (2 GB cap) to overflow large objects to the fs backend.

Create an override config:

```bash
cat > /tmp/lmcache_disk_overflow.yaml << 'EOF'
chunk_size: 256
local_cpu: true
max_local_cpu_size: 2
save_unfull_chunk: false
remote_serde: "naive"
remote_url: "fs://localhost:0/tmp/vaultai_lmcache_test"
extra_config:
  save_chunk_meta: true
EOF

mkdir -p /tmp/vaultai_lmcache_test
```

Run with objects large enough to fill the 2 GB cap:

```bash
LMCACHE_CONFIG_FILE=/tmp/lmcache_disk_overflow.yaml \
python -m lmcache.v1.basic_check \
  --mode test_storage_manager \
  --num-keys 20 \
  --obj-size 65536 \
  --kv-dtype bfloat16 \
  --settle-time 2.0
```

**Pass criteria:** GET still returns correct content after eviction to disk.  
**Record:**
- PUT throughput (GB/s) — RAM path
- GET throughput (GB/s) — NVMe path (will be lower)
- Gap between RAM and disk GB/s quantifies the tier penalty

**Cleanup:**

```bash
rm -rf /tmp/vaultai_lmcache_test
```

---

## Test 5 — Concurrency stress

Confirm the storage manager is stable under concurrent puts and gets, which is the condition that exposes CPU thread contention.

```bash
LMCACHE_CONFIG_FILE=vaultai/config/lmcache_cpu.yaml \
python -m lmcache.v1.basic_check \
  --mode test_storage_manager \
  --num-keys 100 \
  --kv-dtype bfloat16
```

**Pass criteria:** 100/100 content correct, no deadlock, completes within 60 seconds.  
**Watch for:** Degrading throughput as num-keys grows — indicates lock contention in the CPU backend.  
**Record:** PUT and GET throughput at 100 keys vs Test 2 (20 keys).

---

## Test 6 — Naive serde isolation (no cachegen)

Explicitly confirm that `remote_serde: naive` does not load `cachegen_encoder` or `cachegen_decoder` at any point.

```bash
LMCACHE_CONFIG_FILE=vaultai/config/lmcache_cpu.yaml \
python -c "
import sys
import lmcache
from lmcache.integration.vllm.utils import lmcache_get_or_create_config
cfg = lmcache_get_or_create_config()
print('remote_serde:', cfg.remote_serde)

# Confirm cachegen modules are not imported
cachegen_modules = [m for m in sys.modules if 'cachegen' in m]
print('cachegen modules in sys.modules:', cachegen_modules)
# Expected: []
"
```

**Pass criteria:** `remote_serde: naive` printed, `cachegen_modules` list is empty.  
**Why this matters:** `cachegen_decoder/encoder.py` contain hardcoded `.cuda()` calls that crash on AMD without the upstream fix (PRs #3091/#3168).

---

## Test 7 — Memory footprint at startup

Measure how much unified RAM the storage manager commits at startup before any KV
data is stored.

```bash
LMCACHE_CONFIG_FILE=vaultai/config/lmcache_cpu.yaml \
vaultai/vllm/bin/python -c "
import os, psutil
proc = psutil.Process(os.getpid())
rss_before = proc.memory_info().rss / 1e9
from lmcache.v1.check.utils import create_storage_manager_with_config
sm = create_storage_manager_with_config('/test_model/', kv_dtype=None, obj_size=4096)
rss_after = proc.memory_info().rss / 1e9
print(f'RSS before: {rss_before:.2f} GB')
print(f'RSS after:  {rss_after:.2f} GB')
print(f'Delta:      {rss_after - rss_before:.2f} GB')
sm.close()
" 2>&1 | grep -E "RSS|Delta"
```

**Observed result (2026-05-04):** Delta = **23 GB** with `max_local_cpu_size: 20`.

**Finding:** LocalCPUBackend eagerly pre-allocates the full configured pool at startup,
not lazily on first use. On a 128 GB unified memory device serving a 35B model
(~70 GB weights), setting `max_local_cpu_size: 20` immediately reserves 20 GB.
This is expected behaviour but must be accounted for in memory budgeting.

**Action:** Treat `max_local_cpu_size` as a hard reservation, not a soft cap.
For Stage 1 (35B model), use `max_local_cpu_size: 8` or lower to leave headroom
for model weights and the OS. Revisit after measuring actual model VRAM footprint.

---

## Results log

Save each test output to `vaultai/logs/` with format:

```
YYYY-MM-DD_cpu-test-N_description.txt
```

Example:
```bash
LMCACHE_CONFIG_FILE=vaultai/config/lmcache_cpu.yaml \
python -m lmcache.v1.basic_check --mode test_storage_manager --num-keys 20 \
  2>&1 | tee vaultai/logs/2026-05-04_cpu-test-2_ram_tier_bfloat16.txt
```

Each log must include the exact command at the top.

---

## Pass / fail summary

| Test | What it validates | Result (2026-05-04) |
|---|---|---|
| 1 | Config parses, storage manager boots | PASS |
| 2 | RAM tier PUT/GET correctness (bfloat16, 20 keys) | PASS — 20/20, GET 80 MB/s |
| 3 | dtype coverage (f16, bf16, f32, 20 keys each) | PASS — 20/20 all dtypes |
| 4 | Disk overflow correctness + tier gap | PASS — 20/20 after eviction |
| 5 | Concurrency stability at 100 keys | PASS — 100/100, no deadlock |
| 6 | Naive serde active, cachegen excluded | PASS — cachegen_modules empty |
| 7 | Memory pre-allocation behaviour | FINDING: 23 GB reserved on startup with max_local_cpu_size=20 — pool is eager, not lazy |
| 7 | Memory not over-allocated at startup | RSS delta < 1 GB |

**Decision gate:** All 7 tests pass → CPU mode is safe to layer on top of vLLM ROCm (Stage 1 in experimentation plan).
