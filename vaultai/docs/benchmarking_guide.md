# VaultAI Benchmarking Guide — AMD AI MAX 395+

---

## Prerequisites

```bash
# PyTorch ROCm
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2

# vLLM ROCm
pip install vllm --index-url https://download.pytorch.org/whl/rocm6.2

# LMCache (CPU mode, no HIP build)
pip install -e . --no-build-isolation

# Verify AMD GPU visible
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## Stage 1 — Storage sanity check (no model needed)

```bash
python -m lmcache.v1.basic_check --mode test_storage_manager
```

Pass = LMCache can write/read KV tensors correctly on this hardware.

---

## Stage 2 — vLLM + LMCache TTFT benchmark

Start vLLM with LMCache:

```bash
LMCACHE_CONFIG_FILE=vaultai/config/lmcache_cpu.yaml \
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --gpu-memory-utilization 0.7 \
  --max-model-len 8192 \
  --port 8000
```

Run long-doc QA benchmark:

```bash
python benchmarks/long_doc_qa/long_doc_qa.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --num-documents 4 \
  --document-length 4000 \
  --repeat-count 3 \
  --port 8000
```

### What to record

| Metric | Round 1 (miss) | Round 2 (hit) | Round 3 (hit) |
|---|---|---|---|
| TTFT (ms) | | | |
| Total latency (ms) | | | |
| Tokens/sec | | | |
| RAM used (GB) | | | |
| Disk cache size (GB) | | | |

Target: 3–8x TTFT reduction on cache hits.

---

## Stage 3 — Main model benchmark (35B)

Repeat Stage 2 with:

```bash
vllm serve <Qwen3.6-35B-A3B-path> \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768 \
  --port 8000
```

Also test:
- 3–5 concurrent users (`--max-inflight-requests 5`)
- Document lengths: 4K, 8K, 16K tokens
- Hit/miss ratio: `--hit-miss-ratio 3:1`

---

## Metrics to capture per run

```
tokens/sec
time-to-first-token (TTFT)
model load time
RAM usage (unified pool)
GPU utilization (radeontop or rocm-smi)
thermal throttle events
power draw (W)
cache hit rate
NVMe write throughput during cache store
```
