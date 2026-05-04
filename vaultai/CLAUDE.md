# VaultAI — LMCache Integration Context

This directory contains all VaultAI-specific files. The upstream LMCache source is
never modified directly — patches are isolated here so upstream can be pulled cleanly.

---

## Hardware target

```
AMD Ryzen AI MAX 395+ (Strix Halo)
128 GB unified memory — no discrete GPU, no CUDA
ROCm 6.x / 7.x is the GPU compute path
```

## Product context

VaultAI is a plug-and-play private offline AI server appliance.
LMCache is the KV-cache acceleration layer sitting on top of vLLM.
Users never interact with LMCache directly — they see VaultAI Chat / Files / Agents.

## Runtime stack

```
VaultAI UI → VaultAI backend
  → Runtime adapter
    ├── llama.cpp / LM Studio (Vulkan)   ← primary today
    ├── vLLM ROCm                        ← production target
    └── LMCache on vLLM                  ← this integration, after vLLM ROCm validated
```

## LMCache deployment mode for this hardware

- `local_cpu: true` — KV tensors stay in unified RAM, no PCIe copy
- `remote_serde: naive` — avoids cachegen hardcoded `.cuda()` calls
- Disk overflow cap: 200–500 GB NVMe for Business Node
- Do NOT use `BUILD_WITH_HIP=1` build until validated on device
- Do NOT use TensorRT-LLM adapter (NVIDIA-only)

## Known upstream issues to patch before GPU-accelerated mode

See [docs/amd_compatibility.md](docs/amd_compatibility.md) for full details.

1. `lmcache/storage_backend/serde/cachegen_decoder.py` — hardcoded `.cuda()` calls
2. `lmcache/storage_backend/serde/cachegen_encoder.py` — hardcoded `.cuda()` calls
3. `lmcache/usage_context.py` — no AMD/ROCm device detection branch

## Model targets

| Role | Model | Format |
|---|---|---|
| Fast | Qwen2.5-7B / 9B | AWQ or GGUF Q4 |
| Main | Qwen3.6 35B A3B | AWQ / GPTQ / FP8 |

## Directory layout

```
vaultai/
├── CLAUDE.md          ← this file, loaded by Claude Code
├── config/            ← LMCache configs per deployment mode
├── docs/              ← AMD compatibility notes, benchmarking guide, decisions
└── benchmarks/        ← VaultAI-specific benchmark scripts and results
```
