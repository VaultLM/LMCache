"""
Disk-tier benchmark for LMCache on AMD AI MAX 395+.

Architecture note: LMCache writes to ALL backends simultaneously on PUT
(LocalCPUBackend + RemoteBackend/disk). GET checks RAM first, then disk.
To measure pure disk reads, we close the storage manager after PUT (dropping
the RAM cache), reopen it (cold cache), and GET — forcing all reads from NVMe.

Suites:
  A — Object size sweep:   2 KB → 4 MB, fixed 10 keys
  B — Batch size sweep:    1 → 50 keys, fixed 128 KB objects
  C — Settle time sweep:   how long until disk writes are readable
  D — Large-batch stress:  1 MB objects, up to 20 keys

Usage:
    LMCACHE_CONFIG_FILE=vaultai/config/lmcache_disk_bench.yaml \\
    vaultai/vllm/bin/python vaultai/benchmarks/disk_tier_bench.py
"""

# Standard
import hashlib
import os
import shutil
import time

# Third Party
import torch

# First Party
from lmcache.integration.vllm.utils import lmcache_get_or_create_config
from lmcache.utils import CacheEngineKey
from lmcache.v1.event_manager import EventManager
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.remote_backend import RemoteBackend
from lmcache.v1.storage_backend.storage_manager import StorageManager

DTYPE = torch.bfloat16
DTYPE_BYTES = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _metadata(obj_size: int) -> LMCacheMetadata:
    half = obj_size // 2
    return LMCacheMetadata(
        model_name="/vaultai_disk_bench/",
        world_size=1, local_world_size=1,
        worker_id=0, local_worker_id=0,
        kv_dtype=DTYPE,
        kv_shape=(1, 2, half, 1, 1),
    )


def _key(idx: int) -> CacheEngineKey:
    h = int(hashlib.sha256(f"bench_{idx}".encode()).hexdigest(), 16)
    return CacheEngineKey(
        model_name="/vaultai_disk_bench/",
        world_size=1, worker_id=0,
        chunk_hash=h, dtype=DTYPE,
    )


def _make_sm(obj_size: int) -> StorageManager:
    return StorageManager(
        config=lmcache_get_or_create_config(),
        metadata=_metadata(obj_size),
        event_manager=EventManager(),
    )


def _alloc(sm: StorageManager, obj_size: int, fill: float) -> MemoryObj:
    half = obj_size // 2
    shape = torch.Size([2, 1, half, 1])
    obj = sm.allocate(shape, DTYPE, fmt=MemoryFormat.KV_2LTD, eviction=True, busy_loop=True)
    if obj is not None and obj.tensor is not None:
        obj.tensor.fill_(fill)
        obj.ref_count_up()
    return obj


def _wait_disk(sm: StorageManager, timeout: float = 30.0):
    """Block until RemoteBackend has no pending put tasks."""
    rb = next((b for b in sm.storage_backends.values() if isinstance(b, RemoteBackend)), None)
    if rb is None:
        return
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if not rb.put_tasks:
            return
        time.sleep(0.01)


def _throughput(total_bytes: int, elapsed_s: float) -> str:
    if elapsed_s <= 0:
        return "N/A"
    bps = total_bytes / elapsed_s
    if bps >= 1 << 30:
        return f"{bps / (1 << 30):.2f} GB/s"
    return f"{bps / (1 << 20):.2f} MB/s"


def _disk_files_exist(disk_dir: str) -> int:
    """Count KV cache files written to disk."""
    p = disk_dir.replace("fs://localhost:0", "").split("/var/")[0]
    try:
        return sum(1 for _ in __import__("pathlib").Path(disk_dir.split("://localhost:0")[-1]).rglob("*") if _.is_file())
    except Exception:
        return -1


# ---------------------------------------------------------------------------
# Core: PUT (writes RAM + disk) then GET from cold cache (disk only)
# ---------------------------------------------------------------------------

def run_scenario(
    label: str,
    obj_size: int,
    num_keys: int,
    settle_s: float = 1.0,
    disk_dir: str = "/tmp/vaultai_disk_bench",
) -> dict:
    total_bytes = obj_size * DTYPE_BYTES * num_keys
    keys = [_key(i) for i in range(num_keys)]

    # -- Phase 1: PUT (writes to RAM + disk simultaneously) --
    sm_put = _make_sm(obj_size)
    objs = [_alloc(sm_put, obj_size, float(i + 1)) for i in range(num_keys)]

    t_put0 = time.perf_counter()
    sm_put.batched_put(keys, objs)
    _wait_disk(sm_put, timeout=60.0)
    put_elapsed = time.perf_counter() - t_put0

    # Clone tensors now — sm_put.close() frees the memory pool backing obj.tensor,
    # making any later access a use-after-free / SIGSEGV.
    ref_tensors = [
        o.tensor.clone() if (o is not None and o.tensor is not None) else None
        for o in objs
    ]
    for o in objs:
        if o is not None:
            o.ref_count_down()

    sm_put.close()  # drop RAM cache; disk copy persists

    if settle_s > 0:
        time.sleep(settle_s)

    # -- Phase 2: GET from cold cache (disk only) --
    sm_get = _make_sm(obj_size)  # fresh SM, empty RAM

    get_times = []
    got_tensors = []
    none_count = 0
    for key in keys:
        t0 = time.perf_counter()
        result = sm_get.get(key)
        get_times.append(time.perf_counter() - t0)
        # Clone before sm_get.close() frees the memory pool backing result.tensor.
        if result is not None and result.tensor is not None:
            got_tensors.append(result.tensor.clone())
            result.ref_count_down()
        else:
            got_tensors.append(None)
            none_count += 1

    sm_get.close()

    # -- Verify content (all tensors cloned before SM teardown) --
    content_ok = sum(
        1 for got, ref in zip(got_tensors, ref_tensors)
        if got is not None and ref is not None and torch.equal(got, ref)
    )

    get_elapsed = sum(get_times)

    return {
        "label": label,
        "obj_kb": round(obj_size * DTYPE_BYTES / 1024, 1),
        "num_keys": num_keys,
        "total_mb": round(total_bytes / (1 << 20), 1),
        "put_elapsed_s": put_elapsed,
        "put_tp": _throughput(total_bytes, put_elapsed),
        "get_avg_ms": (get_elapsed / num_keys) * 1000,
        "get_max_ms": max(get_times) * 1000,
        "get_min_ms": min(get_times) * 1000,
        "get_tp": _throughput(total_bytes, get_elapsed),
        "content_ok": content_ok,
        "disk_miss": none_count,
        "num_keys": num_keys,
    }


def _hdr():
    print(f"  {'Scenario':<36} {'ObjKB':>8} {'Keys':>5} {'TotalMB':>8} "
          f"{'PUT':>13} {'GET avg ms':>12} {'GET total':>13} {'Content':>9} {'DiskMiss':>9}")
    print("  " + "-" * 115)


def _row(r: dict):
    content_str = f"{r['content_ok']}/{r['num_keys']}"
    print(
        f"  {r['label']:<36} {r['obj_kb']:>8} {r['num_keys']:>5} {r['total_mb']:>8} "
        f"  {r['put_tp']:>11} {r['get_avg_ms']:>12.2f} {r['get_tp']:>13} "
        f"  {content_str:>8} {r['disk_miss']:>8}"
    )


# ---------------------------------------------------------------------------
# Suites
# ---------------------------------------------------------------------------

def suite_object_size(settle_s: float = 1.0):
    print(f"\n=== Suite A: Object size sweep (10 keys, settle={settle_s}s) ===")
    _hdr()
    for label, obj_size in [
        ("2 KB",    1024),
        ("16 KB",   8192),
        ("64 KB",   32768),
        ("128 KB",  65536),
        ("512 KB",  262144),
        ("1 MB",    524288),
        ("4 MB",    2097152),
    ]:
        r = run_scenario(f"obj={label}", obj_size, num_keys=10, settle_s=settle_s)
        _row(r)


def suite_batch_size(settle_s: float = 1.0):
    print(f"\n=== Suite B: Batch size sweep (128 KB objects, settle={settle_s}s) ===")
    _hdr()
    OBJ = 65536  # 128 KB
    for n in [1, 5, 10, 20, 50]:
        r = run_scenario(f"batch={n} keys", OBJ, num_keys=n, settle_s=settle_s)
        _row(r)


def suite_settle_time():
    print("\n=== Suite C: Settle time (128 KB objects, 10 keys) — find flush latency ===")
    _hdr()
    OBJ = 65536
    for settle in [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]:
        r = run_scenario(f"settle={settle}s", OBJ, num_keys=10, settle_s=settle)
        _row(r)


def suite_large(settle_s: float = 2.0):
    print(f"\n=== Suite D: Large objects / high volume (settle={settle_s}s) ===")
    _hdr()
    for label, obj_size, n in [
        ("512 KB × 20 keys",  262144, 20),
        ("1 MB × 10 keys",    524288, 10),
        ("1 MB × 20 keys",    524288, 20),
        ("4 MB × 5 keys",    2097152,  5),
        ("4 MB × 10 keys",   2097152, 10),
    ]:
        r = run_scenario(label, obj_size, num_keys=n, settle_s=settle_s)
        _row(r)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = os.environ.get("LMCACHE_CONFIG_FILE", "")
    if not cfg:
        raise SystemExit("Set LMCACHE_CONFIG_FILE before running.")

    disk_dir = "/tmp/vaultai_disk_bench"
    shutil.rmtree(disk_dir, ignore_errors=True)
    os.makedirs(disk_dir, exist_ok=True)

    print("\nDisk-tier benchmark — LMCache on AMD AI MAX 395+")
    print(f"Config : {cfg}")
    print(f"NVMe   : Lexar NQ790 2TB (/dev/nvme0n1)")
    print(f"Method : PUT writes RAM+disk; SM closed; GET reads cold disk only")
    print(f"dtype  : bfloat16 (2 bytes/element)")

    suite_object_size(settle_s=1.0)
    suite_batch_size(settle_s=1.0)
    suite_settle_time()
    suite_large(settle_s=2.0)

    shutil.rmtree(disk_dir, ignore_errors=True)
    print("\nDone.")
