import json, os
from pathlib import Path
from .config_paths import NVRAR_CACHE_DIR


_ENV_OVERRIDE = "NVSHMEM_ALLREDUCE_CONFIG"  # optional


def load_default_params(num_gpus: int) -> dict:
    return {"0": {
        "num_blocks": 8,
        "threads_per_block": 512,
        "chunk_bytes": 16384,
        "algorithm": "recursive",
        "dtype": None,
        "avg_time_ms": 0.0,
    }}

def signature_key(num_gpus: int, dtype: str) -> str:
    # Stable filename from a subset of fields
    return f"tuning_{num_gpus}gpu_{dtype}"


# TODO: Add dtype support
class LaunchParams:
    def __init__(self, table, source):
        self._table = table  # dict[str(msg_bytes)] -> dict
        self.source = source # path or label

    def for_message_bytes(self, nbytes: int) -> dict:
        # choose nearest bucket, or exact key
        if str(nbytes) in self._table:
            return self._table[str(nbytes)]
        keys = sorted(int(k) for k in self._table.keys())
        # nearest-lte, else nearest
        best = max((k for k in keys if k <= nbytes), default=None)
        if best is None:
            best = min(keys) if keys else None
        return self._table[str(best)] if best is not None else {}

def _load_json(path: Path) -> dict | None:
    try:
        print(f"Loading JSON from {path}")
        return json.loads(path.read_text())
    except Exception:
        return None

def resolve_params(num_gpus: int, dtype: str) -> LaunchParams:
    # 1) explicit override via env var (still no app code path passing)
    env = os.getenv(_ENV_OVERRIDE)
    if env:
        data = _load_json(Path(env))
        if data:
            return LaunchParams(data, source=f"env:{env}")

    # 2) per-machine tuned file
    key = signature_key(num_gpus, dtype)
    tuned = NVRAR_CACHE_DIR / f"{key}.json"
    data = _load_json(tuned)
    if data:
        return LaunchParams(data, source=str(tuned))

    # 3) Otherwise, package defaults
    return LaunchParams(load_default_params(num_gpus), source="package:defaults")