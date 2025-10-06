from pathlib import Path
import os
from platformdirs import user_cache_dir

APP_NAME = "nvrar"

# Default to OS cache dir: Linux ~/.cache/<APP_NAME>, macOS ~/Library/Caches/<APP_NAME>
_default_cache = Path(user_cache_dir(APP_NAME))

NVRAR_CACHE_DIR = Path(os.getenv("NVRAR_CACHE_DIR", _default_cache))
NVRAR_CACHE_DIR.mkdir(parents=True, exist_ok=True)