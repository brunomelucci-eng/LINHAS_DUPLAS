"""
logging_utils.py — Logging setup and stage monitoring utilities.

New additions
-------------
* StageTimer  : context manager that logs elapsed time, process RSS, available
                RAM and free disk space at the end of each pipeline stage.
                Prevents the user from thinking the process has frozen.
* configure_runtime_threads : apply max_cpu_threads to PyTorch, OpenCV and set
                GDAL environment variables at process startup.
"""

import logging
import os
import sys
import time
import threading
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(
    output_dir: str = "outputs",
    log_level: int = logging.INFO,
    run_id: Optional[str] = None,
    pid: Optional[int] = None,
) -> None:
    """
    Configure application logging to console and rotating file.

    If `run_id` and `pid` are provided, a filter is installed so every record
    automatically includes `[run=<id> pid=<pid>]` in its message prefix.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "logs", "execution.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    root = logging.getLogger()
    root.setLevel(log_level)

    for handler in list(root.handlers):
        root.removeHandler(handler)

    fmt_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    if run_id and pid:
        fmt_str = f'%(asctime)s [run={run_id} pid={pid}] - %(name)s - %(levelname)s - %(message)s'

    formatter = logging.Formatter(fmt_str)

    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setLevel(log_level)
    c_handler.setFormatter(formatter)
    root.addHandler(c_handler)

    f_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    f_handler.setLevel(log_level)
    f_handler.setFormatter(formatter)
    root.addHandler(f_handler)

    # Silence verbose 3rd-party loggers
    for noisy in ('rasterio', 'fiona', 'matplotlib', 'pyogrio'):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger.info("Logging initialised. Output dir: %s. Log file: %s", output_dir, log_path)


# ---------------------------------------------------------------------------
# Runtime thread / resource configuration
# ---------------------------------------------------------------------------

def configure_runtime_threads(config: dict) -> None:
    """
    Apply CPU-thread limits to PyTorch, OpenCV and GDAL from the `runtime`
    section of the config.  Call once at process startup before any inference
    or raster I/O.
    """
    runtime_cfg = config.get('runtime', {})
    max_threads  = int(runtime_cfg.get('max_cpu_threads', 4))
    gdal_threads = int(runtime_cfg.get('gdal_threads',    2))

    # OpenCV
    try:
        import cv2
        cv2.setNumThreads(max_threads)
        logger.info("OpenCV threads set to %d.", max_threads)
    except ImportError:
        pass

    # PyTorch
    try:
        import torch
        if not torch.cuda.is_available():
            torch.set_num_threads(max_threads)
            torch.set_num_interop_threads(1)
            logger.info("PyTorch CPU threads set to %d (interop=1).", max_threads)
    except ImportError:
        pass

    # GDAL environment (used by rasterio internally)
    os.environ.setdefault('GDAL_NUM_THREADS', str(gdal_threads))
    os.environ.setdefault('GDAL_CACHEMAX',    str(runtime_cfg.get('gdal_cache_mb', 256)))
    logger.info("GDAL_NUM_THREADS=%s GDAL_CACHEMAX=%s.", os.environ['GDAL_NUM_THREADS'], os.environ['GDAL_CACHEMAX'])


# ---------------------------------------------------------------------------
# StageTimer — context manager for pipeline stage monitoring
# ---------------------------------------------------------------------------

def _get_process_rss_mb() -> float:
    """Return current process RSS in MiB (cross-platform, best-effort)."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except Exception:
        pass
    try:
        # Linux /proc fallback
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return float('nan')


def _get_available_ram_mb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / 1024 / 1024
    except Exception:
        return float('nan')


def _get_free_disk_gb(path: str = '.') -> float:
    try:
        import shutil
        total, used, free = shutil.disk_usage(path)
        return free / 1024 ** 3
    except Exception:
        return float('nan')


@contextmanager
def StageTimer(
    stage_name: str,
    log_interval_s: float = 60.0,
    disk_path: str = '.',
):
    """
    Context manager that:
    1. Logs the stage start.
    2. Periodically logs a "still running" message (default every 60 s) so the
       user does not think the process is frozen.
    3. On exit, logs elapsed time, RSS, available RAM and free disk space.

    Usage
    -----
    with StageTimer('mask_cleaning'):
        clean_mask(...)

    Log format
    ----------
    stage=mask_cleaning elapsed=42.3s rss=2134MB available_ram=7891MB disk_free=87.3GB
    """
    t0 = time.monotonic()
    logger.info("stage=%s START", stage_name)

    stop_event = threading.Event()

    def _heartbeat():
        while not stop_event.wait(timeout=log_interval_s):
            elapsed = time.monotonic() - t0
            rss     = _get_process_rss_mb()
            avail   = _get_available_ram_mb()
            logger.info(
                "stage=%s RUNNING elapsed=%.1fs rss=%.0fMB available_ram=%.0fMB",
                stage_name, elapsed, rss, avail,
            )

    hb = threading.Thread(target=_heartbeat, daemon=True)
    hb.start()

    try:
        yield
    finally:
        stop_event.set()
        hb.join(timeout=2.0)

        elapsed  = time.monotonic() - t0
        rss      = _get_process_rss_mb()
        avail    = _get_available_ram_mb()
        disk_gb  = _get_free_disk_gb(disk_path)

        logger.info(
            "stage=%s DONE elapsed=%.1fs rss=%.0fMB available_ram=%.0fMB disk_free=%.1fGB",
            stage_name, elapsed, rss, avail, disk_gb,
        )
