"""Run-log preservation: tee runner stdout to the result artifact.

Every benchmark/adaptation run writes its console log to
``<output_dir>/run_logs/<name>.log`` so training evidence (per-epoch
losses, gap-exclusion notes, fold summaries) is committed beside the
metrics it produced, instead of being lost in a transient ``logs/``
directory or a scrollback buffer.

Usage (inside a script's ``main()``)::

    from sleep_staging.utils.runlog import tee_run_log
    with tee_run_log(output_dir, "benchmark_seed42"):
        ...  # everything printed inside is captured
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO, Iterator


class _Tee:
    """Write to the original stream and a file simultaneously."""

    def __init__(self, original: TextIO, log_file: TextIO):
        self.original = original
        self.log_file = log_file

    def write(self, data: str) -> int:
        self.log_file.write(data)
        return self.original.write(data)

    def flush(self) -> None:
        self.log_file.flush()
        self.original.flush()

    def isatty(self) -> bool:
        return self.original.isatty()


@contextmanager
def tee_run_log(output_dir: str | Path, name: str) -> Iterator[Path]:
    """Tee stdout (and stderr into the same file) for the block's span.

    Args:
        output_dir: results directory; log goes to
            ``<output_dir>/run_logs/<name>.log``.
        name: log basename (conventionally includes the seed, e.g.
            ``"benchmark_seed42"``).

    Yields:
        The path of the log file being written.
    """
    log_dir = Path(output_dir) / "run_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"

    with open(log_path, "a") as log_file:
        old_stdout = sys.stdout
        sys.stdout = _Tee(old_stdout, log_file)
        try:
            yield log_path
        finally:
            sys.stdout = old_stdout
