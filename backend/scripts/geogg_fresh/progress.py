"""Progress bars that behave in the VS Code / Anaconda Prompt terminal.

Why this module exists
----------------------
Every stage loops over wells, files or millions of SEG-Y traces.  You asked to
*see* progress for every stage, so all loops go through :func:`progress`.

* If ``tqdm`` is installed (your ``torch_cuda12.8`` env has tqdm 4.67.1) we use it.
* If not, a tiny built-in bar is used, so a missing cosmetic package can never
  crash a 30-minute SEG-Y scan.

Log messages are routed through :func:`write` so they print *above* the bar
instead of breaking it into pieces.
"""

from __future__ import annotations

import logging
import sys
import time

try:  # pragma: no cover - depends on the environment
    from tqdm.auto import tqdm as _tqdm

    HAVE_TQDM = True
except Exception:  # pragma: no cover
    _tqdm = None
    HAVE_TQDM = False


class _SimpleBar:
    """Minimal stand-in for ``tqdm`` (only the features this package uses)."""

    def __init__(self, iterable=None, total=None, desc="", unit="it", leave=True,
                 disable=False, unit_scale=False, **_ignored):
        self.iterable = iterable
        if total is None and iterable is not None and hasattr(iterable, "__len__"):
            total = len(iterable)
        self.total, self.desc, self.unit = total, desc, unit
        self.leave, self.disable, self.unit_scale = leave, disable, unit_scale
        self.n, self.postfix = 0, ""
        self._t0 = self._last = time.time()
        self._tty = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()

    def __iter__(self):
        for item in self.iterable:
            yield item
            self.update(1)
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _fmt(self, value):
        if not self.unit_scale:
            return f"{value:,.0f}"
        for suffix in ("", "k", "M", "G", "T"):
            if abs(value) < 1000:
                return f"{value:.1f}{suffix}"
            value /= 1000.0
        return f"{value:.1f}P"

    def _render(self, final=False):
        if self.disable:
            return
        elapsed = time.time() - self._t0
        if self.total:
            frac = min(1.0, self.n / self.total)
            bar = "#" * int(25 * frac) + "-" * (25 - int(25 * frac))
            text = (f"{self.desc}: {100 * frac:5.1f}%|{bar}| {self._fmt(self.n)}/"
                    f"{self._fmt(self.total)} {self.unit} [{elapsed:6.1f}s] {self.postfix}")
        else:
            text = f"{self.desc}: {self._fmt(self.n)} {self.unit} [{elapsed:6.1f}s] {self.postfix}"
        if self._tty:
            sys.stderr.write("\r" + text[:160].ljust(160))
            if final:
                sys.stderr.write("\n" if self.leave else "\r")
            sys.stderr.flush()
        elif final and self.leave:
            sys.stderr.write(text + "\n")

    def update(self, n=1):
        self.n += n
        now = time.time()
        if now - self._last > 0.25:
            self._last = now
            self._render()

    def set_description(self, desc, refresh=True):
        self.desc = desc

    def set_postfix_str(self, text, refresh=True):
        self.postfix = text

    def close(self):
        if not getattr(self, "_closed", False):
            self._closed = True
            self._render(final=True)

    @staticmethod
    def write(message, file=None):
        print(message, file=file or sys.stderr)


def progress(iterable=None, total=None, desc="", unit="it", leave=True, **kwargs):
    """Return a progress bar (tqdm if available).  Use exactly like ``tqdm``."""
    if HAVE_TQDM:
        return _tqdm(iterable=iterable, total=total, desc=desc, unit=unit, leave=leave,
                     dynamic_ncols=True, **kwargs)
    return _SimpleBar(iterable=iterable, total=total, desc=desc, unit=unit, leave=leave, **kwargs)


def write(message: str) -> None:
    """Print a line without corrupting an active progress bar."""
    if HAVE_TQDM:
        _tqdm.write(message, file=sys.stderr)
    else:
        _SimpleBar.write(message)


class TqdmLoggingHandler(logging.Handler):
    """Logging handler that prints through :func:`write` (bars stay intact)."""

    def emit(self, record):
        try:
            write(self.format(record))
        except Exception:  # pragma: no cover - logging must never raise
            self.handleError(record)


class StepTracker:
    """A coarse 'step k of N' bar for the major steps inside one stage.

    Example::

        steps = StepTracker(["Read header", "Scan LAS", "Make figures"], "00_inventory")
        steps.next("Read header")   # bar shows: 00_inventory [Read header]  1/3
        ...
        steps.close()
    """

    def __init__(self, step_names, stage_name):
        self.names = list(step_names)
        self.stage_name = stage_name
        self.bar = progress(total=len(self.names), desc=f"{stage_name}", unit="step", leave=True)
        self._started = False

    def next(self, label: str) -> None:
        if self._started:
            self.bar.update(1)
        self._started = True
        self.bar.set_description(f"{self.stage_name} [{label}]")

    def close(self) -> None:
        if self._started:
            self.bar.update(1)
        self.bar.close()
