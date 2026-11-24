"""Real tSS-BO Bayesian-optimisation backend (BL-206).

Wraps the upstream tSS-BO library (github.com/PascalInn/tSS-BO, BSD-3, built on
torch / gpytorch / botorch) into a clean ``minimize(objective, bounds)`` callable
for Clay's sizing tune. tSS-BO is a CLOSED-LOOP solver (``main_solver(funct, ...)``
drives the loop, calling ``funct`` itself) rather than ask/tell, so this is the
correct integration shape for our ``evaluate``-callback tuner -- not the ask/tell
``TSsBoAdapter``.

The library is NOT pip-installable (a raw research repo). It is fetched into
``external/tSS-BO`` (git-ignored, reproducible -- see ``fetch_tssbo.py``), the same
way the PDKs are resolved rather than vendored. ``tssbo_available()`` reports
whether the clone + its deps are importable.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REFERENCE_TSSBO_DIR = _REPO_ROOT / "refProject" / "tSS-BO"
_FETCHED_TSSBO_DIR = _REPO_ROOT / "external" / "tSS-BO"
_PINNED_REVISION = "35d675a3b86a3d8cb539ef17f0916137e516511a"
_PINNED_SOURCE_TREE_SHA256 = (
    "8836b8f808f0f1d4d89dc11f71a453be2643d6005b5595dfbd57203a8a8a3622"
)
_TSSBO_DIR = (
    _REFERENCE_TSSBO_DIR
    if (_REFERENCE_TSSBO_DIR / "src" / "main.py").is_file()
    else _FETCHED_TSSBO_DIR
)


def tssbo_dir() -> Path:
    return _TSSBO_DIR


def tssbo_available() -> bool:
    """True iff the tSS-BO clone + its deps (torch/gpytorch/botorch) import."""
    if not (_TSSBO_DIR / "src" / "main.py").is_file():
        return False
    try:
        _require_pinned_source_tree()
    except RuntimeError:
        return False
    try:
        import torch  # noqa: F401
        import gpytorch  # noqa: F401
        import botorch  # noqa: F401
    except Exception:
        return False
    return True


def _source_tree_sha256(root: Path) -> str:
    """Hash every executable Python source path+content in stable order."""
    files = sorted(
        (root / "src").rglob("*.py"),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _require_pinned_source_tree() -> str:
    actual = _source_tree_sha256(_TSSBO_DIR)
    if actual != _PINNED_SOURCE_TREE_SHA256:
        raise RuntimeError(
            "tSS-BO source tree drift: "
            f"expected {_PINNED_SOURCE_TREE_SHA256}, measured {actual}"
        )
    return actual


def tssbo_provenance() -> dict[str, str]:
    """Bind a run to the exact ignored reference/fetched solver bytes."""
    main = _TSSBO_DIR / "src" / "main.py"
    if not main.is_file():
        raise RuntimeError(f"tSS-BO main.py is missing at {_TSSBO_DIR}")
    source_tree = _require_pinned_source_tree()
    return {
        "source": "refProject" if _TSSBO_DIR == _REFERENCE_TSSBO_DIR else "external",
        "path": str(_TSSBO_DIR.resolve()),
        "main_sha256": hashlib.sha256(main.read_bytes()).hexdigest(),
        "source_tree_sha256": source_tree,
        "pinned_revision": _PINNED_REVISION,
    }


def _load_main_solver():
    """Import the upstream ``main_solver`` (adds the clone to sys.path)."""
    if not (_TSSBO_DIR / "src" / "main.py").is_file():
        raise RuntimeError(
            f"tSS-BO not found at {_TSSBO_DIR}. Fetch it first:\n"
            f"  python tools/clay_optimize/fetch_tssbo.py\n"
            f"(clones github.com/PascalInn/tSS-BO, BSD-3, into external/)."
        )
    _require_pinned_source_tree()
    root = str(_TSSBO_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)
    from src.main import main_solver  # type: ignore

    return main_solver


def _load_main_solver_constrained():
    """Import upstream ``main_solver_constrained`` unchanged."""
    if not (_TSSBO_DIR / "src" / "main.py").is_file():
        raise RuntimeError(f"tSS-BO not found at {_TSSBO_DIR}")
    _require_pinned_source_tree()
    root = str(_TSSBO_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)
    from src.main import main_solver_constrained  # type: ignore

    return main_solver_constrained


@dataclass(frozen=True)
class ConstrainedResult:
    best_x: np.ndarray
    best_row: np.ndarray
    feasible: bool
    history_x: np.ndarray
    history_y: np.ndarray


class _ConstrainedBudgetComplete(Exception):
    """Stop an upstream batch after retaining only its measured budget prefix."""


def minimize_constrained(
    funct: Callable[[np.ndarray], np.ndarray],
    dim: int,
    *,
    n_max: int,
    outdim: int = 2,
    batch_size: int = 4,
    n_candidates: int = 80,
    n_resample: int = 5,
    sigma: float = 0.2,
    mu: float = 0.5,
    seed: int = 0,
    quiet: bool = True,
    init_x: np.ndarray | None = None,
    init_y: np.ndarray | None = None,
) -> ConstrainedResult:
    """Run the real constrained tSS-BO callback contract.

    Inputs are always normalized to ``[-0.5, 0.5]^dim``.  Callback rows are
    ``[objective, g1, ...]`` with every constraint feasible iff ``g <= 0``.
    The returned incumbent is feasible-only; if no feasible sample exists the
    least-violation diagnostic point is returned with ``feasible=False``.
    ``n_max`` counts all measured rows, including supplied seeds and the
    upstream seed-mean evaluation. A final partial batch measures only the
    remaining prefix; no fabricated rows are returned to the solver.
    """
    import torch

    if dim <= 0 or n_max <= 0 or outdim < 2 or batch_size <= 0:
        raise ValueError(
            "constrained tSS-BO requires dim>0, n_max>0, outdim>=2, batch_size>0"
        )
    solver = _load_main_solver_constrained()
    bounds = torch.tensor([[-0.5, 0.5]] * int(dim), dtype=torch.float32)
    history_x: list[np.ndarray] = []
    history_y: list[np.ndarray] = []

    initial_x = None
    initial_y = None
    if init_x is not None or init_y is not None:
        if init_x is None or init_y is None:
            raise ValueError("constrained seed requires both init_x and init_y")
        initial_x = np.asarray(init_x, dtype=np.float32).reshape(-1, dim)
        initial_y = np.asarray(init_y, dtype=np.float32)
        if initial_y.ndim == 1:
            initial_y = initial_y.reshape(1, -1)
        if initial_y.shape != (len(initial_x), outdim):
            raise ValueError("constrained init_x/init_y shapes do not match")
        if not 1 < len(initial_x) < int(n_max):
            raise ValueError("constrained seed must contain 2..n_max-1 rows")
        if (
            not np.all(np.isfinite(initial_x))
            or not np.all(np.isfinite(initial_y))
            or np.any(initial_x < -0.5 - 1e-6)
            or np.any(initial_x > 0.5 + 1e-6)
        ):
            raise ValueError("constrained seed is non-finite or outside bounds")
        history_x.extend(row.copy() for row in initial_x)
        history_y.extend(row.copy() for row in initial_y)

    def recording_funct(x):
        x_t = x if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float32)
        x_np = x_t.detach().cpu().numpy().reshape(-1, dim).astype(np.float32)
        if np.any(x_np < -0.5 - 1e-6) or np.any(x_np > 0.5 + 1e-6):
            raise RuntimeError("tSS-BO proposed a point outside normalized bounds")
        remaining = int(n_max) - len(history_y)
        if remaining <= 0:
            raise _ConstrainedBudgetComplete
        partial_batch = len(x_np) > remaining
        if partial_batch:
            x_t = x_t.reshape(-1, dim)[:remaining]
            x_np = x_np[:remaining]
        rows = np.asarray(funct(x_t), dtype=np.float32)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        if rows.shape != (len(x_np), outdim):
            raise RuntimeError(
                f"constrained callback returned {rows.shape}, expected "
                f"({len(x_np)}, {outdim})"
            )
        if not np.all(np.isfinite(rows)):
            raise RuntimeError("constrained callback returned non-finite values")
        history_x.extend(row.copy() for row in x_np)
        history_y.extend(row.copy() for row in rows)
        if partial_batch:
            # The upstream caller still holds the full proposal batch. Stop
            # before its X/Y update instead of padding unevaluated points or
            # passing mismatched row counts. Incumbent selection below uses
            # the exact measured history, including this final prefix.
            raise _ConstrainedBudgetComplete
        return rows

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    previous_threads = int(torch.get_num_threads())
    # The reference implementation repeatedly diagonalizes a nearly symmetric
    # covariance.  Multi-threaded CPU LAPACK has produced non-reproducible
    # ``eigh`` convergence failures for the same seed on Windows.  One thread
    # makes the declared seed binding; restore the caller's setting afterward.
    torch.set_num_threads(1)
    tmp_root = Path(tempfile.mkdtemp(prefix="clay_tssbo_constrained_"))
    dataset = tmp_root / "history.pkl"
    sink = io.StringIO()
    cm = contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext()
    try:
        with cm:
            solver(
                recording_funct,
                int(dim),
                bounds,
                init_x=initial_x,
                # Upstream converts init_x itself but directly calls Tensor
                # methods on init_y (clip/argsort); satisfy that exact contract.
                init_y=(
                    torch.as_tensor(initial_y, dtype=torch.float32)
                    if initial_y is not None
                    else None
                ),
                sigma=float(sigma),
                mu=float(mu),
                c1=None,
                c2=None,
                allround_flag=False,
                greedy_flag=False,
                n_training=None,
                batch_size=int(batch_size),
                n_candidates=int(n_candidates),
                n_resample=int(n_resample),
                # Upstream loops while ``history_size <= nMax`` and also
                # measures a seed mean. Stop on the exact callback budget,
                # including a partial final batch when those counts differ.
                nMax=int(n_max) - 1,
                k=max(10, int(dim)),
                dataset_file=str(dataset),
                use_BO=True,
                use_TS=False,
                outdim=int(outdim),
                calculate_model_gradient=False,
            )
    except _ConstrainedBudgetComplete:
        pass
    finally:
        torch.set_num_threads(previous_threads)
        import shutil

        shutil.rmtree(tmp_root, ignore_errors=True)

    if len(history_y) != int(n_max):
        raise RuntimeError(
            f"tSS-BO returned before its evaluation budget: {len(history_y)}/{n_max}"
        )
    hx = np.asarray(history_x, dtype=np.float32)
    hy = np.asarray(history_y, dtype=np.float32)
    feasible_mask = np.all(hy[:, 1:] <= 0.0, axis=1)
    if np.any(feasible_mask):
        candidates = np.flatnonzero(feasible_mask)
        best_index = int(candidates[np.argmin(hy[candidates, 0])])
        feasible = True
    else:
        violation = np.maximum(hy[:, 1:], 0.0).sum(axis=1)
        best_index = int(np.lexsort((hy[:, 0], violation))[0])
        feasible = False
    return ConstrainedResult(
        best_x=hx[best_index].copy(),
        best_row=hy[best_index].copy(),
        feasible=feasible,
        history_x=hx,
        history_y=hy,
    )


def minimize(
    objective: Callable[[np.ndarray], np.ndarray],
    bounds: Sequence[Tuple[float, float]],
    *,
    n_max: int = 60,
    batch_size: int = 8,
    n_candidates: int = 60,
    n_resample: int = 5,
    sigma: float = 0.2,
    mu: float = 0.5,
    seed: int = 0,
    quiet: bool = True,
) -> Tuple[np.ndarray, float]:
    """Minimise ``objective`` over a box with the real tSS-BO solver.

    ``objective`` maps a batch tensor / array of shape ``(n, dim)`` to a length-``n``
    array of scalar losses (lower = better). ``bounds`` is a list of ``(low, high)``
    per dimension. Returns ``(best_x, best_y)`` -- tSS-BO pickles only the Y-history,
    so the best X is captured by a recorder wrapped around ``funct``. tSS-BO's
    verbose stdout is suppressed when ``quiet``."""
    import torch

    main_solver = _load_main_solver()
    dim = len(bounds)
    # tSS-BO works in float32 internally; a float64 bounds tensor triggers a
    # dtype mismatch in its desketch-gradient. Match the library's default.
    bnd = torch.tensor([[float(lo), float(hi)] for lo, hi in bounds], dtype=torch.float32)

    best = {"x": None, "y": float("inf")}

    def recording_funct(x):
        x_t = x if torch.is_tensor(x) else torch.as_tensor(x)
        # tSS-BO is float32 internally; a float64 objective return triggers a
        # dtype mismatch in its desketch-gradient. Return float32.
        y = np.atleast_1d(np.asarray(objective(x_t), dtype=np.float32)).ravel()
        x_np = x_t.detach().cpu().numpy().reshape(len(y), -1)
        for xi, yi in zip(x_np, y):
            if np.isfinite(yi) and yi < best["y"]:
                best["y"] = float(yi)
                best["x"] = xi.copy()
        return y

    try:
        torch.manual_seed(int(seed))
    except Exception:
        pass

    tmp = Path(tempfile.mkdtemp(prefix="clay_tssbo_")) / "hist.pkl"
    sink = io.StringIO()
    cm = contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext()
    cwd = os.getcwd()
    try:
        os.chdir(_TSSBO_DIR)  # main_solver writes dataset_file relative to cwd
        with cm:
            main_solver(
                recording_funct, dim, bnd,
                init_x=None, init_y=None,
                sigma=sigma, mu=mu, c1=None, c2=None,
                allround_flag=False, greedy_flag=False,
                n_training=None, batch_size=batch_size,
                n_candidates=n_candidates, n_resample=n_resample,
                nMax=int(n_max), k=max(10, dim), dataset_file=str(tmp), use_TS=False,
            )
    finally:
        os.chdir(cwd)
        import shutil
        shutil.rmtree(tmp.parent, ignore_errors=True)

    if best["x"] is None:
        raise RuntimeError("tSS-BO produced no finite evaluation")
    return best["x"], best["y"]


__all__ = [
    "ConstrainedResult",
    "minimize",
    "minimize_constrained",
    "tssbo_available",
    "tssbo_dir",
    "tssbo_provenance",
]
