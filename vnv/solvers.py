"""Run a benchmark problem with any of ttnte's three transport solvers.

- ``amen``: block-Jacobi domain decomposition with the AMEn local solver
  (``IGATransportDriver`` + ``IGADDSolver``)
- ``sweep``: the exact transport sweeper (``IGASweepDriver``, ``TransportSolve.EXACT``)
- ``cross``: the sweeper with low-rank cross transport solves
  (``TransportSolve.CROSS``)

Every solver is configured by a nested dict of settings. They are layered, later
layers winning:

1. ``DEFAULTS[name]`` below,
2. the ``opts`` a test passes to ``run()`` (``{"amen": {...}, "cross": {...}}``),
3. ``--solver-opt`` on the pytest command line (see ``parse_overrides``).

Settings of the ``amen`` solver:

- ``tol``, ``k_tol``, ``max_iter``, ``verbose``: outer (power / source) iteration,
  passed to ``driver.solve_eigenvalue()`` / ``driver.solve_fixed_source()``
- ``assembler``: ``DGTransportAssemblerConfig`` attributes; ``eps`` sets both the
  rounding and cross truncation tolerances
- ``strategy``: ``"dd"`` (block-Jacobi DD), ``"direct"`` (the AMEn solver alone on a
  single patch, no DD loop) or ``None``: direct for a single patch on one rank, else
  DD. The DD convergence check only sees patch interfaces, so on a single patch it
  stops after one loose local solve and the outer iteration stalls.
- ``dd``: ``DDSolverConfig`` attributes (``use_gpu: None`` picks the GPU when every
  rank can have one; also used by ``"direct"``)
- ``local``: ``AMEnSolver`` keyword arguments
- ``native``: ``AMEnNativeOptions`` attributes
- ``freeze_eps``: ``StaticFreezePolicy`` freeze tolerance (``None``: no policy)
- ``keep_assemblers``: keep the assemblers after the solve (as ``balance=True``
  does) for diagnostics through ``result.driver.get_assembler(gid)``

Settings of the ``sweep`` and ``cross`` solvers:

- ``k_tol``: passed to ``solve_eigenvalue()``
- every other key: a ``SweepConfig`` attribute (``num_threads: None`` uses
  ``torch.get_num_threads()``)
- ``cross``: ``CrossConfig`` attributes

Enum-valued settings accept the member name as a string (e.g.
``"memory_policy": "OUT_OF_CORE"``).
"""

import ast
import copy
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from ttnte import mpi_context
from ttnte.driver import (
    IGATransportDriver1D,
    IGATransportDriver2D,
    IGATransportDriver3D,
)
from ttnte.linalg import AMEnBackend, AMEnNativeOptions, AMEnPreconditioner
from ttnte.parallel import IGADofHeuristic
from ttnte.physics import DGTransportAssemblerConfig
from ttnte.solvers import (
    AMEnSolver,
    BlockJacobiStrategy,
    DDSolverConfig,
    IGADDSolver,
    StaticFreezePolicy,
)
from ttnte.sweep import (
    IGASweepDriver1D,
    IGASweepDriver2D,
    IGASweepDriver3D,
    SweepConfig,
)

SOLVERS = ("amen", "sweep", "cross")

DEFAULTS = {
    "amen": {
        "tol": 1e-6,
        "k_tol": 1e-5,
        "max_iter": 500,
        "verbose": True,
        "strategy": None,
        "keep_assemblers": False,
        "assembler": {
            "eps": 1e-8,
            "max_dense_size": int(1e10),
            "cross_jacobian_inverse": False,
        },
        "dd": {"use_gpu": None, "verbose": True},
        "local": {
            "nswp": 4,
            "eps": 1e-7,
            "eps_forcing": 0.01,
            "kickrank": 4,
            "local_iterations": 200,
            "resets": 4,
            "max_rank": 500,
        },
        "native": {},
        "freeze_eps": None,
    },
    "sweep": {
        "k_tol": 1e-6,
        "tol": 1e-6,
        "max_iter": 1000,
        "num_threads": None,
        "verbose": True,
    },
    "cross": {
        "k_tol": 1e-6,
        "tol": 1e-6,
        "max_iter": 1000,
        "num_threads": None,
        "verbose": True,
        "transport": "CROSS",
        "cross": {"num_threads": None},
    },
}

_TRANSPORT_DRIVERS = {
    1: IGATransportDriver1D,
    2: IGATransportDriver2D,
    3: IGATransportDriver3D,
}
_SWEEP_DRIVERS = {1: IGASweepDriver1D, 2: IGASweepDriver2D, 3: IGASweepDriver3D}


# ============================================================================
# Settings
def merge(base: dict, update: dict) -> dict:
    """Return a copy of ``base`` with ``update`` merged in recursively."""
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _parse_value(raw: str) -> Any:
    raw = raw.strip()
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        pass
    try:
        return float(raw)  # inf, nan
    except ValueError:
        return raw  # enum names, layouts, ...


def parse_overrides(items) -> dict:
    """Parse ``[SOLVER.]KEY[.SUBKEY...]=VALUE`` strings into per-solver settings.

    ``amen.local.eps=1e-6`` sets one solver's setting; a path that does not start
    with a solver name (``tol=1e-5``) is set for every solver. Values are Python
    literals, or plain strings (enum member names, cross layouts).
    """
    out = {name: {} for name in SOLVERS}
    for item in items:
        path, sep, raw = item.partition("=")
        keys = path.strip().split(".")
        if not sep or not all(keys):
            raise ValueError(
                f"--solver-opt '{item}' is not of the form [SOLVER.]KEY=VALUE"
            )
        if len(keys) > 1 and keys[0] in SOLVERS:
            targets, keys = [keys[0]], keys[1:]
        else:
            targets = SOLVERS
        value = _parse_value(raw)
        for name in targets:
            node = out[name]
            for key in keys[:-1]:
                node = node.setdefault(key, {})
            node[keys[-1]] = value
    return out


def _set(obj, key: str, value, where: str):
    """``setattr`` with a clear error for unknown keys and enum names as strings."""
    if not hasattr(obj, key):
        raise KeyError(f"{where} has no setting '{key}'")
    current = getattr(obj, key)
    members = getattr(type(current), "__members__", None)
    if isinstance(value, str) and members is not None:
        if value not in members:
            raise KeyError(
                f"{where}.{key}: '{value}' is not one of {', '.join(members)}"
            )
        value = members[value]
    setattr(obj, key, value)


def _enum(enum_type, value):
    return enum_type.__members__[value] if isinstance(value, str) else value


def _default_use_gpu() -> bool:
    return torch.cuda.is_available() and (
        torch.cuda.device_count() >= mpi_context.world_size
    )


# ============================================================================
# Results
@dataclass
class RunResult:
    """One solver run of one problem.

    ``solution`` is what the driver returned; ``scalar`` is always the scalar flux
    (use it for ``regular_mesh_average`` and plotting).
    """

    solver: str
    settings: dict
    solution: Any
    scalar: Any
    k_eff: float | None
    seconds: float  # driver construction + setup/assembly + solve
    solve_seconds: float
    iterations: int
    driver: Any
    stats: dict = field(default_factory=dict)
    _assemblers: Any = None
    _angular: bool = False

    def _balance_args(self, kwargs):
        if not self._angular:
            raise RuntimeError(
                "Particle balances need the angular flux: pass balance=True to run()"
            )
        if self._assemblers is not None:
            kwargs.setdefault("assemblers", self._assemblers)
        return kwargs

    def global_balance(self, **kwargs):
        """``TransportSolution.global_balance`` of the angular flux (collective)."""
        return self.solution.global_balance(**self._balance_args(kwargs))

    def patch_balance_table(self, **kwargs):
        """``TransportSolution.patch_balance_table`` of the angular flux
        (collective)."""
        return self.solution.patch_balance_table(**self._balance_args(kwargs))


# ============================================================================
# Running
def run(
    name: str,
    mesh,
    xs_server,
    qset,
    *,
    eigenvalue: bool,
    balance: bool = False,
    opts: dict | None = None,
    overrides: dict | None = None,
) -> RunResult:
    """Solve one problem with solver ``name``.

    Args:
        name: ``"amen"``, ``"sweep"`` or ``"cross"``.
        mesh: A finalized ``IGAMesh``, built for this run only (the drivers
            partition it across MPI ranks).
        xs_server: The cross-section ``Server``.
        qset: The angular quadrature (``QuadratureSet1D`` or ``ProductQuadrature``).
        eigenvalue: Solve the k-eigenvalue problem, else the fixed-source problem.
        balance: Keep what particle balances need (``result.global_balance()``,
            ``result.patch_balance_table()``): the assemblers for ``amen``, the
            angular flux for ``sweep`` / ``cross``.
        opts: Per-solver settings of the test, ``{"amen": {...}, ...}``; only
            ``opts[name]`` is used.
        overrides: Per-solver settings that win over ``opts`` (from
            ``--solver-opt``).
    """
    if name not in SOLVERS:
        raise ValueError(f"Unknown solver '{name}', expected one of {SOLVERS}")
    settings = merge(DEFAULTS[name], (opts or {}).get(name, {}))
    settings = merge(settings, (overrides or {}).get(name, {}))
    num_dim = mesh.blocks[0].ndim  # spatial dimension (qset.ndim is angular)

    if name == "amen":
        return _run_amen(mesh, xs_server, qset, num_dim, eigenvalue, balance, settings)
    return _run_sweep(
        name, mesh, xs_server, qset, num_dim, eigenvalue, balance, settings
    )


def _run_amen(mesh, xs_server, qset, num_dim, eigenvalue, balance, s):
    start = time.perf_counter()
    driver = _TRANSPORT_DRIVERS[num_dim](mesh, xs_server, mpi_context)
    driver.distribute([IGADofHeuristic()])

    # Assemble operators
    assembler = dict(s["assembler"])
    config = DGTransportAssemblerConfig()
    eps = assembler.pop("eps")
    config.rounding.eps = eps
    config.cross.eps = eps
    for key, value in assembler.items():
        _set(config, key, value, "amen.assembler")
    driver.assemble(qset, config)

    # Block-Jacobi DD with the AMEn local solver
    dd = DDSolverConfig()
    for key, value in s["dd"].items():
        if key == "use_gpu" and value is None:
            value = _default_use_gpu()
        _set(dd, key, value, "amen.dd")
    native = AMEnNativeOptions()
    for key, value in s["native"].items():
        _set(native, key, value, "amen.native")
    local = dict(s["local"])
    if "prec" in local:
        local["prec"] = _enum(AMEnPreconditioner, local["prec"])
    if "backend" in local:
        local["backend"] = _enum(AMEnBackend, local["backend"])
    policy = (
        StaticFreezePolicy(freeze_eps=s["freeze_eps"])
        if s["freeze_eps"] is not None
        else None
    )
    local_solver = AMEnSolver(native_opts=native, enrichment_policy=policy, **local)

    strategy = s["strategy"]
    if strategy is None:
        single = mesh.num_blocks == 1 and mpi_context.world_size == 1
        strategy = "direct" if single else "dd"
    if strategy == "direct":
        if mesh.num_blocks != 1 or mpi_context.world_size != 1:
            raise ValueError("amen.strategy 'direct' needs one patch on one rank")
        inner_solver = local_solver
        system = driver.get_system(mesh.blocks[0].gid)
    elif strategy == "dd":
        block_jacobi = BlockJacobiStrategy(dd)
        block_jacobi.set_local_solver(local_solver)
        inner_solver = IGADDSolver(driver.mesh, block_jacobi)
    else:
        raise ValueError(
            f"amen.strategy must be 'dd', 'direct' or None, not {strategy}"
        )

    # Outer iteration
    kwargs = dict(
        tol=s["tol"],
        max_iter=s["max_iter"],
        verbose=s["verbose"],
        clear_assemblers=not (balance or s["keep_assemblers"]),
    )
    solve_start = time.perf_counter()
    if strategy == "direct" and dd.use_gpu:
        system.to_(torch.device("cuda"))
    if eigenvalue:
        solution = driver.solve_eigenvalue(inner_solver, k_tol=s["k_tol"], **kwargs)
    else:
        solution = driver.solve_fixed_source(inner_solver, **kwargs)
    if strategy == "direct" and dd.use_gpu:
        system.to_(torch.device("cpu"))
    end = time.perf_counter()

    return RunResult(
        solver="amen",
        solution=solution,
        scalar=solution.compute_scalar_flux(),
        k_eff=solution.k_eff,
        seconds=end - start,
        solve_seconds=end - solve_start,
        iterations=solution.num_outer_iterations,
        driver=driver,
        stats={"inner iterations": solution.total_inner_iterations},
        settings=s | {"strategy": strategy},
        _assemblers=driver.get_assemblers() if balance else None,
        _angular=True,
    )


def _run_sweep(name, mesh, xs_server, qset, num_dim, eigenvalue, balance, s):
    start = time.perf_counter()
    s = dict(s)
    k_tol = s.pop("k_tol")
    cross = s.pop("cross", {})

    config = SweepConfig()
    for key, value in s.items():
        if key == "num_threads" and value is None:
            value = torch.get_num_threads()
        _set(config, key, value, name)
    for key, value in cross.items():
        if key == "num_threads" and value is None:
            value = torch.get_num_threads()
        _set(config.cross, key, value, f"{name}.cross")
    if balance:
        config.store_angular_flux = True

    driver = _SWEEP_DRIVERS[num_dim](mesh, xs_server, mpi_context)
    driver.setup(qset, config)
    solve_start = time.perf_counter()
    if eigenvalue:
        solution = driver.solve_eigenvalue(k_tol)
    else:
        solution = driver.solve_fixed_source()
    end = time.perf_counter()

    angular = config.store_angular_flux
    stats = {
        "converged": driver.converged,
        "equivalent sweeps": driver.equivalent_sweeps,
    }
    if name == "cross" and len(driver.cross_ranks) > 0:
        stats["final cross ranks"] = driver.cross_ranks[-1]

    return RunResult(
        solver=name,
        settings=s | {"k_tol": k_tol, "cross": cross},
        solution=solution,
        scalar=solution.compute_scalar_flux() if angular else solution,
        k_eff=driver.k_eff if eigenvalue else None,
        seconds=end - start,
        solve_seconds=end - solve_start,
        iterations=driver.num_iterations,
        driver=driver,
        stats=stats,
        _angular=angular,
    )


class Solver:
    """What the ``solver`` pytest fixture hands a test: a solver name plus the
    ``--solver-opt`` overrides of this pytest run."""

    def __init__(self, name: str, overrides: dict | None = None):
        if name not in SOLVERS:
            raise ValueError(f"Unknown solver '{name}', expected one of {SOLVERS}")
        self.name = name
        self.overrides = overrides or {}

    def run(self, mesh, xs_server, qset, **kwargs) -> RunResult:
        """``run(self.name, ...)`` with this pytest run's overrides; see ``run``."""
        return run(self.name, mesh, xs_server, qset, overrides=self.overrides, **kwargs)

    def __repr__(self):
        return self.name
