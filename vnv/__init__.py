"""Helpers shared by the V&V benchmarks: run any ttnte solver (``vnv.solvers``) and
compare against reference results (``vnv.compare``)."""

import os

import pytest
import torch
from ttnte import mpi_context

from vnv.compare import flux_errors, record, zscores
from vnv.solvers import DEFAULTS, NAMES, PRESETS, SOLVERS, RunResult, Solver, run


def setup(max_ranks=None, dtype=torch.float64):
    """Common per-test setup: MPI, threads per rank, default dtype, no autograd.

    Skips the test if more than ``max_ranks`` MPI ranks are running.
    """
    mpi_context.init()
    if max_ranks is not None and mpi_context.world_size > max_ranks:
        pytest.skip(f"Test requires {max_ranks} or fewer processes")
    torch.set_num_threads(min(os.cpu_count() // mpi_context.world_size, 8))
    torch.set_default_dtype(dtype)
    torch.autograd.set_grad_enabled(False)


# Characters actions/upload-artifact rejects in file names (Windows-invalid)
_UNSAFE = str.maketrans({c: "-" for c in ':"<>|*?\r\n'})


def plot_stem(request):
    """The test id as a file-name stem for plots (``{stem}_flux.png``): the
    node name with the characters artifact uploads reject replaced by "-", e.g.
    ``test_slab[galerkin:deim]`` -> ``test_slab[galerkin-deim]``."""
    return request.node.name.translate(_UNSAFE)


__all__ = [
    "DEFAULTS",
    "NAMES",
    "PRESETS",
    "SOLVERS",
    "RunResult",
    "Solver",
    "flux_errors",
    "plot_stem",
    "record",
    "run",
    "setup",
    "zscores",
]
