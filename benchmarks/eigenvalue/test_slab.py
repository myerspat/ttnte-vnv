import pytest
from igakit import cad
import torch

from ttnte import mpi_context
from ttnte.xs.benchmarks import research_reactor
from ttnte.cad import Patch
from ttnte.mesh import IGAMesh
from ttnte.physics import BoundaryType, BCPlane
from ttnte.math import QuadratureSet1D

from vnv import record, setup

# Solver settings for this problem, merged over vnv.solvers.DEFAULTS
# (--solver-opt on the command line wins over these)
OPTS = {
    "amen": {
        "tol": 1e-4,
        "dd": {"tol": 5e-5, "tol_forcing": 0.1, "max_iter": 100},
        "local": {
            "nswp": 10,
            "eps": 1e-5,
            "eps_forcing": 0.01,
            "kickrank": 4,
            "local_iterations": 200,
            "resets": 4,
            "max_rank": 500,
        },
    },
    "sweep": {"tol": 1e-5, "num_threads": 2},
    "cross": {"tol": 1e-5, "num_threads": 2, "cross": {"num_threads": 2}},
}


@pytest.mark.mpi(min_size=1)
def test_research_reactor(request, solver):
    # ========================================================================
    # Setup
    setup(max_ranks=2)
    dtype = torch.float64
    cpu = torch.device("cpu")

    # Create 1-D angular quadrature
    qset = QuadratureSet1D.gauss_legendre(64)
    qset.to_(cpu, dtype)

    # Spatial fidelity
    numel = 10
    degree = 4

    # ========================================================================
    # Get XS info
    fills, xs_server = research_reactor(is_anisotropic=False, dtype=dtype, device=cpu)

    # ========================================================================
    # Create two-patch slab geometry
    c0 = Patch.from_igakit(
        cad.refine(cad.line((-6.696802, 0), (0, 0)), numel, degree),
        device=cpu,
        dtype=dtype,
        fill=fills[0],
    )
    c1 = Patch.from_igakit(
        cad.refine(cad.line((0, 0), (1.126152, 0)), numel, degree),
        device=cpu,
        dtype=dtype,
        fill=fills[1],
    )

    # ========================================================================
    # Create mesh
    mesh = IGAMesh(mpi_context)
    mesh.add_block(c0)
    mesh.add_block(c1)
    mesh.connect()
    mesh.set_axis_aligned_conditions(
        BCPlane(x_min=True), BoundaryType.REFLECTIVE, tol=1e-6
    )
    mesh.finalize()

    # ========================================================================
    # Solve
    result = solver.run(mesh, xs_server, qset, eigenvalue=True, opts=OPTS)

    # ========================================================================
    # Check the solution to the analytical
    keff_error = (result.k_eff - 1.0) * 1e5
    passed = abs(keff_error) < 5

    # ========================================================================
    # Attach data to Pytest for conftest.py to read
    record(
        request,
        result,
        value=result.k_eff,
        ref=1.0,
        passed=passed,
        errors={"dk (pcm)": keff_error},
    )

    # Finally, formally assert to fail the test if outside tolerance
    assert (
        abs(keff_error) < 5
    ), "The eigenvalue was more than 5 pcm from the true solution"
