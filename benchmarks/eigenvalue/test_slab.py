import os

import pytest
from igakit import cad
import torch

from ttnte import mpi_context
from ttnte.xs.benchmarks import research_reactor
from ttnte.cad import Patch
from ttnte.mesh import IGAMesh
from ttnte.physics import (
    BoundaryType,
    BCPlane,
    DGTransportAssemblerConfig,
)
from ttnte.math import QuadratureSet1D
from ttnte.driver import IGATransportDriver1D
from ttnte.solvers import (
    DDSolverConfig,
    MemoryPolicy,
    AMEnSolver,
    BlockJacobiStrategy,
    ExecMode,
    CommMode,
    IGADDSolver,
)

pytest.mark.mpi(min_size=1)


def test_research_reactor(request):
    passed = True

    # ========================================================================
    # Setup
    dtype = torch.float64
    cpu = torch.device("cpu")

    # Initialize MPI
    mpi_context.init()
    num_threads_per_rank = min(os.cpu_count() // mpi_context.world_size, 8)
    torch.set_num_threads(num_threads_per_rank)

    # Check MPI size
    if mpi_context.world_size > 2:
        pytest.skip("Test requires 2 or fewer processes")

    # Check if there are GPUs and enough of them available
    use_gpu = False
    if (
        torch.cuda.is_available()
        and torch.cuda.device_count() >= mpi_context.world_size
    ):
        use_gpu = True

    # Set defaults for PyTorch
    torch.set_default_dtype(dtype)
    torch.autograd.set_grad_enabled(False)

    # Create 1-D angular quadrature
    qset = QuadratureSet1D.gauss_legendre(64)
    qset.to_(cpu, dtype)

    # Spatial fidelity
    numel = 10
    degree = 4

    # pytest specific
    generated_plots = []

    # ========================================================================
    # Get XS info
    fills, xs_server = research_reactor(is_anisotropic=False, dtype=dtype, device=cpu)

    # ========================================================================
    # Create single-patch geometry (homogeneous circle)
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
    # Create the transport driver
    driver = IGATransportDriver1D(mesh, xs_server, mpi_context)

    # ========================================================================
    # Assemble operators
    config = DGTransportAssemblerConfig()
    config.rounding.eps = 1e-8
    config.cross.eps = config.rounding.eps
    config.max_dense_size = int(1e10)
    config.cross_jacobian_inverse = False
    driver.assemble(qset, config)

    # ========================================================================
    # Run DD solver
    outer_tol = 1e-4
    inner_tol = 5e-5
    eps = 1e-5

    # Create Block-Jacobi DD strategy
    config = DDSolverConfig(
        tol=inner_tol,
        tol_forcing=0.1,
        max_iter=100,
        use_gpu=use_gpu,
        memory_policy=MemoryPolicy.RESIDENT,
        exec_mode=ExecMode.ASYNC,
        comm_mode=CommMode.ASYNC,
        verbose=True,
    )
    strategy = BlockJacobiStrategy(config)
    strategy.set_local_solver(
        AMEnSolver(
            nswp=10,
            eps=eps,
            eps_forcing=0.01,
            kickrank=4,
            local_iterations=200,
            resets=4,
            max_rank=500,
        )
    )
    dd_solver = IGADDSolver(driver.mesh, strategy)

    # Run DD eigenvalue solver
    result = driver.solve_eigenvalue(
        dd_solver, tol=outer_tol, max_iter=500, verbose=True
    )

    # ========================================================================
    # Check the solution to the analytical
    keff_error = (result.k_eff - 1.0) * 1e5
    passed &= abs(keff_error) < 5

    # ========================================================================
    # Attach data to Pytest for conftest.py to read
    errors = {"dk (pcm)": keff_error}
    if mpi_context.rank == 0:
        for name, error in errors.items():
            print(f"{name}: {error}")

    request.node.vnv_plots = generated_plots
    request.node.vnv_metrics = {
        "name": request.node.name,
        "ttnte_k": result.k_eff,
        "ref_k": 1.0,
        "passed": passed,
        "detailed_errors": errors,
    }

    # Finally, formally assert to fail the test if outside tolerance
    assert (
        abs(keff_error) < 5
    ), "The eigenvalue was more than 5 pcm from the true solution"
