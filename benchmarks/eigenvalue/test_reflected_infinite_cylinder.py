import os

from igakit import cad
import pytest
import torch

from ttnte import mpi_context
from ttnte.xs.benchmarks import pu239
from ttnte.visualization.style import get_patch_style
from ttnte.cad.surfaces import circle
from ttnte.cad import Patch
from ttnte.mesh import IGAMesh
from ttnte.physics import (
    BoundaryType,
    BCPlane,
    DGTransportAssemblerConfig,
)
from ttnte.math import ProductQuadrature
from ttnte.driver import IGATransportDriver2D
from ttnte.solvers import (
    DDSolverConfig,
    IGADDSolver,
    MemoryPolicy,
    AMEnSolver,
    BlockJacobiStrategy,
    StaticFreezePolicy,
    ExecMode,
    CommMode,
)
from ttnte.parallel import IGADofHeuristic
from ttnte.linalg import AMEnNativeOptions, AMEnEnrichmentMode


@pytest.mark.mpi(min_size=1)
def test_reflected_infinite_cylinder(request):
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
    if mpi_context.world_size > 13:
        pytest.skip("Test requires 13 or fewer processes")

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

    # Create angular quadrature
    qset = ProductQuadrature.gauss_legendre_chebyshev(16, 16, 2)
    qset.to_(torch.device("cpu"), dtype)

    # Spatial fidelity
    numel = 8
    degree = 2

    # pytest specific
    generated_plots = []

    # ========================================================================
    # Get XS information
    fills, xs_server = pu239(num_groups=1, device=cpu, dtype=dtype)

    # ========================================================================
    # Create NURBS surfaces
    # Fuel regions
    r0 = 2.397610
    s0 = circle(r0)

    r1 = r0 + 1.0
    c0 = s0.boundary(0, 0)
    c1 = cad.circle(r1, angle=torch.pi / 2).rotate(-3 * torch.pi / 4)
    s1 = cad.ruled(c0, c1)
    c2 = s0.boundary(0, 1)
    c3 = cad.circle(r1, angle=-torch.pi / 2).rotate(3 * torch.pi / 4)
    s2 = cad.ruled(c2, c3)
    c4 = s0.boundary(1, 0)
    c5 = cad.circle(r1, angle=-torch.pi / 2).rotate(-3 * torch.pi / 4)
    s3 = cad.ruled(c4, c5)
    c6 = s0.boundary(1, 1)
    c7 = cad.circle(r1, angle=torch.pi / 2).rotate(-torch.pi / 4)
    s4 = cad.ruled(c6, c7)

    # Water regions
    r2 = r1 + 1.0
    c8 = cad.circle(r2, angle=torch.pi / 2).rotate(-3 * torch.pi / 4)
    s5 = cad.ruled(c1, c8)
    c9 = cad.circle(r2, angle=-torch.pi / 2).rotate(3 * torch.pi / 4)
    s6 = cad.ruled(c3, c9)
    c10 = cad.circle(r2, angle=-torch.pi / 2).rotate(-3 * torch.pi / 4)
    s7 = cad.ruled(c5, c10)
    c11 = cad.circle(r2, angle=torch.pi / 2).rotate(-torch.pi / 4)
    s8 = cad.ruled(c7, c11)

    r3 = r2 + 2.063725
    c12 = cad.circle(r3, angle=torch.pi / 2).rotate(-3 * torch.pi / 4)
    s9 = cad.ruled(c8, c12)
    c13 = cad.circle(r3, angle=-torch.pi / 2).rotate(3 * torch.pi / 4)
    s10 = cad.ruled(c9, c13)
    c14 = cad.circle(r3, angle=-torch.pi / 2).rotate(-3 * torch.pi / 4)
    s11 = cad.ruled(c10, c14)
    c15 = cad.circle(r3, angle=torch.pi / 2).rotate(-torch.pi / 4)
    s12 = cad.ruled(c11, c15)

    # ========================================================================
    # Create IGA mesh
    mesh = IGAMesh(mpi_context)

    for fill, s in zip(
        5 * [fills[0]] + 8 * [fills[1]],
        [s0, s1, s2, s3, s4] + [s5, s6, s7, s8] + [s9, s10, s11, s12],
    ):
        mesh.add_block(
            Patch.from_igakit(
                cad.refine(s, numel, degree), device=cpu, dtype=dtype, fill=fill
            )
        )

    mesh.connect()
    mesh.finalize()

    # ========================================================================
    # Plot mesh
    generated_plots.append(f"{request.node.name}_model.png")
    backend = "matplotlib"
    style = get_patch_style(backend)
    style.mesh.cmap = {
        fills[0].to_string(): "maroon",
        fills[1].to_string(): "cornflowerblue",
    }
    mesh.plot(
        resolution=25,
        show_ctrlpts=True,
        show_ctrlnet=True,
        show_boundary=True,
        backend="matplotlib",
        filename=generated_plots[-1],
        style=style,
    )

    # ========================================================================
    # Create transport driver and distribute across MPI ranks
    # Create the transport driver
    driver = IGATransportDriver2D(mesh, xs_server, mpi_context)

    # Distribute patches among MPI ranks
    driver.distribute([IGADofHeuristic()])

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
    outer_tol = 1e-5
    inner_tol = 1e-6
    eps = 1e-8

    # Create Block-Jacobi DD strategy
    config = DDSolverConfig(
        tol=inner_tol,
        tol_forcing=0.5,
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
            nswp=4,
            eps=eps,
            eps_forcing=0.01,
            kickrank=4,
            local_iterations=100,
            resets=4,
            max_rank=500,
            native_opts=AMEnNativeOptions(
                enrichment_mode=AMEnEnrichmentMode.FULL,
                als_residual_rank=0,
                proximal_regularization=0.01,
                gmres_mixed_precision=True,
            ),
            enrichment_policy=StaticFreezePolicy(freeze_eps=1e-4),
        )
    )
    dd_solver = IGADDSolver(driver.mesh, strategy)

    # Run DD eigenvalue solver
    result = driver.solve_eigenvalue(
        dd_solver, tol=outer_tol, max_iter=500, verbose=True
    )

    # ========================================================================
    # Plot the solution
    # Compute the scalar flux
    scalar_result = result.compute_scalar_flux()
    generated_plots.append(f"{request.node.name}_flux.png")
    mesh.plot(
        resolution=25,
        field_label=r"$\phi$",
        solution=scalar_result.select_group(0),
        filename=generated_plots[-1],
        gather=True,  # This argument is needed to gather all the data onto a single rank
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
