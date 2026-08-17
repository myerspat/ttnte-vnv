import os
from pathlib import Path

import pytest
import torch
import numpy as np
from igakit import cad
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ttnte import mpi_context
from ttnte.cad import Patch
from ttnte.mesh import IGAMesh
from ttnte.physics import (
    BoundaryType,
    BCPlane,
    DGTransportAssemblerConfig,
)
from ttnte.math import ProductQuadrature
from ttnte.linalg import Operator, TTEngine, mm
from ttnte.driver import IGATransportDriver2D
from ttnte.solvers import (
    DDSolverConfig,
    MemoryPolicy,
    AMEnSolver,
    BlockJacobiStrategy,
    IGADDSolver,
    ExecMode,
    CommMode,
    StaticFreezePolicy,
)
from ttnte.visualization.style import get_patch_style
from ttnte.parallel import IGADofHeuristic
from ttnte.linalg import AMEnNativeOptions, AMEnEnrichmentMode
from ttnte.xs.benchmarks import c5g7


@pytest.mark.slow
@pytest.mark.mpi(min_size=1)
def test_c5g7_pincell_2d(request):
    passed = True

    # ========================================================================
    # Setup
    # Settings
    dtype = torch.float64
    cpu = torch.device("cpu")
    ref_dir = (
        Path(__file__).resolve().parent.parents[1]
        / "reference/eigenvalue/c5g7_pincell_2d"
    )

    # Check the reference data exists
    assert (
        ref_dir.exists() and (ref_dir / "data").exists()
    ), f"Reference data directory not found at {ref_dir / 'data'}"

    # Initialize MPI
    mpi_context.init()
    num_threads_per_rank = min(os.cpu_count() // mpi_context.world_size, 8)
    torch.set_num_threads(num_threads_per_rank)

    # Check MPI size
    if mpi_context.world_size > 3:
        pytest.skip("Test requires 3 or fewer processes")

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
    qset.to_(cpu, dtype)

    # Spatial fidelity
    numel = 12
    degree = 2

    # pytest specific
    generated_plots = []

    # ========================================================================
    # Get XS information
    fills, xs_server = c5g7(device=cpu, dtype=dtype)

    # ========================================================================
    # Create NURBS patches
    # Create quarter circle NURBS surface
    radius = 0.54  # cm
    pitch = 1.26  # cm
    c0 = cad.circle(radius, angle=torch.pi / 4)
    c1 = cad.circle(radius, angle=-torch.pi / 4).rotate(torch.pi / 2)
    l0 = cad.line((0, 0), (radius, 0))
    l1 = cad.line((0, 0), (0, radius))

    s0 = cad.coons([[l1, c0], [l0, c1]])
    c0 = s0.boundary(0, 1)
    c1 = s0.boundary(1, 1)

    # Create water patch
    l2 = cad.line(p0=(pitch / 2, 0), p1=(pitch / 2, pitch / 2))
    l3 = cad.line(p0=(0, pitch / 2), p1=(pitch / 2, pitch / 2))

    # Create NURBS surfaces
    fuel = [s0]
    water = [cad.ruled(c0, l2), cad.ruled(c1, l3)]

    # ========================================================================
    # Create mesh
    mesh = IGAMesh(mpi_context)

    for fill, surface in zip([fills[0], fills[-2], fills[-2]], fuel + water):
        mesh.add_block(
            Patch.from_igakit(
                cad.refine(surface, numel, degree), device=cpu, dtype=dtype, fill=fill
            )
        )

    # Connect patches
    mesh.connect()

    # Set the boundary conditions
    mesh.set_axis_aligned_conditions(
        BCPlane(x_min=True, y_min=True, x_max=True, y_max=True),
        BoundaryType.REFLECTIVE,
        tol=1e-6,
    )
    mesh.finalize()

    # ========================================================================
    # Plot mesh
    generated_plots.append(f"{request.node.name}_model.png")
    backend = "matplotlib"
    style = get_patch_style(backend)
    style.mesh.cmap = {
        fills[0].to_string(): "maroon",
        fills[-2].to_string(): "cornflowerblue",
    }
    mesh.plot(
        resolution=25,
        show_ctrlpts=True,
        show_ctrlnet=True,
        show_boundary=True,
        backend=backend,
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

    for patch in mesh.blocks:
        assembler = driver.get_assembler(patch.gid)

        string = f"GID: {patch.gid}\n"

        op = assembler.interior_loss_op.as_tt()
        string += f"H: Ranks = {op.ranks}, CR = {op.compression}\n"
        op = assembler.scatter_op.as_tt()
        string += f"S: Ranks = {op.ranks}, CR = {op.compression}\n"

        if assembler.fission_op.defined():
            op = assembler.fission_op.as_tt()
            string += f"F: Ranks = {op.ranks}, CR = {op.compression}\n"

        for op in assembler.inflow_ops:
            op = op.as_tt()
            string += f"Bin: Ranks = {op.ranks}, CR = {op.compression}\n"
        for op in assembler.outflow_ops:
            op = op.as_tt()
            string += f"Bout: Ranks = {op.ranks}, CR = {op.compression}\n"

        print(string, end="")

    # ========================================================================
    # Run DD solver
    outer_tol = 1e-6
    inner_tol = 5e-7
    eps = 1e-7

    # Local solver
    local_solver = AMEnSolver(
        nswp=4,
        eps=eps,
        eps_forcing=0.01,
        kickrank=4,
        local_iterations=200,
        resets=4,
        max_rank=200,
        native_opts=AMEnNativeOptions(
            enrichment_mode=AMEnEnrichmentMode.FULL,
            als_residual_rank=0,
            proximal_regularization=0.01,
            gmres_mixed_precision=True,
        ),
        enrichment_policy=StaticFreezePolicy(freeze_eps=1e-4),
    )

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
    strategy.set_local_solver(local_solver)
    dd_solver = IGADDSolver(driver.mesh, strategy)

    # Run power iteration + DD solver
    result = driver.solve_eigenvalue(
        dd_solver, tol=outer_tol, max_iter=100, verbose=True
    )

    # ========================================================================
    # Plot the solution
    # Compute the scalar flux
    scalar_result = result.compute_scalar_flux()

    for g in range(xs_server.num_groups):
        generated_plots.append(f"{request.node.name}_flux_{g + 1}.png")
        mesh.plot(
            resolution=25,
            solution=scalar_result.select_group(g),
            filename=generated_plots[-1],
            field_label=rf"$\phi_{g + 1}$",
            gather=True,
        )

    # ========================================================================
    # Load OpenMC solution and compare
    k_mc = [1.325593, 0.000032]

    # Load OpenMC scalar flux data
    data_dir = ref_dir / "data"
    phi_mc = np.load(data_dir / "mesh_flux.npy")
    phi_mc_stdev = np.load(data_dir / "mesh_stdev.npy")

    # Ensure OpenMC solution is normalized
    phi_mc_stdev /= np.linalg.norm(phi_mc.flatten(), 2)
    phi_mc /= np.linalg.norm(phi_mc.flatten(), 2)

    # Calculate the eigenvalue error
    keff_error = (result.k_eff - k_mc[0]) * 1e5
    passed = abs(keff_error) < 100.0

    # Average the NURBS solution onto a global regular mesh
    avg_scalar_result = (
        scalar_result.regular_mesh_average([phi_mc.shape[1], phi_mc.shape[2]], [5, 5])
        .moveaxis(-1, 0)
        .cpu()
        .numpy()
    )
    flux_tol = 0.04

    # Normalize the average scalar flux
    avg_scalar_result /= np.linalg.norm(avg_scalar_result.flatten(), ord=2)

    # Get the relative L2-errors between ttnte and OpenMC
    group_errors = np.linalg.norm(
        (avg_scalar_result - phi_mc).reshape(xs_server.num_groups, -1), axis=1, ord=2
    ) / np.linalg.norm(phi_mc.reshape(xs_server.num_groups, -1), axis=1, ord=2)

    # Get total error
    total_error = np.linalg.norm(group_errors, ord=2)
    passed &= total_error < flux_tol

    # Compute and plot z-score per energy group
    zscores = np.abs(avg_scalar_result - phi_mc) / phi_mc_stdev
    X, Y = np.meshgrid(
        np.linspace(0, pitch / 2, phi_mc.shape[1]),
        np.linspace(0, pitch / 2, phi_mc.shape[2]),
    )

    stats = {
        name: np.zeros(xs_server.num_groups + 1)
        for name in ["Minimum", "Q1", "Median", "Q2", "Maximum", "Mean"]
    }

    for g in range(xs_server.num_groups):
        passed &= group_errors[g] < flux_tol

        generated_plots.append(f"{request.node.name}_zscore_{g + 1}.png")

        plt.clf()
        ax = plt.gca()
        cmesh = ax.pcolormesh(X, Y, zscores[g,], cmap="plasma")
        divider = make_axes_locatable(ax)
        cbar = plt.colorbar(
            cmesh,
            cax=divider.append_axes("right", size="5%", pad=0.05),
        )
        cbar.set_label(
            r"$\frac{|\phi_"
            + str(g + 1)
            + r"^{\text{ttnte}} - \phi_"
            + str(g + 1)
            + r"^{\text{OpenMC}}|}{|\sigma_"
            + str(g + 1)
            + r"^{\text{OpenMC}}|}$",
            size=14,
        )
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        plt.tight_layout()
        plt.savefig(generated_plots[-1], dpi=300)

        # Get some stats
        stats["Minimum"][g] = np.min(zscores[g,])
        stats["Q1"][g] = np.percentile(zscores[g,], 25)
        stats["Median"][g] = np.median(zscores[g,])
        stats["Q2"][g] = np.percentile(zscores[g,], 75)
        stats["Maximum"][g] = np.max(zscores[g,])
        stats["Mean"][g] = np.mean(zscores[g,])

    stats["Minimum"][-1] = np.min(zscores)
    stats["Q1"][-1] = np.percentile(zscores, 25)
    stats["Median"][-1] = np.median(zscores)
    stats["Q2"][-1] = np.percentile(zscores, 75)
    stats["Maximum"][-1] = np.max(zscores)
    stats["Mean"][-1] = np.mean(zscores)

    # ========================================================================
    # Attach data to Pytest for conftest.py to read
    errors = {
        "dk (pcm)": keff_error,
        "dk (z-score)": abs(keff_error) / (k_mc[1] * 1e5),
        "Total relative L2 error": total_error,
    }
    for g in range(xs_server.num_groups):
        errors[f"Relative relative L2 error (g = {g + 1})"] = group_errors[g]
    for name, stat in stats.items():
        for g in range(xs_server.num_groups):
            errors[f"{name} ({g + 1})"] = stat[g]
        errors[name] = stat[-1]

    if mpi_context.rank == 0:
        for name, error in errors.items():
            print(f"{name}: {error}")

    request.node.vnv_plots = generated_plots
    request.node.vnv_metrics = {
        "name": request.node.name,
        "ttnte_k": result.k_eff,
        "ref_k": k_mc[0],
        "passed": passed,
        "detailed_errors": errors,
    }

    # Finally, formally assert to fail the test if outside tolerance
    assert (
        abs(keff_error) < 100
    ), "The eigenvalue was more than 100 pcm from the true solution"

    for g in range(xs_server.num_groups):
        assert (
            group_errors[g] < flux_tol
        ), f"Scalar flux error for group {g + 1} is greater than the allowed tolerance"
    assert (
        total_error < flux_tol
    ), "The total scalar flux error is greater than the allowed tolerance"
