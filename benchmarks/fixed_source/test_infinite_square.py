import os
from pathlib import Path

import pytest
from igakit.cad import refine, bilinear
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ttnte import mpi_context
from ttnte.xs import Material, Server
from ttnte.cad import Patch
from ttnte.mesh import IGAMesh
from ttnte.physics import (
    BoundaryType,
    BCPlane,
    DGTransportAssemblerConfig,
    FixedSource,
)
from ttnte.math import ProductQuadrature
from ttnte.driver import IGATransportDriver2D
from ttnte.linalg import AMEnNativeOptions, AMEnEnrichmentMode
from ttnte.solvers import (
    DDSolverConfig,
    MemoryPolicy,
    AMEnSolver,
    BlockJacobiStrategy,
    ExecMode,
    CommMode,
    StaticFreezePolicy,
)


def test_infinite_square(request):
    passed = True

    # ========================================================================
    # Setup
    dtype = torch.float64
    cpu = torch.device("cpu")
    ref_dir = (
        Path(__file__).resolve().parent.parents[1]
        / "reference/fixed_source/infinite_square"
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
    if mpi_context.world_size > 1:
        pytest.skip("Test requires 1 or fewer processes")

    # Set defaults for PyTorch
    torch.set_default_dtype(dtype)
    torch.autograd.set_grad_enabled(False)

    # Create angular quadrature
    qset = ProductQuadrature.gauss_legendre_chebyshev(32, 32, 2)
    qset.to_(torch.device("cpu"), dtype)

    # Spatial fidelity
    numel = 13
    degree = 3

    # pytest specific
    generated_plots = []

    # ========================================================================
    # Get XS information
    source = Material("source")
    source.total = torch.tensor([1], dtype=dtype, device=cpu)
    source.scatter_gtg = torch.tensor([[[0.9]]], dtype=dtype, device=cpu)
    source.finalize()

    xs_server = Server()
    xs_server.add_material(source)
    xs_server.finalize()

    # ========================================================================
    # Create NURBS patch
    length = 10  # cm
    points = np.array(
        [
            [-length / 2, -length / 2, 0],
            [length / 2, -length / 2, 0],
            [-length / 2, length / 2, 0],
            [length / 2, length / 2, 0],
        ]
    ).reshape((2, 2, -1))

    s0 = Patch.from_igakit(
        refine(bilinear(points), numel, degree),
        device=cpu,
        dtype=dtype,
        fill=source.label,
    )

    # Set the uniform fixed source
    s0.source = FixedSource(
        isotropic_strength=torch.tensor([1.0], device=cpu, dtype=dtype)
    )

    # ========================================================================
    # Create mesh
    mesh = IGAMesh(mpi_context)
    mesh.add_block(s0)
    mesh.connect()
    mesh.finalize()

    # ========================================================================
    # Plot mesh
    generated_plots.append(f"fixed_source_{request.node.name}_model.png")
    mesh.plot(
        resolution=25,
        show_ctrlpts=True,
        show_ctrlnet=True,
        show_boundary=True,
        backend="matplotlib",
        filename=generated_plots[-1],
    )

    # ========================================================================
    # Create transport driver and distribute across MPI ranks
    # Create the transport driver
    driver = IGATransportDriver2D(mesh, xs_server, mpi_context)

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
    # Send the lone system to the GPU
    if torch.cuda.is_available():
        sys = driver.get_system(s0.gid)
        sys.to_(torch.device("cuda"))

    result = driver.solve_fixed_source(
        AMEnSolver(
            nswp=4,
            eps=1e-7,
            eps_forcing=0.01,
            kickrank=4,
            local_iterations=40,
            resets=2,
            max_rank=500,
            verbose=True,
            native_opts=AMEnNativeOptions(
                enrichment_mode=AMEnEnrichmentMode.SIMPLIFIED,
                als_residual_rank=0,
                proximal_regularization=0.001,
            ),
            enrichment_policy=StaticFreezePolicy(freeze_eps=1e-4),
        ),
        tol=1e-5,
        max_iter=100,
        clear_assemblers=False,
    )

    # Remove the lone system from the GPU
    if torch.cuda.is_available():
        sys = driver.get_system(s0.gid)
        sys.to_(cpu)

    # ========================================================================
    # Plot the solution
    scalar_result = result.compute_scalar_flux()
    generated_plots.append(f"fixed_source_{request.node.name}_flux.png")
    mesh.plot(
        resolution=25,
        solution=scalar_result,
        filename=generated_plots[-1],
        field_label=r"$\phi$",
    )

    # ========================================================================
    # Load OpenMC solution and compare
    leakage_frac_mc = [0.42095701399999963, 2.2038687252709062e-05]

    # Load OpenMC scalar flux data
    data_dir = ref_dir / "data"
    phi_mc = np.load(data_dir / "mesh_flux.npy")
    phi_mc_stdev = np.load(data_dir / "mesh_stdev.npy")

    # Calculate the leakage fraction error
    gb = result.global_balance(assemblers=driver.get_assemblers(), eps=0)
    leakage_frac = (gb.leakage / gb.fixed_source).item()
    leakage_frac_error = leakage_frac - leakage_frac_mc[0]
    passed &= abs(leakage_frac_error) / leakage_frac_mc[1] < 1

    # Calculate the per patch particle balance info
    table = result.patch_balance_table(assemblers=driver.get_assemblers()).patches[0]
    loss = table.scatter_out + table.absorption
    source = table.fixed_source + table.scatter_in

    for face in table.faces:
        loss += face.outgoing
        if face.incoming != None:
            source += face.incoming

    balance = (source - loss).item() / source.item()
    passed &= abs(balance) < 1e-5

    # Average the NURBS solution onto a global regular mesh
    avg_scalar_result = (
        scalar_result.regular_mesh_average([phi_mc.shape[1], phi_mc.shape[2]], [5, 5])
        .moveaxis(-1, 0)
        .cpu()
        .numpy()
    )
    flux_tol = 0.04

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
        np.linspace(0, length, phi_mc.shape[1]),
        np.linspace(0, length, phi_mc.shape[2]),
    )

    stats = {
        name: np.zeros(xs_server.num_groups)
        for name in ["Minimum", "Q1", "Median", "Q2", "Maximum", "Mean"]
    }

    for g in range(xs_server.num_groups):
        passed &= group_errors[g] < flux_tol

        generated_plots.append(f"fixed_source_{request.node.name}_zscore_{g + 1}.png")

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

    stats["Minimum"][0] = np.min(zscores)
    stats["Q1"][0] = np.percentile(zscores, 25)
    stats["Median"][0] = np.median(zscores)
    stats["Q2"][0] = np.percentile(zscores, 75)
    stats["Maximum"][0] = np.max(zscores)
    stats["Mean"][0] = np.mean(zscores)

    # ========================================================================
    # Attach data to Pytest for conftest.py to read
    errors = {
        "df (z-score)": leakage_frac_error / leakage_frac_mc[1],
        "balance residual fraction": balance,
        "Total relative L2 error": total_error,
    }
    for name, stat in stats.items():
        errors[name] = stat[-1]

    if mpi_context.rank == 0:
        for name, error in errors.items():
            print(f"{name}: {error}")

    request.node.vnv_plots = generated_plots
    request.node.vnv_metrics = {
        "name": request.node.name,
        "metric": "Leakage fraction",
        "ttnte_val": leakage_frac,
        "ref_val": leakage_frac_mc[0],
        "passed": passed,
        "detailed_errors": errors,
    }

    # Finally, formally assert to fail the test if outside tolerance
    assert (
        abs(leakage_frac_error) / leakage_frac_mc[1] < 1
    ), "The leakage fraction is more than one standard deviation from OpenMC"
    assert (
        total_error < flux_tol
    ), "The total scalar flux error is greater than the allowed tolerance"
    assert abs(balance) < 1e-5, "Per patch residual balance is too high"
