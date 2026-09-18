import os
import random
from pathlib import Path

import pytest
from igakit import cad
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ttnte import mpi_context
from ttnte.visualization.style import get_patch_style
from ttnte.xs import Material, Server
from ttnte.cad.curves import qtrlobe
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
from ttnte.parallel import IGADofHeuristic
from ttnte.linalg import AMEnNativeOptions, AMEnEnrichmentMode
from ttnte.solvers import (
    DDSolverConfig,
    MemoryPolicy,
    AMEnSolver,
    BlockJacobiStrategy,
    ExecMode,
    CommMode,
    IGADDSolver,
    StaticFreezePolicy,
    HardFreezeWrapper,
    AdaptiveRevalidationPolicy,
)


@pytest.mark.slow
@pytest.mark.mpi(min_size=1)
def test_cruciform(request):
    passed = True

    # ========================================================================
    # Setup
    dtype = torch.float64
    cpu = torch.device("cpu")
    ref_dir = (
        Path(__file__).resolve().parent.parents[1] / "reference/fixed_source/cruciform"
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
    if mpi_context.world_size > 12:
        pytest.skip("Test requires 12 or fewer processes")

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
    qset = ProductQuadrature.gauss_legendre_chebyshev(32, 32, 2)
    qset.to_(cpu, dtype)

    # Spatial fidelity
    numel = 13
    degree = 3

    # pytest specific
    generated_plots = []

    # ========================================================================
    # Get XS information
    source = Material("Source")
    source.total = torch.tensor([0.01], dtype=dtype, device=cpu)
    source.scatter_gtg = torch.tensor([[[0.008]]], dtype=dtype, device=cpu)
    source.finalize()

    void = Material("Void")
    void.total = torch.tensor([0], dtype=dtype, device=cpu)
    void.scatter_gtg = torch.tensor([[[0]]], dtype=dtype, device=cpu)
    void.finalize()

    shield = Material("Shield")
    shield.total = torch.tensor([3], dtype=dtype, device=cpu)
    shield.scatter_gtg = torch.tensor([[[0.5]]], dtype=dtype, device=cpu)
    shield.finalize()

    xs_server = Server()
    xs_server.add_material(source)
    xs_server.add_material(void)
    xs_server.add_material(shield)
    xs_server.finalize()

    # ========================================================================
    # Create NURBS patch

    ## Initialize dimensional variables
    X = 10  # Channel pitch

    # Cruciform
    R = 2  # Radius defining valleys of fixed source
    delta = 1  # Width of lobes
    d2 = delta * 0.5  # Half width of lobes
    x = 0.25  # Portrusion of lobes

    # Shielding
    I = 3.75  # Inner radius
    O = 4.5  # Outer radius

    # NURBS curves
    origin = cad.line(p0=(0, 0), p1=(0, 0))
    cruciform = qtrlobe(outrad=R, portrs=x, hfwidth=d2)
    circleI = cad.circle(radius=I, angle=[np.pi / 2, 0])
    circleO = cad.circle(radius=O, angle=[np.pi / 2, 0])
    topedge = cad.line(p0=(0, X / 2), p1=(X / 2, X / 2))
    corner = cad.line(p1=(X / 2, X / 2), p0=(X / 2, X / 2))
    rightedge = cad.line(p1=(X / 2, 0), p0=(X / 2, X / 2))

    # Create and add NURBS surfaces
    sections = [0, 1 / 3, 2 / 3, 1]
    edges = [topedge, corner, rightedge]

    # Create mesh
    mesh = IGAMesh(mpi_context)

    for i in range(len(sections) - 1):
        # Line sections
        csec = origin.slice(0, sections[i], sections[i + 1])
        ssec = cruciform.slice(0, sections[i], sections[i + 1])
        isec = circleI.slice(0, sections[i], sections[i + 1])
        osec = circleO.slice(0, sections[i], sections[i + 1])

        # Create source patch
        s_source = Patch.from_igakit(
            cad.refine(cad.ruled(csec, ssec), numel, degree),
            device=cpu,
            dtype=dtype,
            fill=source.label,
        )
        s_source.source = FixedSource(
            isotropic_strength=torch.tensor([1.0], device=cpu, dtype=dtype)
        )
        mesh.add_block(s_source)

        # First Void patch
        mesh.add_block(
            Patch.from_igakit(
                cad.refine(cad.ruled(ssec, isec), numel, degree),
                device=cpu,
                dtype=dtype,
                fill=void.label,
            )
        )

        # Shield patch
        mesh.add_block(
            Patch.from_igakit(
                cad.refine(cad.ruled(isec, osec), numel, degree),
                device=cpu,
                dtype=dtype,
                fill=shield.label,
            )
        )

        # Second Void patch
        mesh.add_block(
            Patch.from_igakit(
                cad.refine(cad.ruled(osec, edges[i]), numel, degree),
                device=cpu,
                dtype=dtype,
                fill=void.label,
            )
        )

    # Connect patches
    mesh.connect()

    # Set the boundary conditions
    mesh.set_axis_aligned_conditions(
        BCPlane(x_min=True, y_min=True),
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
        source.label.to_string(): "maroon",
        void.label.to_string(): "cornflowerblue",
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

        if assembler.scatter_op.defined():
            op = assembler.scatter_op.as_tt()
            string += f"S: Ranks = {op.ranks}, CR = {op.compression}\n"

        for op in assembler.inflow_ops:
            if op.defined():
                op = op.as_tt()
                string += f"Bin: Ranks = {op.ranks}, CR = {op.compression}\n"
        for op in assembler.outflow_ops:
            if op.defined():
                op = op.as_tt()
                string += f"Bout: Ranks = {op.ranks}, CR = {op.compression}\n"

        print(string, end="")

    # ========================================================================
    # Run DD solver
    outer_tol = 1e-4
    inner_tol = 5e-5
    eps = 1e-6

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
            nswp=1,
            eps=eps,
            eps_forcing=0.1,
            kickrank=4,
            local_iterations=1000,
            resets=10,
            native_opts=AMEnNativeOptions(
                enrichment_mode=AMEnEnrichmentMode.FULL,
                als_residual_rank=0,
                proximal_regularization=0.01,
                gmres_mixed_precision=True,
            ),
            enrichment_policy=StaticFreezePolicy(freeze_eps=1e-5),
        )
    )
    dd_solver = IGADDSolver(driver.mesh, strategy)

    result = driver.solve_fixed_source(
        dd_solver, tol=outer_tol, max_iter=100, verbose=True
    )

    # ========================================================================
    # Plot the solution
    scalar_result = result.compute_scalar_flux()
    generated_plots.append(f"fixed_source_{request.node.name}_flux.png")
    mesh.plot(
        resolution=25,
        solution=scalar_result,
        filename=generated_plots[-1],
        field_label=r"$\phi$",
        gather=True,
    )

    # ========================================================================
    # Load OpenMC solution and compare
    leakage_frac_mc = [0.06913173400000001, 1.1401809264552177e-05]

    # Load OpenMC scalar flux data
    data_dir = ref_dir / "data"
    phi_mc = np.load(data_dir / "mesh_flux.npy")
    phi_mc_stdev = np.load(data_dir / "mesh_stdev.npy")

    # Calculate the eigenvalue error
    gb = result.global_balance(assemblers=driver.get_assemblers(), eps=0)
    leakage_frac = (gb.leakage / gb.fixed_source).item()
    leakage_frac_error = leakage_frac - leakage_frac_mc[0]
    passed &= abs(leakage_frac_error) / leakage_frac_mc[1] < 3

    # Calculate the per patch particle balance info
    tables = result.patch_balance_table(assemblers=driver.get_assemblers())
    balances = []

    for table in tables.patches:
        loss = table.scatter_out + table.absorption
        source = table.fixed_source + table.scatter_in

        for face in table.faces:
            loss += face.outgoing
            if face.incoming is not None:
                source += face.incoming

        balance = (source - loss).item() / source.item()
        passed &= abs(balance) < 5e-4
        balances.append(balance)

    # Average the NURBS solution onto a global regular mesh
    avg_scalar_result = (
        scalar_result.regular_mesh_average(
            [phi_mc.shape[1], phi_mc.shape[2]],
            [2, 2],
            seed_resolution=60,
            tol=0.02,
        )
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
        np.linspace(0, X / 2, phi_mc.shape[1]),
        np.linspace(0, X / 2, phi_mc.shape[2]),
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
        "Total relative L2 error": total_error,
    }

    for i in range(len(balances)):
        table = tables.patches[i]
        errors[f"balance residual fraction (gid={table.gid})"] = balances[i]

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
        abs(leakage_frac_error) / leakage_frac_mc[1] < 3
    ), "The leakage fraction is more than one standard deviation from OpenMC"
    assert (
        total_error < flux_tol
    ), "The total scalar flux error is greater than the allowed tolerance"
    assert (
        np.abs(np.array(balances)) < 5e-4
    ).all(), "Per patch residual balance is too high"
