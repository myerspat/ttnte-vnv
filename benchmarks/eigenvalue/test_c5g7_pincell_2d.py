from pathlib import Path

import pytest
import torch
import numpy as np
from igakit import cad

from ttnte import mpi_context
from ttnte.cad import Patch
from ttnte.mesh import IGAMesh
from ttnte.physics import BoundaryType, BCPlane
from ttnte.math import ProductQuadrature
from ttnte.visualization.style import get_patch_style
from ttnte.xs.benchmarks import c5g7

from vnv import flux_errors, record, setup, zscores

# Solver settings for this problem, merged over vnv.solvers.DEFAULTS
# (--solver-opt on the command line wins over these)
OPTS = {
    "amen": {
        "tol": 1e-6,
        "k_tol": 1e-6,
        "max_iter": 100,
        "keep_assemblers": True,  # operator rank printout below
        "dd": {"tol": 5e-7, "tol_forcing": 0.5, "max_iter": 100},
        "local": {
            "nswp": 1,
            "eps": 1e-7,
            "eps_forcing": 0.1,
            "kickrank": 4,
            "local_iterations": 250,
            "resets": 4,
            "max_rank": 2**31 - 1,
        },
        "native": {
            "enrichment_mode": "FULL",
            "als_residual_rank": 0,
            "proximal_regularization": 0.01,
            "gmres_mixed_precision": True,
        },
        "freeze_eps": 1e-4,
    },
    "sweep": {"tol": 1e-5, "num_threads": 2},
    "cross": {"tol": 1e-5, "num_threads": 2, "cross": {"num_threads": 2}},
}


@pytest.mark.slow
@pytest.mark.mpi(min_size=1)
def test_c5g7_pincell_2d(request, solver):
    # ========================================================================
    # Setup
    setup(max_ranks=3)
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
    # Solve
    result = solver.run(mesh, xs_server, qset, eigenvalue=True, opts=OPTS)

    # Operator ranks and compression of this rank's patches (TT operators only)
    if solver.name == "amen":
        for patch in result.driver.mesh.blocks:
            assembler = result.driver.get_assembler(patch.gid)

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
    # Plot the solution
    for g in range(xs_server.num_groups):
        generated_plots.append(f"{request.node.name}_flux_{g + 1}.png")
        mesh.plot(
            resolution=25,
            solution=result.scalar.select_group(g),
            filename=generated_plots[-1],
            field_label=rf"$\phi_{g + 1}$",
            gather=True,
        )

    # ========================================================================
    # Load OpenMC solution and compare
    k_mc = [1.325593334007463, 3.229591167123522e-05]

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

    # Average the NURBS solution onto a global regular mesh, normalize it and
    # compare
    flux_tol = 0.04
    avg_scalar_result, group_errors, total_error = flux_errors(
        result.scalar, phi_mc, normalize=True
    )
    passed &= total_error < flux_tol
    passed &= bool(np.all(group_errors < flux_tol))

    # Compute and plot z-score per energy group
    stats, plots = zscores(
        avg_scalar_result,
        phi_mc,
        phi_mc_stdev,
        extent=(0, pitch / 2, 0, pitch / 2),
        prefix=request.node.name,
    )
    generated_plots += plots

    # ========================================================================
    # Attach data to Pytest for conftest.py to read
    errors = {
        "dk (pcm)": keff_error,
        "dk (z-score)": abs(keff_error) / (k_mc[1] * 1e5),
        "Total relative L2 error": total_error,
    }
    for g in range(xs_server.num_groups):
        errors[f"Relative relative L2 error (g = {g + 1})"] = group_errors[g]
    errors.update(stats)
    record(
        request,
        result,
        value=result.k_eff,
        ref=k_mc[0],
        passed=passed,
        errors=errors,
        plots=generated_plots,
    )

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
