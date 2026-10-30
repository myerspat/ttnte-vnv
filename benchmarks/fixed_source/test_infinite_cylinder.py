from pathlib import Path

import numpy as np
import pytest
import torch
from igakit import cad
from ttnte import mpi_context
from ttnte.cad import Patch
from ttnte.cad.surfaces import circle
from ttnte.math import ProductQuadrature
from ttnte.mesh import IGAMesh
from ttnte.physics import BCPlane, BoundaryType, FixedSource
from ttnte.visualization.style import get_patch_style
from ttnte.xs import Material, Server

from vnv import flux_errors, record, setup, zscores

# Solver settings for this problem, merged over vnv.solvers.DEFAULTS
# (--solver-opt on the command line wins over these)
OPTS = {
    "amen": {
        "tol": 1e-5,
        "max_iter": 100,
        "local": {
            "nswp": 1,
            "eps": 5e-7,
            "eps_forcing": 0.1,
            "kickrank": 4,
            "local_iterations": 250,
            "resets": 4,
            "max_rank": 500,
        },
        "native": {
            "enrichment_mode": "FULL",
            "als_residual_rank": 0,
            "proximal_regularization": 0.01,
            "gmres_mixed_precision": True,
        },
        "freeze_eps": 1e-6,
    },
    "sweep": {"tol": 1e-5, "num_threads": 2},
    "cross": {"tol": 1e-5, "num_threads": 2, "cross": {"num_threads": 2}},
}


@pytest.mark.mpi(min_size=1)
def test_infinite_cylinder(request, solver):
    # ========================================================================
    # Setup
    setup(max_ranks=1)
    dtype = torch.float64
    cpu = torch.device("cpu")

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
    source = Material("source")
    source.total = torch.tensor([1], dtype=dtype, device=cpu)
    source.scatter_gtg = torch.tensor([[[0.9]]], dtype=dtype, device=cpu)
    source.finalize()

    xs_server = Server()
    xs_server.add_material(source)
    xs_server.finalize()

    # ========================================================================
    # Create NURBS patch
    rc = 5  # cm
    s0 = Patch.from_igakit(
        cad.refine(circle(rc), numel, degree),
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
    # Solve (balance=True keeps what the particle balances below need)
    result = solver.run(
        mesh, xs_server, qset, eigenvalue=False, balance=True, opts=OPTS
    )

    # ========================================================================
    # Plot the solution
    generated_plots.append(f"fixed_source_{request.node.name}_flux.png")
    mesh.plot(
        resolution=25,
        solution=result.scalar,
        filename=generated_plots[-1],
        field_label=r"$\phi$",
    )

    # ========================================================================
    # Check leakage fraction with OpenMC
    leakage_frac_mc = [0.43995423399999983, 2.2245143201699137e-05]

    # Calculate the leakage fraction error
    gb = result.global_balance(eps=0)
    leakage_frac = (gb.leakage / gb.fixed_source).item()
    leakage_frac_error = leakage_frac - leakage_frac_mc[0]
    passed = abs(leakage_frac_error) / leakage_frac_mc[1] < 3

    # Calculate the per patch particle balance info
    table = result.patch_balance_table().patches[0]
    loss = table.scatter_out + table.absorption
    source = table.fixed_source + table.scatter_in

    for face in table.faces:
        loss += face.outgoing
        if face.incoming is not None:
            source += face.incoming

    balance = (source - loss).item() / source.item()
    passed &= abs(balance) < 1e-5

    # ========================================================================
    # Attach data to Pytest for conftest.py to read
    errors = {
        "df (z-score)": leakage_frac_error / leakage_frac_mc[1],
        "balance residual fraction": balance,
    }
    record(
        request,
        result,
        metric="Leakage fraction",
        value=leakage_frac,
        ref=leakage_frac_mc[0],
        passed=passed,
        errors=errors,
        plots=generated_plots,
    )

    # Finally, formally assert to fail the test if outside tolerance
    assert (
        abs(leakage_frac_error) / leakage_frac_mc[1] < 3
    ), "The leakage fraction is more than one standard deviation from OpenMC"
    assert abs(balance) < 1e-5, "Per patch residual balance is too high"


# Solver settings for the 4-patch problem, merged over vnv.solvers.DEFAULTS
# (--solver-opt on the command line wins over these)
OPTS_4PATCH = {
    "amen": {
        "tol": 1e-5,
        "max_iter": 100,
        "dd": {"tol": 5e-6, "tol_forcing": 0.5, "max_iter": 100},
        "local": {
            "nswp": 1,
            "eps": 5e-7,
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
        "freeze_eps": 1e-6,
    },
    "sweep": {"tol": 1e-7, "num_threads": 1},
    "cross": {"tol": 1e-7, "num_threads": 1, "cross": {"num_threads": 1}},
}


@pytest.mark.mpi(min_size=1)
def test_infinite_cylinder_4patch(request, solver):
    # ========================================================================
    # Setup
    setup(max_ranks=4)
    dtype = torch.float64
    cpu = torch.device("cpu")

    ref_dir = (
        Path(__file__).resolve().parent.parents[1]
        / "reference/fixed_source/infinite_cylinder"
    )

    # Check the reference data exists
    assert (
        ref_dir.exists() and (ref_dir / "data").exists()
    ), f"Reference data directory not found at {ref_dir / 'data'}"

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
    source.total = torch.tensor([1], dtype=dtype, device=cpu)
    source.scatter_gtg = torch.tensor([[[0.9]]], dtype=dtype, device=cpu)
    source.finalize()

    void = Material("Void")
    void.total = torch.tensor([0], dtype=dtype, device=cpu)
    void.scatter_gtg = torch.tensor([[[0]]], dtype=dtype, device=cpu)
    void.finalize()

    xs_server = Server()
    xs_server.add_material(source)
    xs_server.add_material(void)
    xs_server.finalize()

    # ========================================================================
    # Create NURBS patch
    radius = 5  # cm
    pitch = 12  # cm
    c0 = cad.circle(radius=radius, angle=torch.pi / 2)
    c1 = c0.slice(0, 0, 0.5)
    c2 = c0.slice(0, 0.5, 1)
    l0 = cad.line(p0=(0, 0), p1=(0, 0))

    # Create water patch
    l1 = cad.line(p0=(pitch / 2, 0), p1=(pitch / 2, pitch / 2))
    l2 = cad.line(p0=(pitch / 2, pitch / 2), p1=(0, pitch / 2))

    # Create NURBS surfaces
    s_void = [cad.ruled(c1, l1), cad.ruled(c2, l2)]
    s_source = [cad.ruled(l0, c1), cad.ruled(l0, c2)]

    # ========================================================================
    # Create mesh
    mesh = IGAMesh(mpi_context)

    for fill, surface in zip(2 * [source.label] + 2 * [void.label], s_source + s_void):
        surface = Patch.from_igakit(
            cad.refine(surface, numel, degree), device=cpu, dtype=dtype, fill=fill
        )

        if fill == source.label:
            # Set the uniform fixed source
            surface.source = FixedSource(
                isotropic_strength=torch.tensor([1.0], device=cpu, dtype=dtype)
            )

        mesh.add_block(surface)

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
    # Solve (balance=True keeps what the particle balances below need)
    result = solver.run(
        mesh, xs_server, qset, eigenvalue=False, balance=True, opts=OPTS_4PATCH
    )

    # ========================================================================
    # Plot the solution
    generated_plots.append(f"fixed_source_{request.node.name}_flux.png")
    mesh.plot(
        resolution=25,
        solution=result.scalar,
        filename=generated_plots[-1],
        field_label=r"$\phi$",
        gather=True,
    )

    # ========================================================================
    # Load OpenMC solution and compare
    leakage_frac_mc = [0.43995486599999994, 2.2442114486922458e-05]

    # Load OpenMC scalar flux data
    data_dir = ref_dir / "data"
    phi_mc = np.load(data_dir / "mesh_flux.npy")
    phi_mc_stdev = np.load(data_dir / "mesh_stdev.npy")

    # Calculate the leakage fraction error
    gb = result.global_balance(eps=0)
    leakage_frac = (gb.leakage / gb.fixed_source).item()
    leakage_frac_error = leakage_frac - leakage_frac_mc[0]
    passed = abs(leakage_frac_error) / leakage_frac_mc[1] < 3

    # Calculate the per patch particle balance info
    balances = {}
    for table in result.patch_balance_table().patches:
        loss = table.scatter_out + table.absorption
        source = table.fixed_source + table.scatter_in

        for face in table.faces:
            loss += face.outgoing
            if face.incoming is not None:
                source += face.incoming

        balances[table.gid] = (source - loss).item() / source.item()
    passed &= all(abs(b) < 1e-5 for b in balances.values())

    # Average the NURBS solution onto a global regular mesh and compare
    flux_tol = 0.04
    avg_scalar_result, group_errors, total_error = flux_errors(
        result.scalar, phi_mc, n=(2, 2), seed_resolution=60, tol=0.02
    )
    passed &= total_error < flux_tol
    passed &= bool(np.all(group_errors < flux_tol))

    # Compute and plot z-score per energy group
    stats, plots = zscores(
        avg_scalar_result,
        phi_mc,
        phi_mc_stdev,
        extent=(0, pitch / 2, 0, pitch / 2),
        prefix=f"fixed_source_{request.node.name}",
    )
    generated_plots += plots

    # ========================================================================
    # Attach data to Pytest for conftest.py to read
    errors = {
        "df (z-score)": leakage_frac_error / leakage_frac_mc[1],
        "Total relative L2 error": total_error,
    }
    for gid, balance in balances.items():
        errors[f"balance residual fraction (gid={gid})"] = balance
    errors.update(stats)
    record(
        request,
        result,
        metric="Leakage fraction",
        value=leakage_frac,
        ref=leakage_frac_mc[0],
        passed=passed,
        errors=errors,
        plots=generated_plots,
    )

    # Finally, formally assert to fail the test if outside tolerance
    assert (
        abs(leakage_frac_error) / leakage_frac_mc[1] < 3
    ), "The leakage fraction is more than three standard deviations from OpenMC"
    assert (
        total_error < flux_tol
    ), "The total scalar flux error is greater than the allowed tolerance"
    assert all(
        abs(b) < 1e-5 for b in balances.values()
    ), "Per patch residual balance is too high"
