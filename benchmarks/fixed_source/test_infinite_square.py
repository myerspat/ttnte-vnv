from pathlib import Path

import pytest

import numpy as np
import torch
from igakit.cad import bilinear, refine
from ttnte import mpi_context
from ttnte.cad import Patch
from ttnte.math import ProductQuadrature
from ttnte.mesh import IGAMesh
from ttnte.physics import FixedSource
from ttnte.xs import Material, Server

from vnv import flux_errors, record, setup, zscores

# Solver settings for this problem, merged over vnv.solvers.DEFAULTS
# (--solver-opt on the command line wins over these)
OPTS = {
    "amen": {
        "tol": 1e-5,
        "max_iter": 100,
        "local": {
            "nswp": 4,
            "eps": 1e-7,
            "eps_forcing": 0.01,
            "kickrank": 4,
            "local_iterations": 40,
            "resets": 2,
            "max_rank": 500,
        },
        "native": {
            "enrichment_mode": "SIMPLIFIED",
            "als_residual_rank": 0,
            "proximal_regularization": 0.001,
        },
        "freeze_eps": 1e-4,
    },
    "sweep": {"tol": 1e-5, "num_threads": 2},
    "cross": {"tol": 1e-5, "num_threads": 2, "cross": {"num_threads": 2}},
}


@pytest.mark.mpi(min_size=1)
def test_infinite_square(request, solver):
    # ========================================================================
    # Setup
    setup(max_ranks=1)
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
    # Load OpenMC solution and compare
    leakage_frac_mc = [0.42095701399999963, 2.2038687252709062e-05]

    # Load OpenMC scalar flux data
    data_dir = ref_dir / "data"
    phi_mc = np.load(data_dir / "mesh_flux.npy")
    phi_mc_stdev = np.load(data_dir / "mesh_stdev.npy")

    # Calculate the leakage fraction error
    gb = result.global_balance(eps=0)
    leakage_frac = (gb.leakage / gb.fixed_source).item()
    leakage_frac_error = leakage_frac - leakage_frac_mc[0]
    passed = abs(leakage_frac_error) / leakage_frac_mc[1] < 1

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

    # Average the NURBS solution onto a global regular mesh and compare
    flux_tol = 0.04
    avg_scalar_result, group_errors, total_error = flux_errors(
        result.scalar, phi_mc, seed_resolution=10
    )
    passed &= total_error < flux_tol
    passed &= bool(np.all(group_errors < flux_tol))

    # Compute and plot z-score per energy group
    stats, plots = zscores(
        avg_scalar_result,
        phi_mc,
        phi_mc_stdev,
        extent=(0, length, 0, length),
        prefix=f"fixed_source_{request.node.name}",
    )
    generated_plots += plots

    # ========================================================================
    # Attach data to Pytest for conftest.py to read
    errors = {
        "df (z-score)": leakage_frac_error / leakage_frac_mc[1],
        "balance residual fraction": balance,
        "Total relative L2 error": total_error,
    }
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
        abs(leakage_frac_error) / leakage_frac_mc[1] < 1
    ), "The leakage fraction is more than one standard deviation from OpenMC"
    assert (
        total_error < flux_tol
    ), "The total scalar flux error is greater than the allowed tolerance"
    assert abs(balance) < 1e-5, "Per patch residual balance is too high"
