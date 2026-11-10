from pathlib import Path

import pytest
from igakit import cad
import torch
import numpy as np

from ttnte import mpi_context
from ttnte.visualization.style import get_patch_style
from ttnte.xs import Material, Server
from ttnte.cad.curves import qtrlobe
from ttnte.cad import Patch
from ttnte.mesh import IGAMesh
from ttnte.physics import BoundaryType, BCPlane, FixedSource
from ttnte.math import ProductQuadrature

from vnv import flux_errors, record, setup, zscores

# Solver settings for this problem, merged over vnv.solvers.DEFAULTS
# (--solver-opt on the command line wins over these)
OPTS = {
    "amen": {
        "tol": 1e-4,
        "max_iter": 100,
        "dd": {"tol": 5e-5, "tol_forcing": 0.5, "max_iter": 100},
        "local": {
            "nswp": 1,
            "eps": 1e-6,
            "eps_forcing": 0.1,
            "kickrank": 4,
            "local_iterations": 1000,
            "resets": 10,
            "max_rank": 2**31 - 1,
        },
        "native": {
            "enrichment_mode": "FULL",
            "als_residual_rank": 0,
            "proximal_regularization": 0.01,
            "gmres_mixed_precision": True,
        },
        "freeze_eps": 1e-5,
    },
    "sweep": {"tol": 1e-5, "num_threads": 2},
    "cross": {"tol": 1e-5, "num_threads": 2, "cross_threads": 2},
}


@pytest.mark.slow
@pytest.mark.mpi(min_size=1)
def test_cruciform(request, solver):
    # ========================================================================
    # Setup
    setup(max_ranks=12)
    dtype = torch.float64
    cpu = torch.device("cpu")
    ref_dir = (
        Path(__file__).resolve().parent.parents[1] / "reference/fixed_source/cruciform"
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
    # Solve (balance=True keeps what the particle balances below need)
    result = solver.run(
        mesh, xs_server, qset, eigenvalue=False, balance=True, opts=OPTS
    )

    # Operator ranks and compression of this rank's patches (TT operators only)
    if solver.name == "amen":
        for patch in result.driver.mesh.blocks:
            assembler = result.driver.get_assembler(patch.gid)

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
    leakage_frac_mc = [0.06913173400000001, 1.1401809264552177e-05]

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
    passed &= all(abs(b) < 5e-4 for b in balances.values())

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
        extent=(0, X / 2, 0, X / 2),
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
        abs(b) < 5e-4 for b in balances.values()
    ), "Per patch residual balance is too high"
