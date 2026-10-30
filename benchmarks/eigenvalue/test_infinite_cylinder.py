import pytest
from igakit.cad import refine
import torch
import numpy as np
import matplotlib.pyplot as plt

from ttnte import mpi_context
from ttnte.cad.surfaces import circle
from ttnte.xs.benchmarks import pu239
from ttnte.cad import Patch
from ttnte.mesh import IGAMesh
from ttnte.math import ProductQuadrature

from vnv import record, setup

# Solver settings for this problem, merged over vnv.solvers.DEFAULTS
# (--solver-opt on the command line wins over these)
OPTS = {
    "amen": {
        "tol": 1e-6,
        "max_iter": 500,
        "local": {
            "nswp": 4,
            "eps": 8e-5,
            "eps_forcing": 0.01,
            "kickrank": 4,
            "local_iterations": 100,
            "resets": 4,
            "max_rank": 500,
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


def evaluate_radius(patch, field, r, dtype, max_iter=10, tol=1e-8):
    # Convert to a dense tensor
    dense_field = field.to_dense().reshape(
        patch.get_ctrlpts_size(0), patch.get_ctrlpts_size(1), -1
    )

    # Compute the flux at the center
    center_flux = patch.evaluate_field(
        dense_field, torch.tensor([[0.5, 0.5]])
    ).flatten()

    # Calculate physical locations
    points = r * torch.ones((400, 2), dtype=dtype)
    angular_points = torch.zeros((400, 2))
    angular_points[:, 0] = torch.linspace(-torch.pi, torch.pi, 400)
    points[:, 0] *= torch.cos(angular_points[:, 0])
    points[:, 1] *= torch.sin(angular_points[:, 0])

    # Compute an inverse map
    result = patch.inverse_map(points, max_iter=max_iter, tol=tol)
    assert result.converged.all()
    angular_points[:, 1] = (
        patch.evaluate_field(dense_field, result.coords).flatten() / center_flux
    )

    return angular_points


def evaluate_boundary(patch, field, rc, dtype):
    # Convert to a dense tensor
    dense_field = field.to_dense().reshape(
        patch.get_ctrlpts_size(0), patch.get_ctrlpts_size(1), -1
    )

    # Plot and evaluate boundary flux
    center_flux = patch.evaluate_field(
        dense_field, torch.tensor([[0.5, 0.5]])
    ).flatten()
    points = torch.zeros((400, 2), dtype=dtype)
    points[:100, 0] = torch.linspace(0, 1, 100, dtype=dtype)
    points[100:200, 0] = 1
    points[100:200, 1] = torch.linspace(0, 1, 100, dtype=dtype)
    points[200:300, 0] = torch.linspace(0, 1, 100, dtype=dtype).flip(0).contiguous()
    points[200:300, 1] = 1
    points[300:400, 1] = torch.linspace(0, 1, 100, dtype=dtype).flip(0).contiguous()

    points = torch.cat(
        (patch(points), patch.evaluate_field(dense_field, points)), dim=-1
    )
    points[:, -1] /= center_flux

    # Convert to angle
    angular_points = torch.zeros((points.shape[0], 2))
    angular_points[:, -1] = points[:, -1]
    angular_points[(points[:, 0] >= 0), 0] = torch.arcsin(
        points[(points[:, 0] >= 0), 1] / rc
    )
    angular_points[(points[:, 0] < 0) & (points[:, 1] >= 0), 0] = (
        -torch.arcsin(points[(points[:, 0] < 0) & (points[:, 1] >= 0), 1] / rc)
        + torch.pi
    )
    angular_points[(points[:, 0] < 0) & (points[:, 1] < 0), 0] = (
        -torch.arcsin(points[(points[:, 0] < 0) & (points[:, 1] < 0), 1] / rc)
        - torch.pi
    )
    return angular_points[angular_points[:, 0].argsort()]


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
    numel = 10
    degree = 4

    # pytest specific
    generated_plots = []

    # ========================================================================
    # Get XS information
    fills, xs_server = pu239(num_groups=1, device=cpu, dtype=dtype)

    # ========================================================================
    # Create NURBS patch
    rc = 4.279960
    s0 = Patch.from_igakit(
        refine(circle(rc), numel, degree), device=cpu, dtype=dtype, fill=fills[0]
    )

    # ========================================================================
    # Create mesh
    mesh = IGAMesh(mpi_context)
    mesh.add_block(s0)
    mesh.connect()
    mesh.finalize()

    # ========================================================================
    # Plot mesh
    generated_plots.append(f"{request.node.name}_model.png")
    mesh.plot(
        resolution=25,
        show_ctrlpts=True,
        show_ctrlnet=True,
        show_boundary=True,
        backend="matplotlib",
        filename=generated_plots[-1],
    )

    # ========================================================================
    # Solve
    result = solver.run(mesh, xs_server, qset, eigenvalue=True, opts=OPTS)

    # ========================================================================
    # Plot the solution
    scalar_result = result.scalar
    generated_plots.append(f"{request.node.name}_flux.png")
    mesh.plot(
        resolution=25,
        solution=scalar_result.select_group(0),
        filename=generated_plots[-1],
        field_label=r"$\phi$",
    )

    # ========================================================================
    # Check the solution to the reference analytical solution
    keff_error = (result.k_eff - 1.0) * 1e5
    passed = abs(keff_error) < 5

    # Calculate the field along the boundary
    expected = 0.2926
    points = evaluate_boundary(s0, scalar_result.get_local_field(s0.gid), rc, dtype)

    # Plot the points
    plt.clf()
    plt.plot(points[:, 0] + np.pi, (points[:, 1] - expected) / expected)
    plt.ylabel(r"$\delta\phi(r = r_c, \theta)$")
    plt.xlabel(r"$\theta$")
    plt.xticks(
        [
            0,
            np.pi / 4,
            np.pi / 2,
            3 * np.pi / 4,
            np.pi,
            5 * np.pi / 4,
            3 * np.pi / 2,
            7 * np.pi / 4,
            2 * np.pi,
        ],
        [
            "$0$",
            r"$\frac{\pi}{4}$",
            r"$\frac{\pi}{2}$",
            r"$\frac{3\pi}{4}$",
            r"$\pi$",
            r"$\frac{5\pi}{4}$",
            r"$\frac{3\pi}{2}$",
            r"$\frac{7\pi}{4}$",
            r"$2\pi$",
        ],
    )
    plt.grid()
    plt.tight_layout()
    generated_plots.append(f"{request.node.name}_rc.png")
    plt.savefig(generated_plots[-1], dpi=300)

    # Check the L2-error is less than 0.008
    error_rc = np.sqrt(
        np.trapz((points[:, 1] - expected) ** 2, points[:, 0])
    ) / np.sqrt(2 * np.pi * expected**2)
    passed &= error_rc < 0.008

    # Calculate the field along the 0.5rc circle
    expected = 0.8093
    points = evaluate_radius(s0, scalar_result.get_local_field(s0.gid), 0.5 * rc, dtype)

    # Plot the points
    plt.clf()
    plt.plot(points[:, 0] + np.pi, (points[:, 1] - expected) / expected)
    plt.ylabel(r"$\delta\phi(r = 0.5r_c, \theta)$")
    plt.xlabel(r"$\theta$")
    plt.xticks(
        [
            0,
            np.pi / 4,
            np.pi / 2,
            3 * np.pi / 4,
            np.pi,
            5 * np.pi / 4,
            3 * np.pi / 2,
            7 * np.pi / 4,
            2 * np.pi,
        ],
        [
            "$0$",
            r"$\frac{\pi}{4}$",
            r"$\frac{\pi}{2}$",
            r"$\frac{3\pi}{4}$",
            r"$\pi$",
            r"$\frac{5\pi}{4}$",
            r"$\frac{3\pi}{2}$",
            r"$\frac{7\pi}{4}$",
            r"$2\pi$",
        ],
    )
    plt.grid()
    plt.tight_layout()
    generated_plots.append(f"{request.node.name}_rc0.5.png")
    plt.savefig(generated_plots[-1], dpi=300)

    # Check the L2-error is less than 0.0002
    error_half_rc = np.sqrt(
        np.trapz((points[:, 1] - expected) ** 2, points[:, 0])
    ) / np.sqrt(2 * np.pi * expected**2)
    passed &= error_half_rc < 0.0002

    # ========================================================================
    # Attach data to Pytest for conftest.py to read
    errors = {
        "dk (pcm)": keff_error,
        "Relative L2 error (rc)": error_rc,
        "Relative L2 error (0.5rc)": error_half_rc,
    }
    record(
        request,
        result,
        value=result.k_eff,
        ref=1.0,
        passed=passed,
        errors=errors,
        plots=generated_plots,
    )

    # Finally, formally assert to fail the test if outside tolerance
    assert (
        abs(keff_error) < 5
    ), "The eigenvalue was more than 5 pcm from the true solution"
    assert error_rc < 0.008, "Scalar flux at radius rc exceeds the allowed tolerance"
    assert (
        error_half_rc < 0.0002
    ), "Scalar flux at radius 0.5 * rc exceeds the allowed tolerance"
