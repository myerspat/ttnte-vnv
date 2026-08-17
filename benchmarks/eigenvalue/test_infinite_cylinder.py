import os

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
from ttnte.physics import (
    BoundaryType,
    BCPlane,
    DGTransportAssemblerConfig,
)
from ttnte.math import ProductQuadrature
from ttnte.driver import IGATransportDriver2D
from ttnte.solvers import (
    DDSolverConfig,
    MemoryPolicy,
    AMEnSolver,
    BlockJacobiStrategy,
    StaticFreezePolicy,
    ExecMode,
    CommMode,
)
from ttnte.linalg import AMEnNativeOptions, AMEnEnrichmentMode


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


def test_infinite_cylinder(request):
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
    if mpi_context.world_size > 1:
        pytest.skip("Test requires 1 or fewer processes")

    # Set defaults for PyTorch
    torch.set_default_dtype(dtype)
    torch.autograd.set_grad_enabled(False)

    # Create angular quadrature
    qset = ProductQuadrature.gauss_legendre_chebyshev(32, 32, 2)
    qset.to_(torch.device("cpu"), dtype)

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

    result = driver.solve_eigenvalue(
        AMEnSolver(
            nswp=4,
            eps=8e-5,
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
        ),
        tol=1e-6,
        max_iter=500,
        verbose=True,
    )

    # Remove the lone system from the GPU
    if torch.cuda.is_available():
        sys = driver.get_system(s0.gid)
        sys.to_(cpu)

    # ========================================================================
    # Plot the solution
    # Compute the scalar flux
    scalar_result = result.compute_scalar_flux()
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
    passed &= abs(keff_error) < 5

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

    # ========================================================================
    # Attach data to Pytest for conftest.py to read
    errors = {
        "dk (pcm)": keff_error,
        "Relative L2 error (rc)": error_rc,
        "Relative L2 error (0.5rc)": error_half_rc,
    }
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
    assert error_rc < 0.008, "Scalar flux at radius rc exceeds the allowed tolerance"
    assert (
        error_half_rc < 0.0002
    ), "Scalar flux at radius 0.5 * rc exceeds the allowed tolerance"
