import os

from igakit import cad
import pytest
import torch
import numpy as np
import matplotlib.pyplot as plt

from ttnte import mpi_context
from ttnte.xs.benchmarks import pu239
from ttnte.visualization.style import get_patch_style
from ttnte.visualization.window import export_pv_plot
from ttnte.cad import Patch
from ttnte.mesh import IGAMesh
from ttnte.physics import (
    BoundaryType,
    BCPlane,
    DGTransportAssemblerConfig,
)
from ttnte.math import ProductQuadrature
from ttnte.driver import IGATransportDriver3D
from ttnte.solvers import (
    DDSolverConfig,
    IGADDSolver,
    MemoryPolicy,
    AMEnSolver,
    BlockJacobiStrategy,
    ExecMode,
    CommMode,
)
from ttnte.parallel import IGADofHeuristic


def _octant_shell_ctrl(radius):
    """Homogeneous control net for one azimuth x elevation octant layer of a
    sphere's surface at the given radius (exact rational biquadratic: V sweeps
    the X-axis to the Y-axis, W sweeps the XY-plane to the Z-axis, each a 90
    degree circular arc with the standard weight sqrt(2)/2 at its midpoint).

    Returns
    -------
    ctrl: numpy.ndarray, shape (3, 3, 4)
        Homogeneous [x*w, y*w, z*w, w] control points, indexed [v, w].
    """
    s2 = np.sqrt(2.0)
    ctrl = np.zeros((3, 3, 4))

    # W = 0 (Equator / XY plane)
    ctrl[0, 0] = [radius, 0, 0, 1.0]
    ctrl[1, 0] = [radius * s2 / 2, radius * s2 / 2, 0, s2 / 2]
    ctrl[2, 0] = [0, radius, 0, 1.0]

    # W = 1 (Mid elevation)
    ctrl[0, 1] = [radius * s2 / 2, 0, radius * s2 / 2, s2 / 2]
    ctrl[1, 1] = [radius * 0.5, radius * 0.5, radius * 0.5, 0.5]
    ctrl[2, 1] = [0, radius * s2 / 2, radius * s2 / 2, s2 / 2]

    # W = 2 (North pole / Z-axis)
    ctrl[0, 2] = [0, 0, radius, 1.0]
    ctrl[1, 2] = [0, 0, radius * s2 / 2, s2 / 2]
    ctrl[2, 2] = [0, 0, radius, 1.0]

    return ctrl


def create_solid_octant_sphere(radius=1.0):
    """Generates a 1/8th solid sphere (octant) with axis-aligned reflective boundaries.

    U: Radial direction (Degree 1, bounds the origin to the outer shell)
    V: Azimuthal direction (Degree 2, sweeps from X-axis to Y-axis)
    W: Elevation direction (Degree 2, sweeps from XY-plane to Z-axis)
    """
    # Dimensions: [U (radial), V (azimuth), W (elevation), 4 (homogeneous coords)]
    # U = 2 control points, V = 3 control points, W = 3 control points
    ctrl = np.zeros((2, 3, 3, 4))

    # --- Outer Shell (U = 1) ---
    ctrl[1] = _octant_shell_ctrl(radius)

    # --- Inner Core (U = 0) ---
    # All x,y,z coordinates collapse to exactly 0 (the origin).
    # CRITICAL: The weights (index 3) must perfectly match the Outer Shell
    # to guarantee straight radial elements and prevent Jacobian inversion.
    ctrl[0, :, :, 3] = ctrl[1, :, :, 3]

    # --- Knot Vectors ---
    # U is degree 1 (linear radial interpolation)
    knots_u = [0, 0, 1, 1]

    # V and W are degree 2 (quadratic circular arcs)
    knots_v = [0, 0, 0, 1, 1, 1]
    knots_w = [0, 0, 0, 1, 1, 1]

    # Generate and return the single-patch NURBS volume
    return cad.NURBS([knots_u, knots_v, knots_w], ctrl)


def _seeded_inverse_map(patch, targets, n_grid=60, max_iter=10, tol=1e-8):
    """Like `Patch.inverse_map()`, but seeds each target's Newton iteration
    from the nearest point on a coarse forward-evaluated parametric grid
    instead of the domain center.

    Needed because the default (domain-center) start fails to converge for
    targets near a coordinate singularity -- e.g. the octant shell's entire
    W=1 edge collapses to the single pole point (0, 0, r) for every V (as it
    must, to be a valid sphere), so the Jacobian goes rank-deficient nearby
    and plain Newton stalls at a wrong fixed point for a whole ring of
    targets close to the pole, regardless of their true azimuth. Starting
    from a nearby grid point keeps Newton in the correct basin.

    Parameters
    ----------
    patch: ttnte.cad.Patch
        The (2-D, single-element) patch to invert on.
    targets: torch.Tensor, shape (N, ndim)
        Physical target points.
    n_grid: int
        Resolution of the coarse seeding grid along each parametric axis.
    max_iter, tol: int, float
        Passed to `Patch.inverse_map()`.

    Returns
    -------
    result: ttnte.cad.InverseMapResult
    """
    vv, ww = torch.meshgrid(
        torch.linspace(0, 1, n_grid, dtype=targets.dtype),
        torch.linspace(0, 1, n_grid, dtype=targets.dtype),
        indexing="ij",
    )
    grid_params = torch.stack([vv.flatten(), ww.flatten()], dim=-1)
    grid_pts = patch.evaluate(grid_params)
    nn_idx = torch.cdist(targets, grid_pts).argmin(dim=1)
    initial_guess = grid_params[nn_idx]
    return patch.inverse_map(
        targets, max_iter=max_iter, tol=tol, initial_guess=initial_guess
    )


def evaluate_shell(
    patch, field, r, rc, dtype, device, n_elev=50, n_az=50, max_iter=10, tol=1e-8
):
    """Evaluate the scalar flux on the r=const spherical shell within the
    reflected octant, normalized by the flux at the sphere's center.

    Uses physical (elevation, azimuth) angles directly -- elevation measured
    up from the equator (XY-plane, elevation=0) to the pole (Z-axis,
    elevation=pi/2), azimuth measured from the X-axis to the Y-axis -- rather
    than the patch's own (V, W) parametric coordinates, since the biquadratic
    NURBS octant patch is not a uniform latitude/longitude parametrization.

    Rather than inverting the target physical points directly against `patch`
    (which, after spatial refinement, is a multi-element volume -- Newton's
    method from a single global initial guess can land in the wrong element
    and stall well short of `tol`), this factors the inversion: `create_solid_
    octant_sphere()` sweeps a straight radial line from the origin (matching
    weights at U=0 and U=1), so physical radius = U * rc exactly regardless of
    refinement, and the (V, W) -> direction mapping is identical at every
    radius (a pure radial scaling) and identical before/after refinement
    (knot insertion doesn't change the geometric map). So (V, W) is inverted
    ONCE per (elevation, azimuth) sample on the un-refined, single-Bezier-
    element UNIT shell -- far better conditioned -- and combined with the
    exact U = r / rc to directly forward-evaluate the field on `patch` (no
    inversion against the refined volume at all).

    Parameters
    ----------
    patch: ttnte.cad.Patch
        The (solid octant sphere) patch to evaluate on.
    field: ttnte.linalg.State or similar
        The local DOF-coefficient field on this patch (e.g. from
        `TransportSolution.get_local_field()`), spatial-only (one channel).
    r: float
        The physical radius of the shell to evaluate on.
    rc: float
        The outer radius `patch` was built at (i.e. `create_solid_octant_
        sphere(rc)`).
    dtype: torch.dtype
        The dtype to build sample points with.
    device: torch.device
        The device to build sample points on.
    n_elev, n_az: int
        Number of sample points along elevation and azimuth.
    max_iter, tol: int, float
        Passed to `Patch.inverse_map()` (for the unit-shell (V, W) inversion).

    Returns
    -------
    elev, az: torch.Tensor, shape (n_elev,), (n_az,)
        The sampled elevation and azimuth angles, in radians.
    flux: torch.Tensor, shape (n_elev, n_az)
        The flux at each (elevation, azimuth) sample, normalized by the
        center flux.
    """
    # Convert to a dense tensor -- 3 leading (U, V, W) axes for this patch
    dense_field = field.to_dense().reshape(
        patch.get_ctrlpts_size(0),
        patch.get_ctrlpts_size(1),
        patch.get_ctrlpts_size(2),
        -1,
    )

    # Flux at the center of the sphere -- U=0 collapses to the origin
    # regardless of V, W (any V, W works; 0.5, 0.5 is an arbitrary choice)
    center_flux = patch.evaluate_field(
        dense_field, torch.tensor([[0.0, 0.5, 0.5]], dtype=dtype)
    ).flatten()

    # Build target UNIT direction vectors covering the reflected octant
    elev = torch.linspace(0, torch.pi / 2, n_elev, dtype=dtype)
    az = torch.linspace(0, torch.pi / 2, n_az, dtype=dtype)
    E, A = torch.meshgrid(elev, az, indexing="ij")
    directions = torch.stack(
        [
            torch.cos(E) * torch.cos(A),
            torch.cos(E) * torch.sin(A),
            torch.sin(E),
        ],
        dim=-1,
    ).reshape(-1, 3)

    # Invert direction -> (V, W) once on the un-refined, single-Bezier UNIT
    # shell (radius 1)
    unit_shell = Patch.from_igakit(
        cad.NURBS([[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]], _octant_shell_ctrl(1.0)),
        device=device,
        dtype=dtype,
    )
    result = _seeded_inverse_map(unit_shell, directions, max_iter=max_iter, tol=tol)
    assert result.converged.all()

    # U is exact -- combine with the inverted (V, W) and forward-evaluate the
    # field directly on the refined patch (no inversion against it needed)
    u = torch.full((directions.shape[0], 1), r / rc, dtype=dtype)
    points = torch.cat([u, result.coords], dim=-1)

    flux = (patch.evaluate_field(dense_field, points).flatten() / center_flux).reshape(
        n_elev, n_az
    )

    return elev, az, flux


def relative_l2_error_shell(elev, az, flux, expected):
    """Angularly-integrated (solid-angle-weighted) relative L2 error of a
    constant-radius flux grid against a spherically-symmetric analytic
    value, over the reflected octant.

    Parameters
    ----------
    elev, az: torch.Tensor or numpy.ndarray, shape (n_elev,), (n_az,)
        Elevation and azimuth sample angles, in radians (see
        `evaluate_shell()`).
    flux: torch.Tensor or numpy.ndarray, shape (n_elev, n_az)
        The (normalized) computed flux at each (elevation, azimuth) sample.
    expected: float
        The analytic (normalized) flux at this radius -- constant over the
        shell by spherical symmetry.

    Returns
    -------
    error: float
        sqrt(integral[(flux - expected)^2 dOmega]) / sqrt(integral[expected^2
        dOmega]), with dOmega = cos(elev) d(elev) d(az) and the integral
        restricted to the reflected octant.
    """
    elev = np.asarray(elev)
    az = np.asarray(az)
    flux = np.asarray(flux)

    diff_sq = (flux - expected) ** 2
    inner = np.trapz(diff_sq, az, axis=1)
    numerator = np.trapz(inner * np.cos(elev), elev)
    denominator = expected**2 * np.pi / 2
    return np.sqrt(numerator / denominator)


def test_sphere(request):
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
    qset = ProductQuadrature.gauss_legendre_chebyshev(4, 4, 3)
    qset.to_(torch.device("cpu"), dtype)

    # Spatial fidelity
    numel = 4
    degree = 2

    # pytest specific
    generated_plots = []

    # ========================================================================
    # Get XS information
    fills, xs_server = pu239(num_groups=1, device=cpu, dtype=dtype)

    # ========================================================================
    # Create NURBS surfaces
    # Fuel region
    rc = 6.082547
    v0 = Patch.from_igakit(
        cad.refine(create_solid_octant_sphere(rc), numel, degree),
        device=cpu,
        dtype=dtype,
        fill=fills[0],
    )

    # ========================================================================
    # Create IGA mesh
    mesh = IGAMesh(mpi_context)
    mesh.add_block(v0)
    mesh.connect()

    mesh.set_axis_aligned_conditions(
        BCPlane(x_min=True, y_min=True, z_min=True),
        BoundaryType.REFLECTIVE,
        tol=1e-6,
    )

    mesh.finalize()

    # ========================================================================
    # Plot mesh
    # Plot a 2-D slice of this geometry with a plane that is defined by the point
    # (0, 0, 1) and normal (0, 0, 1)
    backend = "matplotlib"
    style = get_patch_style(backend)
    style.normal = (0, 0, 1)
    style.origin = (0, 0, 1)
    generated_plots.append(f"{request.node.name}_model_2d.png")
    mesh.plot(
        resolution=25,
        show_boundary=True,
        backend=backend,
        filename=generated_plots[-1],
        style=style,
    )

    # Plot the full 3-D geometry
    backend = "pyvista"
    style = get_patch_style(backend)
    generated_plots.append(f"{request.node.name}_model_3d.png")
    plotter = mesh.plot(
        resolution=25,
        show_ctrlpts=True,
        show_ctrlnet=True,
        show_boundary=True,
        backend=backend,
        style=style,
    )
    plotter[0].camera.azimuth = -90  # degrees, off the default view direction
    plotter[0].camera.elevation = -10
    export_pv_plot(plotter[0], generated_plots[-1], style)

    # ========================================================================
    # Create transport driver and distribute across MPI ranks
    # Create the transport driver
    driver = IGATransportDriver3D(mesh, xs_server, mpi_context)

    # Distribute patches among MPI ranks
    driver.distribute([IGADofHeuristic()])

    # ========================================================================
    # Assemble operators
    config = DGTransportAssemblerConfig()
    config.rounding.eps = 1e-8
    config.cross.eps = config.rounding.eps
    driver.assemble(qset, config)

    # ========================================================================
    # Run DD solver
    # Send the lone system to the GPU
    if torch.cuda.is_available():
        sys = driver.get_system(v0.gid)
        sys.to_(torch.device("cuda"))

    result = driver.solve_eigenvalue(
        AMEnSolver(
            nswp=4,
            eps=8e-5,
            eps_forcing=0.01,
            kickrank=4,
            local_iterations=100,
            resets=4,
            rmax=500,
        ),
        tol=1e-4,
        max_iter=500,
        verbose=True,
    )

    # Remove the lone system from the GPU
    if torch.cuda.is_available():
        sys = driver.get_system(v0.gid)
        sys.to_(cpu)

    # ========================================================================
    # Plot the solution
    scalar_result = result.compute_scalar_flux()

    # Plot a 2-D slice of this geometry with a plane that is defined by the point
    # (0, 0, 1) and normal (0, 0, 1)
    backend = "matplotlib"
    style = get_patch_style(backend)
    style.normal = (0, 0, 1)
    style.origin = (0, 0, 1)
    generated_plots.append(f"{request.node.name}_flux_2d.png")
    mesh.plot(
        resolution=25,
        backend=backend,
        solution=scalar_result.select_group(0),
        filename=generated_plots[-1],
        style=style,
        field_label=r"$\phi$",
    )

    # Plot the full 3-D geometry
    backend = "pyvista"
    style = get_patch_style(backend)
    generated_plots.append(f"{request.node.name}_flux_3d.png")
    plotter = mesh.plot(
        resolution=25,
        solution=scalar_result.select_group(0),
        backend=backend,
        style=style,
        field_label=r"$\phi$",
    )
    plotter[0].camera.azimuth = -90  # degrees, off the default view direction
    plotter[0].camera.elevation = -10
    export_pv_plot(plotter[0], generated_plots[-1], style)

    # ========================================================================
    # Check the solution to the analytical
    keff_error = (result.k_eff - 1.0) * 1e5
    passed &= abs(keff_error) < 10

    # Check the scalar flux at four radii against the analytic (normalized by
    # the center flux) bare-critical-sphere solution -- constant over each
    # r=const shell by spherical symmetry, so each check is an
    # angularly-integrated (solid-angle-weighted) relative L2 error over the
    # reflected octant
    field = scalar_result.get_local_field(v0.gid)
    radial_checks = [
        (0.25, 0.93538006),
        (0.50, 0.75575352),
        (0.75, 0.49884364),
        (1.00, 0.19222603),
    ]
    flux_tol = 0.02

    flux_errors = {}
    for frac, expected in radial_checks:
        r = frac * rc
        elev, az, flux = evaluate_shell(v0, field, r, rc, dtype, cpu)
        error = relative_l2_error_shell(elev, az, flux, expected)
        flux_errors[f"Relative L2 error ({frac}rc)"] = error
        passed &= error < flux_tol

        # Plot the relative deviation over the octant shell
        plt.clf()
        rel_dev = (flux.numpy() - expected) / expected
        mesh_plot = plt.pcolormesh(
            az.numpy(), elev.numpy(), rel_dev, shading="gouraud", cmap="RdBu_r"
        )
        plt.colorbar(mesh_plot, label=rf"$\delta\phi(r={frac}r_c)$")
        plt.xlabel(r"Azimuth $\varphi$")
        plt.ylabel(r"Elevation $\theta$")
        plt.tight_layout()
        generated_plots.append(f"{request.node.name}_shell_{frac}rc.png")
        plt.savefig(generated_plots[-1], dpi=300)

    # ========================================================================
    # Attach data to Pytest for conftest.py to read
    errors = {"dk (pcm)": keff_error, **flux_errors}
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
        abs(keff_error) < 10
    ), "The eigenvalue was more than 10 pcm from the true solution"
    for frac, _ in radial_checks:
        assert (
            flux_errors[f"Relative L2 error ({frac}rc)"] < flux_tol
        ), f"Scalar flux at radius {frac}rc exceeds the allowed tolerance"
