import os

from igakit import cad
import pytest
import torch
import numpy as np

from ttnte import mpi_context
from ttnte.xs.benchmarks import u235
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


def create_hollow_octant_sphere(r_inner, r_outer):
    """Generates a 1/8th hollow spherical shell (octant) between `r_inner` and
    `r_outer`, with axis-aligned reflective boundaries on its flat faces.

    U: Radial direction (Degree 1, bounds r_inner to r_outer)
    V: Azimuthal direction (Degree 2, sweeps from X-axis to Y-axis)
    W: Elevation direction (Degree 2, sweeps from XY-plane to Z-axis)
    """
    ctrl = np.zeros((2, 3, 3, 4))

    # --- Inner Shell (U = 0) / Outer Shell (U = 1) ---
    ctrl[0] = _octant_shell_ctrl(r_inner)
    ctrl[1] = _octant_shell_ctrl(r_outer)

    # --- Knot Vectors ---
    knots_u = [0, 0, 1, 1]
    knots_v = [0, 0, 0, 1, 1, 1]
    knots_w = [0, 0, 0, 1, 1, 1]

    # Generate and return the single-patch NURBS volume
    return cad.NURBS([knots_u, knots_v, knots_w], ctrl)


@pytest.mark.mpi(min_size=1)
def test_reflected_sphere(request):
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
    if mpi_context.world_size > 2:
        pytest.skip("Test requires 2 or fewer processes")

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
    qset = ProductQuadrature.gauss_legendre_chebyshev(4, 4, 3)
    qset.to_(torch.device("cpu"), dtype)

    # Spatial fidelity
    numel = 4
    degree = 2

    # pytest specific
    generated_plots = []

    # ========================================================================
    # Get XS information
    fills, xs_server = u235(device=cpu, dtype=dtype)

    # ========================================================================
    # Create NURBS surfaces
    # Fuel region
    r0 = 6.12745
    v0 = Patch.from_igakit(
        cad.refine(create_solid_octant_sphere(r0), numel, degree),
        device=cpu,
        dtype=dtype,
        fill=fills[0],
    )

    # Water reflector region
    r1 = 12.2549
    v1 = Patch.from_igakit(
        cad.refine(create_hollow_octant_sphere(r0, r1), numel, degree),
        device=cpu,
        dtype=dtype,
        fill=fills[1],
    )

    # ========================================================================
    # Create IGA mesh
    mesh = IGAMesh(mpi_context)
    mesh.add_block(v0)
    mesh.add_block(v1)
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
    style.mesh.cmap = {
        fills[0].to_string(): "maroon",
        fills[1].to_string(): "cornflowerblue",
    }
    generated_plots.append(f"{request.node.name}_model_2d.png")
    mesh.plot(
        resolution=25,
        backend=backend,
        filename=generated_plots[-1],
        style=style,
    )

    if mpi_context.rank == 0:
        # Plot the full 3-D geometry
        backend = "pyvista"
        style = get_patch_style(backend)
        style.mesh.cmap = {
            fills[0].to_string(): "maroon",
            fills[1].to_string(): "cornflowerblue",
        }
        generated_plots.append(f"{request.node.name}_model_3d.png")
        plotter = mesh.plot(
            resolution=25,
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
    outer_tol = 1e-4
    inner_tol = 5e-5
    eps = 1e-5

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
    strategy.set_local_solver(
        AMEnSolver(
            nswp=10,
            eps=eps,
            eps_forcing=0.01,
            kickrank=4,
            local_iterations=200,
            resets=4,
            rmax=500,
        )
    )
    dd_solver = IGADDSolver(driver.mesh, strategy)

    # Run DD eigenvalue solver
    result = driver.solve_eigenvalue(
        dd_solver, tol=outer_tol, max_iter=500, verbose=True
    )

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
        show_boundary=True,
        backend=backend,
        solution=scalar_result.select_group(0),
        filename=generated_plots[-1],
        style=style,
        gather=True,
        feild_name=r"$\phi$",
    )

    # Plot the full 3-D geometry
    backend = "pyvista"
    style = get_patch_style(backend)
    generated_plots.append(f"{request.node.name}_flux_3d.png")
    plotter = mesh.plot(
        resolution=25,
        show_ctrlpts=True,
        show_ctrlnet=True,
        show_boundary=True,
        solution=scalar_result.select_group(0),
        backend=backend,
        style=style,
        gather=True,
        feild_name=r"$\phi$",
    )
    if mpi_context.rank == 0:
        plotter[0].camera.azimuth = -90  # degrees, off the default view direction
        plotter[0].camera.elevation = -10
        export_pv_plot(plotter[0], generated_plots[-1], style)

    # ========================================================================
    # Check the solution to the analytical
    keff_error = (result.k_eff - 1.0) * 1e5
    passed &= abs(keff_error) < 5

    # ========================================================================
    # Attach data to Pytest for conftest.py to read
    errors = {"dk (pcm)": keff_error}
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
        abs(keff_error) < 5
    ), "The eigenvalue was more than 5 pcm from the true solution"
