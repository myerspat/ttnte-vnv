"""Comparison against reference (OpenMC) results and the V&V report hooks."""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ttnte import mpi_context

STATS = ("Minimum", "Q1", "Median", "Q2", "Maximum", "Mean")


def flux_errors(scalar, phi_ref, normalize=False, n=(5, 5), **average_kwargs):
    """Average a scalar flux onto the reference's regular mesh and compare.

    Args:
        scalar: Scalar-flux ``TransportSolution`` (``RunResult.scalar``).
        phi_ref: Reference flux of shape ``(G, *mesh_shape)``.
        normalize: Scale both fluxes to unit L2 norm first (eigenvalue problems).
            ``phi_ref`` is not modified.
        n: Integration points per regular-mesh cell and dimension.
        average_kwargs: Passed on to ``regular_mesh_average`` (e.g.
            ``seed_resolution``).

    Returns:
        ``(phi, group_errors, total_error)``: the averaged flux ``(G, *mesh_shape)``,
        the per-group relative L2 errors and their L2 norm.
    """
    num_groups = phi_ref.shape[0]
    phi = (
        scalar.regular_mesh_average(list(phi_ref.shape[1:]), list(n), **average_kwargs)
        .moveaxis(-1, 0)
        .cpu()
        .numpy()
    )
    if normalize:
        phi = phi / np.linalg.norm(phi.flatten(), ord=2)
        phi_ref = phi_ref / np.linalg.norm(phi_ref.flatten(), ord=2)
    group_errors = np.linalg.norm(
        (phi - phi_ref).reshape(num_groups, -1), axis=1, ord=2
    ) / np.linalg.norm(phi_ref.reshape(num_groups, -1), axis=1, ord=2)
    return phi, group_errors, np.linalg.norm(group_errors, ord=2)


def zscores(phi, phi_ref, stdev_ref, extent, prefix):
    """Per-group z-score plots and statistics of ``|phi - phi_ref| / stdev_ref``.

    Args:
        phi, phi_ref, stdev_ref: Fluxes and reference stdev, ``(G, nx, ny)``. Pass
            ``stdev_ref`` scaled like ``phi_ref`` if you normalized.
        extent: ``(x_min, x_max, y_min, y_max)`` of the regular mesh.
        prefix: Plot file name prefix; files are ``{prefix}_zscore_{g}.png``.

    Returns:
        ``(stats, plots)``: ``{"Mean": ..., "Mean (1)": ..., ...}`` over all groups
        (and per group when there is more than one), and the plot file names.
    """
    z = np.abs(phi - phi_ref) / stdev_ref
    num_groups = z.shape[0]
    X, Y = np.meshgrid(
        np.linspace(extent[0], extent[1], z.shape[1]),
        np.linspace(extent[2], extent[3], z.shape[2]),
    )

    def summarize(values):
        return dict(
            zip(
                STATS,
                [
                    np.min(values),
                    np.percentile(values, 25),
                    np.median(values),
                    np.percentile(values, 75),
                    np.max(values),
                    np.mean(values),
                ],
            )
        )

    stats = {}
    if num_groups > 1:
        for g in range(num_groups):
            for name, value in summarize(z[g]).items():
                stats[f"{name} ({g + 1})"] = value
    stats.update(summarize(z))

    plots = []
    for g in range(num_groups):
        plots.append(f"{prefix}_zscore_{g + 1}.png")
        if mpi_context.rank != 0:
            continue
        plt.clf()
        ax = plt.gca()
        cmesh = ax.pcolormesh(X, Y, z[g], cmap="plasma")
        cbar = plt.colorbar(
            cmesh,
            cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05),
        )
        cbar.set_label(
            rf"$\frac{{|\phi_{g + 1}^{{\text{{ttnte}}}} - "
            rf"\phi_{g + 1}^{{\text{{OpenMC}}}}|}}"
            rf"{{|\sigma_{g + 1}^{{\text{{OpenMC}}}}|}}$",
            size=14,
        )
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        plt.tight_layout()
        plt.savefig(plots[-1], dpi=300)
    return stats, plots


def record(
    request,
    result,
    *,
    value,
    ref,
    passed,
    errors,
    plots=(),
    metric="k_eff",
):
    """Attach a run's metrics and plots to the test for the V&V report.

    Args:
        request: The pytest ``request`` fixture.
        result: The ``RunResult``.
        value, ref: The headline metric of ttnte and of the reference.
        passed: Whether every check passed.
        errors: ``{label: value}`` error breakdown for the report.
        plots: Plot file names to embed.
        metric: Name of the headline metric.
    """
    errors = dict(errors)
    errors["wall time (s)"] = result.seconds
    errors["solve time (s)"] = result.solve_seconds
    errors["outer iterations"] = result.iterations
    for name, stat in result.stats.items():
        if isinstance(stat, (int, float)):
            errors[name] = stat

    if mpi_context.rank == 0:
        for name, error in errors.items():
            print(f"{name}: {error}")

    request.node.vnv_plots = list(plots)
    request.node.vnv_metrics = {
        "name": request.node.name,
        "solver": result.solver,
        "seconds": result.seconds,
        "metric": metric,
        "ttnte_val": value,
        "ref_val": ref,
        "passed": bool(passed),
        "detailed_errors": errors,
    }
