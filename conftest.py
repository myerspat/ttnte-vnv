import base64
import os
import sys
import pytest
from pytest_html import extras
from ttnte import mpi_context  # <-- Import your MPI context!
from vnv.solvers import SOLVERS, Solver, parse_overrides

VNV_RESULTS = []
VNV_LABEL = ""
_EXIT_STATUS = None  # set once the test session has run


def pytest_configure(config):
    global VNV_LABEL
    VNV_LABEL = config.getoption("--vnv-label")
    config.addinivalue_line(
        "markers",
        "solvers(*names): run this test only with these solvers "
        f"(of {', '.join(SOLVERS)})",
    )

    # Validate the solver selection and overrides before collecting anything
    solvers = [name.strip() for name in config.getoption("--solvers").split(",")]
    unknown = [name for name in solvers if name not in SOLVERS]
    if unknown:
        _usage_error(
            f"--solvers: unknown solver(s) {unknown}, expected some of {SOLVERS}"
        )
    config.vnv_solvers = solvers
    try:
        config.vnv_overrides = parse_overrides(config.getoption("--solver-opt"))
    except ValueError as error:
        _usage_error(str(error))


def _usage_error(message):
    # Importing ttnte forces a zero exit status at interpreter exit, so a
    # pytest.UsageError would not fail the run: report and exit directly
    print(f"ERROR: {message}", file=sys.stderr, flush=True)
    os._exit(pytest.ExitCode.USAGE_ERROR)


def pytest_generate_tests(metafunc):
    """Run every test taking a `solver` argument once per selected solver."""
    if "solver" not in metafunc.fixturenames:
        return
    names = metafunc.config.vnv_solvers
    marker = metafunc.definition.get_closest_marker("solvers")
    if marker is not None:
        names = [name for name in names if name in marker.args]
    metafunc.parametrize(
        "solver",
        [Solver(name, metafunc.config.vnv_overrides) for name in names],
        ids=names,
    )


def _vnv_title(base: str) -> str:
    return f"{base} — {VNV_LABEL}" if VNV_LABEL else base


def _get_vnv_values(res: dict):
    """Safely extract metric label, ttnte value, and reference value.

    Supports both legacy k_eff keys and generic metric keys.
    """
    metric = res.get("metric", res.get("metric_name", "k_eff"))

    # Extract ttnte value (generic first, then fallback)
    ttnte_val = res.get("ttnte_val", res.get("ttnte_k", 0.0))

    # Extract reference value (generic first, then ref_k / openmc_k fallbacks)
    ref_val = res.get("ref_val", res.get("ref_k", res.get("openmc_k", 0.0)))

    return metric, ttnte_val, ref_val


def _get_vnv_run(res: dict):
    """Solver name and wall time of a result ('-' for tests not using vnv.run)."""
    seconds = res.get("seconds")
    return res.get("solver", "-"), "-" if seconds is None else f"{seconds:.1f}"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    extras_list = getattr(report, "extras", [])

    if report.when == "call":
        plot_files = getattr(item, "vnv_plots", [])

        for plot_filename in plot_files:
            if os.path.exists(plot_filename):
                with open(plot_filename, "rb") as image_file:
                    image_b64 = base64.b64encode(image_file.read()).decode("utf-8")

                html_snippet = f'<div><img src="data:image/png;base64,{image_b64}" style="max-width: 600px; margin: 10px;" alt="{os.path.basename(plot_filename)}" /></div>'
                extras_list.append(extras.html(html_snippet))

        report.extras = extras_list

        if hasattr(item, "vnv_metrics"):
            VNV_RESULTS.append(item.vnv_metrics)


def pytest_sessionfinish(session, exitstatus):
    """Hook executed at the very end of the entire test run."""
    global _EXIT_STATUS
    _EXIT_STATUS = exitstatus

    if mpi_context.rank != 0:
        return

    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")

    if summary_file and VNV_RESULTS:
        with open(summary_file, "a") as f:
            f.write(f"## {_vnv_title('🚀 ttnte Verification & Validation Summary')}\n")
            f.write(
                "| Test Name | Solver | Metric | ttnte Value | Reference Value | "
                "Error Breakdown | Time (s) | Status |\n"
            )
            f.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")

            for res in VNV_RESULTS:
                status = "✅ PASSED" if res["passed"] else "❌ FAILED"
                metric, ttnte_val, ref_val = _get_vnv_values(res)

                # Build pure breakdown items without Max entry
                if "detailed_errors" in res and isinstance(
                    res["detailed_errors"], dict
                ):
                    breakdown_items = []
                    for label, err_val in res["detailed_errors"].items():
                        breakdown_items.append(f"• {label}: {err_val:.5g}")
                    error_str = "<br>".join(breakdown_items)
                else:
                    error_str = "N/A"

                solver, seconds = _get_vnv_run(res)
                f.write(
                    f"| {res['name']} | {solver} | {metric} | {ttnte_val:.5f} | "
                    f"{ref_val:.5f} | {error_str} | {seconds} | {status} |\n"
                )
            f.write(
                "\n\n*💡 Note: The complete HTML report containing 2D spatial error plots is attached below as a workflow artifact.*"
            )


def pytest_addoption(parser):
    parser.addoption(
        "--limited",
        action="store_true",
        default=False,
        help="Run a fast subset of V&V tests",
    )
    parser.addoption(
        "--vnv-label",
        action="store",
        default="",
        help="Label distinguishing this run (e.g. 'No MPI', '2 MPI Ranks') "
        "in the summary tables",
    )
    parser.addoption(
        "--solvers",
        action="store",
        default=",".join(SOLVERS),
        help="Comma-separated solvers to run every benchmark with "
        f"(default: {','.join(SOLVERS)})",
    )
    parser.addoption(
        "--solver-opt",
        action="append",
        default=[],
        metavar="[SOLVER.]KEY=VALUE",
        help="Override a solver setting, e.g. 'amen.local.eps=1e-6', "
        "'cross.cross.tol=1e-5', or 'tol=1e-5' for every solver. Repeatable; "
        "wins over the settings in the test files. See vnv/solvers.py.",
    )


def pytest_unconfigure(config):
    """This hook runs absolutely last, AFTER pytest-html has successfully

    written report_mpi.html to the disk.
    """
    # 1. Synchronize and Finalize safely
    try:
        from mpi4py import MPI

        if MPI.Is_initialized() and not MPI.Is_finalized():
            MPI.COMM_WORLD.Barrier()
            MPI.Finalize()
    except Exception:
        pass

    # 2. Instantly terminate to prevent Torch/C++ static destructor segfaults.
    # Not before a session ran: pytest still has to report usage errors
    if _EXIT_STATUS is not None:
        os._exit(_EXIT_STATUS)


def pytest_collection_modifyitems(config, items):
    if config.getoption("--limited", default=False):
        skip_slow = pytest.mark.skip(
            reason="Skipping slow test because --limited flag is active"
        )
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


def pytest_html_results_summary(prefix, summary, postfix):
    """Hook to inject custom HTML into the top of the pytest-html report."""
    if mpi_context.rank != 0 or not VNV_RESULTS:
        return

    table_html = [
        f"<h2>{_vnv_title('🚀 ttnte Verification & Validation Summary')}</h2>",
        (
            "<table style='width: 100%; border-collapse: collapse; text-align:"
            " left; margin-bottom: 20px;' border='1'>"
        ),
        "<tr style='background-color: #f8f9fa;'>",
        "<th style='padding: 10px; border: 1px solid #ddd;'>Test Name</th>",
        "<th style='padding: 10px; border: 1px solid #ddd;'>Solver</th>",
        "<th style='padding: 10px; border: 1px solid #ddd;'>Metric</th>",
        ("<th style='padding: 10px; border: 1px solid #ddd;'>ttnte" " Value</th>"),
        ("<th style='padding: 10px; border: 1px solid #ddd;'>Reference" " Value</th>"),
        ("<th style='padding: 10px; border: 1px solid #ddd;'>Error" " Breakdown</th>"),
        "<th style='padding: 10px; border: 1px solid #ddd;'>Time (s)</th>",
        "<th style='padding: 10px; border: 1px solid #ddd;'>Status</th>",
        "</tr>",
    ]

    for res in VNV_RESULTS:
        status = "✅ PASSED" if res["passed"] else "❌ FAILED"
        metric, ttnte_val, ref_val = _get_vnv_values(res)

        # Build pure HTML list items without Max entry
        if "detailed_errors" in res and isinstance(res["detailed_errors"], dict):
            error_html = "<ul style='margin: 0; padding-left: 15px; font-size: 0.9em;'>"
            for label, err_val in res["detailed_errors"].items():
                error_html += f"<li><b>{label}:</b> {err_val:.5g}</li>"
            error_html += "</ul>"
        else:
            error_html = "N/A"

        solver, seconds = _get_vnv_run(res)
        table_html.append(
            f"<tr>"
            f"<td style='padding: 10px; border: 1px solid #ddd;'>{res['name']}</td>"
            f"<td style='padding: 10px; border: 1px solid #ddd;'>{solver}</td>"
            f"<td style='padding: 10px; border: 1px solid #ddd;'>{metric}</td>"
            f"<td style='padding: 10px; border: 1px solid #ddd;'>{ttnte_val:.5f}</td>"
            f"<td style='padding: 10px; border: 1px solid #ddd;'>{ref_val:.5f}</td>"
            f"<td style='padding: 10px; border: 1px solid #ddd;'>{error_html}</td>"
            f"<td style='padding: 10px; border: 1px solid #ddd;'>{seconds}</td>"
            f"<td style='padding: 10px; border: 1px solid #ddd;'>{status}</td>"
            f"</tr>"
        )

    table_html.append("</table>")
    prefix.extend(table_html)
