import os
import base64
import pytest
from pytest_html import extras
from ttnte import mpi_context  # <-- Import your MPI context!

VNV_RESULTS = []
VNV_LABEL = ""


def pytest_configure(config):
    global VNV_LABEL
    VNV_LABEL = config.getoption("--vnv-label")


def _vnv_title(base: str) -> str:
    return f"{base} — {VNV_LABEL}" if VNV_LABEL else base


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
    """
    Hook executed at the very end of the entire test run.
    """
    if mpi_context.rank == 0:
        summary_file = os.environ.get("GITHUB_STEP_SUMMARY")

        if summary_file and VNV_RESULTS:
            with open(summary_file, "a") as f:
                f.write(
                    f"## {_vnv_title('🚀 ttnte Verification & Validation Summary')}\n"
                )
                f.write(
                    "| Test Name | ttnte $k_{eff}$ | Reference $k_{eff}$ | Error Breakdown | Status |\n"
                )
                f.write("| --- | --- | --- | --- | --- |\n")

                for res in VNV_RESULTS:
                    status = "✅ PASSED" if res["passed"] else "❌ FAILED"
                    ref_k = res.get("ref_k", res.get("openmc_k", 0.0))

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

                    f.write(
                        f"| {res['name']} | {res['ttnte_k']:.5f} | {ref_k:.5f} | "
                        f"{error_str} | {status} |\n"
                    )
                f.write(
                    "\n\n*💡 Note: The complete HTML report containing 2D spatial error plots is attached below as a workflow artifact.*"
                )

    # Ensure MPI closes correctly
    try:
        from mpi4py import MPI

        if MPI.Is_initialized() and not MPI.Is_finalized():
            MPI.COMM_WORLD.Barrier()
    except Exception:
        pass


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


def pytest_collection_modifyitems(config, items):
    if config.getoption("--limited", default=False):
        skip_slow = pytest.mark.skip(
            reason="Skipping slow test because --limited flag is active"
        )
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


def pytest_html_results_summary(prefix, summary, postfix):
    """
    Hook to inject custom HTML into the top of the pytest-html report.
    """
    if mpi_context.rank != 0 or not VNV_RESULTS:
        return

    table_html = [
        f"<h2>{_vnv_title('🚀 ttnte Verification & Validation Summary')}</h2>",
        "<table style='width: 100%; border-collapse: collapse; text-align: left; margin-bottom: 20px;' border='1'>",
        "<tr style='background-color: #f8f9fa;'>",
        "<th style='padding: 10px; border: 1px solid #ddd;'>Test Name</th>",
        "<th style='padding: 10px; border: 1px solid #ddd;'>ttnte <i>k</i><sub>eff</sub></th>",
        "<th style='padding: 10px; border: 1px solid #ddd;'>Reference <i>k</i><sub>eff</sub></th>",
        "<th style='padding: 10px; border: 1px solid #ddd;'>Error Breakdown</th>",
        "<th style='padding: 10px; border: 1px solid #ddd;'>Status</th>",
        "</tr>",
    ]

    for res in VNV_RESULTS:
        status = "✅ PASSED" if res["passed"] else "❌ FAILED"
        ref_k = res.get("ref_k", res.get("openmc_k", 0.0))

        # Build pure HTML list items without Max entry
        if "detailed_errors" in res and isinstance(res["detailed_errors"], dict):
            error_html = "<ul style='margin: 0; padding-left: 15px; font-size: 0.9em;'>"
            for label, err_val in res["detailed_errors"].items():
                error_html += f"<li><b>{label}:</b> {err_val:.5g}</li>"
            error_html += "</ul>"
        else:
            error_html = "N/A"

        table_html.append(
            f"<tr>"
            f"<td style='padding: 10px; border: 1px solid #ddd;'>{res['name']}</td>"
            f"<td style='padding: 10px; border: 1px solid #ddd;'>{res['ttnte_k']:.5f}</td>"
            f"<td style='padding: 10px; border: 1px solid #ddd;'>{ref_k:.5f}</td>"
            f"<td style='padding: 10px; border: 1px solid #ddd;'>{error_html}</td>"
            f"<td style='padding: 10px; border: 1px solid #ddd;'>{status}</td>"
            f"</tr>"
        )

    table_html.append("</table>")
    prefix.extend(table_html)
