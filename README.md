# ttnte-vnv

Verification and validation benchmarks for ttnte. Every benchmark that takes a
`solver` argument runs once per solver: `amen` (block-Jacobi DD + AMEn), `sweep`
(exact transport sweeper) and `cross` (sweeper with cross transport solves).

## Running

```shell
pytest benchmarks --limited                      # all non-slow benchmarks, all solvers
pytest benchmarks --solvers sweep,cross          # pick solvers
pytest benchmarks -k "slab and cross"            # or select test ids directly
pytest benchmarks --solver-opt amen.local.eps=1e-6 --solver-opt cross.cross.tol=1e-5
pytest benchmarks --solver-opt tol=1e-7          # no solver prefix: every solver
```

`--solver-opt [SOLVER.]KEY[.SUBKEY]=VALUE` overrides a solver setting and wins
over the test file. Values are Python literals or plain strings (enum member
names such as `amen.dd.memory_policy=OUT_OF_CORE`, cross layouts). The settings
of each solver are documented at the top of `vnv/solvers.py`, with their defaults
in `DEFAULTS`.

## Writing a benchmark

```python
from vnv import flux_errors, record, setup, zscores

# Per-solver settings for this problem, merged over vnv.solvers.DEFAULTS
OPTS = {
    "amen": {"tol": 1e-5, "local": {"eps": 1e-7}, "dd": {"tol": 5e-6}},
    "sweep": {"tol": 1e-7},
    "cross": {"tol": 1e-7, "cross": {"tol": 1e-5}},
}


def test_problem(request, solver):        # `solver` is parametrized by conftest.py
    setup(max_ranks=2)                    # MPI, threads, dtype, no autograd
    ...                                   # build xs_server, qset and a finalized mesh
    result = solver.run(mesh, xs_server, qset, eigenvalue=True, opts=OPTS)

    keff_error = (result.k_eff - k_ref) * 1e5
    phi, group_errors, total_error = flux_errors(result.scalar, phi_mc, normalize=True)
    ...
    record(request, result, value=result.k_eff, ref=k_ref, passed=passed,
           errors={"dk (pcm)": keff_error, ...}, plots=generated_plots)
```

- Build the mesh inside the test: the drivers partition it across ranks, so
  every run needs its own.
- `result.scalar` is always the scalar flux; `result.k_eff` is `None` for fixed
  source problems.
- Pass `balance=True` to `run()` to use `result.global_balance()` and
  `result.patch_balance_table()`.
- `@pytest.mark.solvers("sweep", "cross")` restricts a test to some solvers.
