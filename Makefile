# Run the V&V suite, limited (no `slow` benchmarks) or full, on one MPI rank or
# several. Mirrors .github/workflows/vnv.yml.
#
#   make limited                 # fast subset, 1 rank, all solvers
#   make full-mpi NP=4           # every benchmark on 4 ranks
#   make limited SOLVERS=sweep,cross OPTS="cross.cross.tol=1e-5 tol=1e-7"
#   make full ARGS='-k "slab and cross"'
#
# Every target runs under mpirun, even on one rank: pytest-mpi skips the
# mpi-marked benchmarks without --with-mpi.

# ============================================================================
# Settings (override on the command line: make <target> VAR=VALUE)

# MPI ranks of the *-mpi targets
NP ?= 2
# Solvers of the 1-rank and multi-rank targets. Cross transport solves are
# serial in space for now, so the multi-rank runs leave cross out
SOLVERS ?= amen,sweep,cross
MPI_SOLVERS ?= amen,sweep
# Space-separated --solver-opt overrides, [SOLVER.]KEY=VALUE each
OPTS ?=
# Extra pytest arguments (-k, -x, test paths under benchmarks, ...)
ARGS ?=
# Benchmarks to collect
BENCHMARKS ?= benchmarks

MPIRUN ?= mpirun
# Open MPI pins each rank to one core otherwise, starving the solver threads
MPI_FLAGS ?= --bind-to none
PYTEST ?= pytest
# Virtual display for the plots, when xvfb-run is installed
XVFB ?= $(if $(shell command -v xvfb-run 2>/dev/null),xvfb-run -a,)

# ============================================================================
# Internals

SOLVER_OPTS = $(foreach opt,$(OPTS),--solver-opt '$(opt)')
PYTEST_FLAGS = --with-mpi --import-mode=importlib --self-contained-html \
	$(SOLVER_OPTS) $(ARGS)

# $(call vnv,ranks,label,solvers,report,extra pytest flags)
define vnv
	$(XVFB) $(MPIRUN) -n $(1) $(MPI_FLAGS) $(PYTEST) $(BENCHMARKS) \
		--vnv-label "$(2)" --solvers $(3) --html=$(4) $(5) $(PYTEST_FLAGS)
endef

RANK_LABEL = $(if $(filter 1,$(NP)),1 MPI Rank,$(NP) MPI Ranks)

# ============================================================================
# Targets

.PHONY: help limited full limited-mpi full-mpi limited-all full-all clean

help:
	@echo "V&V targets:"
	@echo "  limited      fast subset (no slow benchmarks), 1 MPI rank"
	@echo "  full         every benchmark, 1 MPI rank"
	@echo "  limited-mpi  fast subset, NP MPI ranks (default $(NP))"
	@echo "  full-mpi     every benchmark, NP MPI ranks"
	@echo "  limited-all  limited, then limited-mpi (what CI runs)"
	@echo "  full-all     full, then full-mpi"
	@echo "  clean        remove the reports and plots"
	@echo
	@echo "Variables: NP, SOLVERS (1 rank, default $(SOLVERS)),"
	@echo "  MPI_SOLVERS (NP ranks, default $(MPI_SOLVERS)),"
	@echo "  OPTS (space-separated [SOLVER.]KEY=VALUE overrides), ARGS (extra pytest"
	@echo "  arguments), BENCHMARKS, MPIRUN, MPI_FLAGS, PYTEST, XVFB"

limited:
	$(call vnv,1,1 MPI Rank,$(SOLVERS),report_no_mpi.html,--limited)

full:
	$(call vnv,1,1 MPI Rank,$(SOLVERS),report_no_mpi.html,)

limited-mpi:
	$(call vnv,$(NP),$(RANK_LABEL),$(MPI_SOLVERS),report_mpi.html,--limited)

full-mpi:
	$(call vnv,$(NP),$(RANK_LABEL),$(MPI_SOLVERS),report_mpi.html,)

# Run the multi-rank suite even when the 1-rank one fails, then fail if
# either did
limited-all:
	@$(MAKE) limited; status=$$?; $(MAKE) limited-mpi && exit $$status

full-all:
	@$(MAKE) full; status=$$?; $(MAKE) full-mpi && exit $$status

clean:
	rm -f report_no_mpi.html report_mpi.html *.png
	rm -rf assets
