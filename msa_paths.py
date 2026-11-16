"""Shared path resolution for the MSA-ablation scripts.

Everything the pipeline *produces* (AF3 inputs, fold outputs, derived tables)
lives under one work directory, named by $MSA_DECISION_ROOT. That directory is
large -- hundreds of gigabytes once the folds are run -- so it is deliberately
kept outside this repository.

    export MSA_DECISION_ROOT=/path/to/a/big/disk/msa_decision

`rpath(name)` resolves a derived table: it prefers your own copy under
$MSA_DECISION_ROOT, and falls back to the published copy in `results/` that
ships with this repo. That fallback is what lets the analysis and figure
scripts run against our results without re-folding anything.
"""
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get('MSA_DECISION_ROOT', os.getcwd())
RESULTS = os.path.join(REPO, 'results')

# the analysis helpers (ipsae_recompute, compile_all_folds, ...) import each
# other by bare module name whichever directory a script was launched from
for _d in (REPO, os.path.join(REPO, 'analysis'), os.path.join(REPO, 'figures')):
    if _d not in sys.path:
        sys.path.insert(0, _d)


def rpath(name):
    """Path to a derived table: your work directory if present, else results/."""
    mine = os.path.join(ROOT, name)
    if os.path.exists(mine):
        return mine
    shipped = os.path.join(RESULTS, name)
    if os.path.exists(shipped):
        return shipped
    return mine          # report the work-directory path in the error
