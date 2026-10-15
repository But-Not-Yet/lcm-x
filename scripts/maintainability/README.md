# Report-only maintainability inventory

Run from the repository root, with Python 3.11+ and Ruff on PATH:

```sh
python scripts/maintainability/report.py
python scripts/maintainability/report.py --format json
python scripts/maintainability/report.py --baseline scripts/maintainability/baseline.json --write-baseline
```

Numbers, import violations, missing dependencies and operational errors are advisory.
Only argument usage errors return nonzero. Warnings mean incomplete evidence, not a
clean inventory. An incomplete scan will not replace the baseline. No extraction is
required to land a small fix. This script does not modify source or Ruff configuration.

The baseline represents the source tree containing it; it deliberately has no embedded
commit SHA (which would become stale on committing the baseline itself). JSON contains
all metrics, sites and changes; Markdown is limited to 60 lines with the largest function
LOC increases first. Baseline refresh is explicit; ordinary runs never write it.

## Definitions

- Scan Python files recursively, pruning hidden directories, `venv`, `node_modules` and
  `__pycache__`. Functions include tests and tooling. Production means root Python modules
  plus `access_context/`, `access_policy/`, and `teams/` (the audit's scope).
- Function identity is `file::qualname`, including nested functions and classes. AST LOC
  runs from `def` through `end_lineno`, including blank lines but excluding decorators.
  File LOC counts physical lines. Repeated definitions with the same qualified name use
  the last definition. Ruff provides CC, branch and statement counts. Thresholds are
  overridden to zero on the command line to obtain all positive values; these are
  measurement thresholds, never build limits. Noqa and per-file ignores are disabled
  for this measurement. `c901_over_10` permits comparison with the audit's default C901
  count. Ruff uses one thread, no cache and a 15-second subprocess budget.
- Report every positive change to a baselined function's LOC/CC/branches/statements;
  report new functions exceeding 80 LOC or CC 15. Decreases are visible in the inventory
  but do not create warnings. Missing Ruff values remain null, never guessed.
- Import contracts inspect all AST imports (including lazy and TYPE_CHECKING imports)
  in the six storage modules named in `report.py`. Relative, bare and `hermes_lcm`
  imports of the six orchestration modules are reported. Dynamic imports are not inferred.
- Attribute inventory follows LCMEngine's repository-defined base class names without
  importing product code. It records direct `self.attr` Store targets (including
  tuple/annotated/augmented assignments), with writer modules. It excludes container
  mutations, `setattr`, deletes and unrelated classes. New writer means a new module.
- `CONFIG_GETATTR` in `report.py` is the exact lexical definition: a `getattr` prefix
  with `config` or `_config` (possibly dotted), a quoted field and a following comma
  for the default argument. Production only; comments/strings can match. It does not
  recognize aliases, computed field names or optional whitespace before `(`.
- Private test references are AST attribute accesses whose immediate receiver name
  ends in `engine` or `store`, optionally underscore-delimited and numbered, plus
  private imports from `hermes_lcm.*`. This is a naming heuristic, not type inference;
  other aliases and generic private attributes are deliberately excluded.
- DDL matches CREATE TABLE, CREATE [UNIQUE] INDEX and ALTER TABLE in production string
  literals, including literal fragments of f-strings. Standalone strings/docstrings
  and comments are excluded. `db_bootstrap.py` is the schema owner; all other DDL sites
  are reported. Totals include the schema owner for audit comparison. SQL is not executed
  or parsed; dynamically assembled keywords are outside this metric.
- Site identity is `file::qualname::content#occurrence`, independent of line number.
  DDL uses a digest of its literal instead of copying SQL. The baseline retains line
  numbers for navigation. Reordering identical sites does not change their count.

A post-GA CI step can run `python scripts/maintainability/report.py >> "$GITHUB_STEP_SUMMARY"`.
No workflow is installed by this lane. Keep the step report-only; do not compare numbers
in a blocking assertion. Use the same Ruff version when comparing baselines.
