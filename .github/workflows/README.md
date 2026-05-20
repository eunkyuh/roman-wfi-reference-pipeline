# Roman WFI Reference File Pipeline

## Continuous Integration

The CI workflow (`.github/workflows/ci_workflow.yml`) runs on every PR and on pushes to `main`. It:

1. Lints with `ruff`.
2. Runs `pytest` with coverage.
3. On PRs to `main`, posts a per-file coverage diff against the `coverage-info` branch (the latest `main` baseline).
4. On pushes to `main`, updates the coverage badge and `coverage.xml` on the `coverage-info` branch.

### Tuning the coverage gate

The PR coverage check is configured at the bottom of `ci_workflow.yml`:

```yaml
- name: Coverage diff on PR
  uses: ./.github/jobs/coverage_diff
  with:
    fail-under-total: '0'    # Fail PR if total coverage drops below this %
    fail-file-decrease: '100' # Fail PR if any file's coverage drops more than this many points
```

Raise `fail-under-total` to enforce a minimum overall coverage. Lower `fail-file-decrease` to block PRs that significantly reduce any single file's coverage. The defaults above effectively disable the gate.

The CI requires the existance of a branch named `coverage-info` which contains the baseline coverage.xml file.