# Task Completion Checklist

After completing a coding task, run:

1. `black --line-length 100 --check <changed_files>` — Check formatting
2. `ruff check <changed_files>` — Check linting
3. `mypy <changed_files>` — Check types (if applicable)
4. `pytest` — Run test suite
5. Verify no secrets or .env files are staged
