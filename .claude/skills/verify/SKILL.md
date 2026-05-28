---
name: verify
description: Run the full lint + test suite (make lint && make test) and report any failures with context. Use after making changes to confirm nothing is broken.
---

Run `make lint` then `make test` in the project root.

Report the results clearly:
- If both pass: say so in one line.
- If lint fails: show the ruff/mypy output, identify which files and rules are failing, and offer to fix them.
- If tests fail: show the pytest failure output, identify which tests failed and why, and offer to investigate.

Do not modify any files during this skill — only report. If the user wants fixes applied, ask before proceeding.
