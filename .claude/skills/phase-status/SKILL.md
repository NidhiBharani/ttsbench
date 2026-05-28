---
name: phase-status
description: Read docs/ttsbench_build_plan.md and the codebase to report which phases are complete, in-progress, or still stubbed. Useful for getting oriented on what to work on next.
---

1. Read `docs/ttsbench_build_plan.md` to get the full list of phases and their exit criteria.
2. For each phase, check whether the exit criteria are met by reading the relevant source files (adapters, benchmarks, schemas, etc.).
3. Report a concise status table:
   - Phase number and name
   - Status: ✓ Complete | ⚡ In progress | ○ Not started
   - One-line note on what's done or what's missing

Keep the report short — one line per phase. If the user wants details on a specific phase, offer to expand on it.
