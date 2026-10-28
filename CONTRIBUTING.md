# Team 7 collaboration rules

1. Create a GitHub Issue for each task; agree on definition of done and reproducible command.
2. Branch from `main`: `feature/data-audit`, `feature/verification`, `feature/crf`, `feature/transformer`, `docs/results`, etc.
3. Commit **code and sanitized aggregate metrics only**, never PERSUADE text, zipped archives, private attributes, checkpoint weights, credentials or university IDs.
4. Open a pull request. Another teammate reviews logic, tests, leakage controls, and whether statistics are genuinely measured.
5. After review, merge into `main`. Link Issues/PRs in `contributions/member_0X.md`.
6. For reproducibility track source hashes, random seed, Python/package versions, hardware, prompt/essay split IDs, config, exact metric definition and code commit.
7. Distinguish measured 69% offset mismatches / 99.46% segment retention from **unmeasured** alignment correctness.
8. Never claim a completed feature, accuracy or benchmark before running and checking it.

## Suggested roles (assign in Issues)
Data ingestion and cleaning; independent alignment validation; classical models; neural models; evaluation/reporting. Everyone understands the whole pipeline and reviews at least one other member's work.

