# Dataset storage (local only)

Download unmodified PERSUADE 2.0 archives from the **official** README:
https://github.com/scrosseye/persuade_corpus_2.0

Place them in `data/raw/` (ignored). Keep original names, checksums, source date, and extraction procedure in local logs.

Generated `train.jsonl.gz`, `val.jsonl.gz`, `test_seen.jsonl.gz`, `test_unseen.jsonl.gz`, `splits.json`, `stats.json`, `cleaning_report.json`, and `verification_report.json` belong in `data/processed/` (also ignored). **Do not automatically upload them.** Inspect whether any report contains student text or identifying fields before sharing sanitized aggregates.

The repository intentionally tracks only empty-folder markers.

