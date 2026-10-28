# Data pipeline and cleaning report

**Status: template.** Replace TODOs with results from your actual pipeline and include a command + input checksums that reproduces each number.

## Source and schemas
Official PERSUADE 2.0: https://github.com/scrosseye/persuade_corpus_2.0
- Exact source archives, hashes, extraction and table names: TODO
- Column names and any differences between 1.0/2.0: TODO
- Essay count / segment count and missing annotations: TODO

## Offset mismatch audit (measured by Team 7)
- Reported **69%** original official character-offset mismatch.
- **Numerator** (number of mismatching segments): TODO
- **Denominator** (segments checked): TODO
- Check: `full_text[start:end]` versus `discourse_text`? Specify whitespace normalization, inclusive/exclusive end, Unicode normalization, `\r\n` versus `\n`, and indexing convention.
- Drift analysis (per segment, essay position and class): TODO
- Do not infer official annotation error without ruling out text-version and offset-convention mismatch.

## Realignment (measured by Team 7)
- Search/nearest-candidate algorithm and tie-breaks: TODO — include exact implementation link
- Repeated phrase ambiguity and cases not found: TODO
- Reported **99.46% retention**, NOT accuracy.
- **Numerator** kept: TODO; **denominator** original segments: TODO; removed/unsolved counts: TODO.
- Retention by class, prompt, essay position and offset shift: TODO

## BIO conversion
- Annotation units: word boundaries / tokenizer: TODO
- Seven semantic types: Lead, Position, Claim, Counterclaim, Rebuttal, Evidence, Concluding Statement
- 15 tags under strict IOB2 (B/I of each type + O).
- Rules for overlapping, adjacent, missing, ambiguous and partial word spans: TODO
- Assert `0 <= start < end <= len(full_text)` for exclusive end; verify word/span alignment.

## Data partitioning
- Train/validation/test split at **essay** level, not segment level.
- For cross-prompt tests, leave held-out prompt IDs outside training and validation.
- Source split policy, prompt IDs, random seed, duplicates/near duplicates: TODO.
- Never tune preprocessing on final test labels.

## Validation and limitations
- Automatic text equality test is necessary but insufficient for true correct-instance matching.
- Do separate blinded human validation as in `verification_sample.md`.
- Disclose irrecoverable spans, ambiguous repeats, residual boundary uncertainty, class imbalance and long essays.

