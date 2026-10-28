# Third-party data licensing

The [MIT LICENSE](LICENSE) applies to Team 7's original source code only.

`data/raw/` contains the PERSUADE 2.0 archives exactly as downloaded from the
[official repository](https://github.com/scrosseye/persuade_corpus_2.0), whose
README states a **Creative Commons Attribution 4.0 International (CC BY 4.0)**
license: https://creativecommons.org/licenses/by/4.0/

`data/processed/` is our derivative of that data, distributed under the same
license.

**Attribution:** Crossley, S. A., Baffour, P., Tian, Y., Franklin, A., Benner, M., & Boser, U. (2024). *A large-scale corpus for assessing written argumentation: PERSUADE 2.0*. Assessing Writing, 61.

**Changes made in `data/processed/`** (details in [docs/data_pipeline.md](docs/data_pipeline.md)):

- character offsets of annotated segments re-aligned to the essay text;
- demographic columns (gender, race/ethnicity, disability, ELL and economic status) removed;
- essays tokenized and labelled with BIO tags;
- essays split into train / val / test_seen / test_unseen.

**Privacy.** The essays were written by students in grades 6–12. The corpus
authors released them publicly with personal details already masked (for
example, place names replaced by `LOCATION`). We redistribute the official
archives unchanged and add no information about the writers; our processed
files contain no demographic fields.

| Archive | SHA-256 |
| --- | --- |
| `persuade_corpus_2.0_train.zip` | `561bdde982beee855881a7c22bd38676ef5dd2d46efe839078e8cc4682c76a9c` |
| `persuade_corpus_2.0_test.zip` | `51c907f90b1303d610e8fcd8c0cbf3656ec88116bc34b60b93794affa77facb2` |
