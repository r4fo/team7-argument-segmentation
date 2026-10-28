# Benchmark

Macro F1 over the 7 discourse types (protocol: [docs/metrics_protocol.md](../docs/metrics_protocol.md)).
No row is filled until `src/evaluate.py` produced it; each filled row links its results JSON.

| # | Architecture | Specific model | Decoder | Seen: overlap F1 | Seen: exact F1 | Unseen: overlap F1 | Unseen: exact F1 | Results file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Lookup | Most-frequent tag per word | Argmax | | | | | |
| 2 | Linear-chain CRF | sklearn-crfsuite, hand features | Viterbi | | | | | |
| 3 | BiLSTM | BiLSTM + GloVe | CRF | | | | | |
| 4 | BERT | bert-base-cased | Softmax | | | | | |
| 5 | BERT | bert-base-cased | CRF | | | | | |
| 6 | DeBERTa | deberta-v3-base | CRF | | | | | |
