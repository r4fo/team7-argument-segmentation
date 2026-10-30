### DeBERTa-v3 + CRF Model Training

- **Model:** `microsoft/deberta-v3-base` with a CRF layer
- **Data:** 11,721 articles (16,772 windows)
- **Setup:** 3 epochs on Google Colab (Tesla T4 GPU), ~37 min/epoch, 116 min total
- **Training:** Loss decreased from 661 to 496 over training
- **Outputs:** Predictions and evaluation results generated for the validation, 
- `test_seen`, and `test_unseen` splits

**Results:** This is the best-performing model in the project, scoring **0.652** 
on seen topics and **0.647** on unseen topics. That is an improvement of about 
**0.07** over the BERT baseline.