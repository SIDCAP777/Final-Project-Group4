# Master Results Comparison

All numbers are **test-set** metrics. Twitter no-hashtags is the honest evaluation; Twitter with hashtags shows leakage.

## Test Accuracy

| Model | SARC | Twitter (with hashtags) | Twitter (no hashtags) |
|---|---|---|---|
| **Logistic Regression** | 0.7238 | 0.9898 | 0.8218 |
| **Linear SVM** | 0.7230 | 0.9910 | 0.8187 |
| **Naive Bayes** | 0.6873 | 0.9250 | 0.7993 |
| **BiLSTM (GloVe)** | 0.7514 | 0.9917 | 0.8296 |
| **DistilBERT** | 0.7760 | 0.9921 | 0.8625 |
| **RoBERTa-base** | 0.7987 | — | 0.8722 |

## Test F1 (macro)

| Model | SARC | Twitter (with hashtags) | Twitter (no hashtags) |
|---|---|---|---|
| **Logistic Regression** | 0.7236 | 0.9898 | 0.8208 |
| **Linear SVM** | 0.7229 | 0.9910 | 0.8179 |
| **Naive Bayes** | 0.6862 | 0.9247 | 0.7964 |
| **BiLSTM (GloVe)** | 0.7507 | 0.9916 | 0.8289 |
| **DistilBERT** | 0.7760 | 0.9921 | 0.8616 |
| **RoBERTa-base** | 0.7987 | — | 0.8716 |

## Hashtag-Leakage Drop on Twitter

Stripping all hashtags exposes label leakage. Drop = (with hashtags) − (no hashtags).

| Model | Twitter w/ hashtags | Twitter no hashtags | Drop (pp) |
|---|---|---|---|
| **Logistic Regression** | 0.9898 | 0.8218 | +16.81 |
| **Linear SVM** | 0.9910 | 0.8187 | +17.22 |
| **Naive Bayes** | 0.9250 | 0.7993 | +12.57 |
| **BiLSTM (GloVe)** | 0.9917 | 0.8296 | +16.20 |
| **DistilBERT** | 0.9921 | 0.8625 | +12.96 |

## Contrast Features Ablation (TF-IDF + 11 VADER features)

Comparison vs the same model without contrast features. Δ = with-contrast − baseline.

| Model | Dataset | TF-IDF only | + contrast | Δ |
|---|---|---|---|---|
| **Logistic Regression** | SARC | 0.7238 | 0.7243 | +0.05 pp |
| **Linear SVM** | SARC | 0.7230 | 0.7232 | +0.02 pp |
| **Naive Bayes** | SARC | 0.6873 | 0.6867 | -0.06 pp |
| **Logistic Regression** | Twitter (with hashtags) | 0.9898 | 0.9914 | +0.16 pp |
| **Linear SVM** | Twitter (with hashtags) | 0.9910 | 0.9924 | +0.14 pp |
| **Naive Bayes** | Twitter (with hashtags) | 0.9250 | 0.9213 | -0.37 pp |
