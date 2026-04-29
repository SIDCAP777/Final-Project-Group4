# Sarcasm Aware Sentiment Analysis Using Contextual Contrast Detection

**Author:** Siddardha Reddy Yaraguti
**GWID:** G39234370
**Group:** Group 4 (solo)

## Project overview

Sentiment analysis models often fail on sarcastic text because the surface words say one thing while the writer means the opposite. This project trains six model families on two sarcasm datasets (SARC and Twitter), runs careful ablations to expose label leakage in the Twitter corpus, and uses LIME and SHAP to understand what each model has actually learned. The flagship result is RoBERTa with parent comment context on SARC at 79.87 percent test accuracy.

## Headline results

| Model | SARC | Twitter (no hashtags) |
|---|---|---|
| Logistic Regression | 0.7238 | 0.8218 |
| Linear SVM | 0.7230 | 0.8187 |
| Naive Bayes | 0.6873 | 0.7993 |
| BiLSTM with GloVe | 0.7514 | 0.8296 |
| DistilBERT | 0.7760 | 0.8625 |
| **RoBERTa with parent context** | **0.7987** | **0.8722** |

The often reported 99 percent accuracy on Twitter is mostly hashtag leakage. Stripping all hashtags drops accuracy by 13 to 17 percentage points across every model family. The numbers above are the honest evaluation.

## Datasets

* **SARC (Self Annotated Reddit Corpus):** 1,010,771 Reddit comments with parent context. Self labeled by writers using the `/s` tag. Perfectly class balanced.
* **Twitter Sarcasm Dataset:** 43,240 tweets after binarizing to sarcasm vs regular.

## Methods

* Classical: Logistic Regression, Linear SVM, Multinomial Naive Bayes, all on TF-IDF (word and character n-grams)
* Recurrent: BiLSTM with GloVe 100d initialization
* Transformers: DistilBERT and RoBERTa base
* Contrast features: 11 VADER based features tested as an ablation (no measurable gain over TF-IDF)
* Interpretability: LIME and SHAP on RoBERTa, with cross method agreement reported
* Demo: side by side Streamlit app comparing Twitter RoBERTa and SARC RoBERTa

## Repository structure

Final-Project-Group4/
- Group-Proposal/                    Project proposal PDF
- Final-Group-Project-Report/        Final report (markdown source plus PDF)
- Final-Group-Presentation/          Presentation slides PDF
- Code/                              All source code (see Code/README.md)
- outputs/                           Metrics, plots, interpretability outputs
- config.yaml                        Central config for paths and hyperparameters
- requirements.txt                   Python dependencies
- README.md                          This file

## Quick start

    git clone https://github.com/SIDCAP777/Final-Project-Group4.git
    cd Final-Project-Group4
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

Set the data paths in `config.yaml` (SARC, Twitter, GloVe). Then follow the running order in `Code/README.md`.

## Hardware used

AWS EC2 g5.2xlarge: NVIDIA A10G GPU (24 GB VRAM), 32 GB RAM, Ubuntu 24, CUDA 12.2. Total training time about 14 hours, dominated by RoBERTa.

## Key findings

1. **Hashtag leakage is real and large.** Twitter accuracy drops 13 to 17 points uniformly when all hashtags are stripped. Future Twitter sarcasm work should report the no hashtags ablation.
2. **Conversational context is the strongest single signal.** Adding the parent comment to RoBERTa SARC gives a 2.27 point gain over DistilBERT.
3. **Hand engineered contrast features do not help on top of strong text representations.** The deltas are within noise (-0.4 to +0.2 percentage points).
4. **LIME and SHAP cross method agreement reveals what the model has learned.** On Twitter the model has learned a clean ironic intensifier template (love, boring, another, quite, etc.) with 53 percent agreement between methods. On SARC the cues are more diffuse (33 percent agreement), reflecting that the model relies more on parent context than lexical patterns.
