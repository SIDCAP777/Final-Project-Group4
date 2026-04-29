# Code

All scripts read paths and hyperparameters from `config.yaml` at the repository root. Run everything from the project root with the venv activated.

```bash
cd ~/Final-Project-Group4
source venv/bin/activate
```

## Required data

Set these paths in `config.yaml`:

* SARC: `train-balanced-sarcasm.csv`
* Twitter: `train.csv` and `test.csv`
* GloVe 6B 100d: `glove.6B.100d.txt`

These files are not in git because they are too large.

## Running order

```bash
# 1. Train classical models (LogReg, SVM, NB) on SARC and Twitter
python Code/training/train_classical.py

# 2. Train BiLSTM with GloVe
python Code/training/train_lstm.py

# 3. Fine tune DistilBERT
python Code/training/train_distilbert.py

# 4. Fine tune RoBERTa (uses parent context on SARC)
python Code/training/train_roberta.py

# 5. Hashtag leakage ablation on Twitter
python Code/training/twitter_leakage_test_classical.py
python Code/training/twitter_leakage_test_lstm.py
python Code/training/twitter_leakage_test_distilbert.py

# 6. Contrast feature ablation
python Code/training/train_classical_with_contrast.py

# 7. LIME and SHAP on RoBERTa
python Code/interpretability/lime_explainer.py --model roberta --dataset sarc
python Code/interpretability/lime_explainer.py --model roberta --dataset twitter_no_hashtags
python Code/interpretability/shap_explainer.py --model roberta --dataset sarc
python Code/interpretability/shap_explainer.py --model roberta --dataset twitter_no_hashtags

# 8. Aggregate results and build report figures
python Code/aggregation/aggregate_results.py
python Code/aggregation/generate_report_plots.py

# 9. Streamlit demo
streamlit run Code/streamlit_demo/app.py
```

## Folder map

| Folder | What it has |
|---|---|
| `data/` | Dataset loaders and the text preprocessor |
| `features/` | TF-IDF, GloVe vocab and embedding loader, VADER contrast features |
| `models/` | Model definitions (classical, LSTM, DistilBERT, RoBERTa) |
| `training/` | One training script per model family, plus the hashtag and contrast ablations |
| `evaluation/` | Metric computation and confusion matrix plotting |
| `interpretability/` | LIME and SHAP explainers (model agnostic, pick via `--model` flag) |
| `aggregation/` | Builds the master comparison tables and the report figures |
| `streamlit_demo/` | Side by side demo app comparing Twitter RoBERTa and SARC RoBERTa |
| `utils/` | Config loader, seeding, logging |

## Outputs

Results go to `outputs/results/` (metric JSONs), `outputs/plots/` (confusion matrices and report figures), `outputs/interpretability/` (LIME and SHAP), and `outputs/aggregation/` (master tables).

Trained model weights go to `saved_models/`, which is gitignored.

## Hardware

Trained on AWS EC2 g5.2xlarge with one NVIDIA A10G GPU (24 GB VRAM) and 32 GB RAM.