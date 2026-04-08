# Final-Project-Group4

## Sarcasm-Aware Sentiment Analysis Using Contextual Contrast Detection

**Author:** Siddardha Reddy Yaraguti  
**GWID:** G39234370  
**Course:** Natural Language Processing

---

### Project Overview

Sentiment analysis models consistently fail on sarcastic text. This project builds a sarcasm-aware sentiment analysis pipeline that detects the contextual contrast within a sentence and uses it to correct sentiment predictions. The approach compares classical models, LSTM, and fine-tuned transformers, with interpretability analysis using LIME and SHAP.

---

### Repository Structure

```
Final-Project-Group4/
├── Group-Proposal/                    # Project proposal (PDF)
├── Final-Group-Project-Report/        # Final report (PDF)
├── Final-Group-Presentation/          # Presentation slides (PDF)
├── Code/                              # All project code with README
└── README.md                          # This file
```

---

### Dataset

- **SARC (Self-Annotated Reddit Corpus):** ~1.3M sarcastic Reddit comments
- **Twitter Sarcasm Dataset:** Twitter sarcastic comments (supplementary)

### Methods

- Classical Models (TF-IDF + Logistic Regression, Naive Bayes)
- Recurrent Networks (LSTM)
- Pretrained Transformers (DistilBERT, sentence-transformers)
- Interpretability (LIME, SHAP)
