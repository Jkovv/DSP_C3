# CBS News-to-Report Linkage: Audit & Semantic Enhancement

This repository contains the source code, data pipelines, and performance audits for a system designed to link news articles to official CBS (Centraal Bureau voor de Statistiek) reports. The project demonstrates the transition from traditional keyword-based matching to a **Hybrid Semantic Approach** using S-BERT embeddings and spaCy NLP.

## Project Overview
The primary goal was to address **semantic blindness** and **high overfitting** observed in legacy linkage systems. By implementing Transformer-based embeddings (S-BERT) and Named Entity Recognition (NER), we evolved the system into a high-precision ranking engine capable of understanding the context of Dutch news.

## Final Performance Matrix
The following results were obtained using an **80/20 Group-Aware Temporal Split**, ensuring the model generalizes to entirely new news articles.

# TODO: FIX THE TABLE 

### Key Audit Findings:
* **Semantic Power**: The Hybrid dataset (S-BERT) achieved an **AUC of 0.9540**, proving that semantic embeddings effectively separate true matches from thematic noise.
* **Overfitting Elimination**: While the legacy Baseline showed a massive **~33-49% Gap**, the Hybrid approach maintained a stable **~3.7% Gap**, indicating excellent generalization.
* **Ranking Excellence**: With **Success@1** - **Success@5** scores, showcasing the proof of concept for introducing semantic-clustering in the recommendations for manual reviewers.

## Methodology
To ensure scientific validity, the evaluation suite implements:
1.  **Group-Aware Splitting**: Prevents "Data Leakage" by ensuring all rows related to a single news article are kept together in either the training or test set.
2.  **Quantile Thresholding**: Because matches are rare (1:24 ratio), we use a 96th percentile threshold to force the model to rank the most likely candidates, overcoming "model conservatism."
3.  **Success@K Metrics**: Focuses on the system's utility as a recommendation engine for human auditors.

## File Structure

### Preprocessing Pipelines
* `preprocessing.py`: The **Hybrid** pipeline. Uses `SentenceTransformer` and `spaCy` to extract deep semantic features and NER overlaps.
* `preprocessing_2.py`: The **Fixed Basic** pipeline. A mathematically corrected version of legacy logic using Jaccard Similarity and numerical overlap.

### Evaluation & Audit
* `model_comparison.py`: The core audit engine. Trains and compares Balanced RF, XGBoost, LightGBM, and CatBoost. Generates ROC curves and Confusion Matrices.
* `error_cases.py`: Qualitative audit tool that exports True Positives and False Positives to `classification_examples.csv` for manual review.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Jkovv/dsp.git
   cd dsp

2.  **Install Dependencies**:
    ```bash
    pip install pandas numpy scikit-learn sentence-transformers spacy xgboost lightgbm catboost shap
    ```

3.  **Download Dutch NLP Model**:
    ```bash
    python -m spacy download nl_core_news_lg
    ```
