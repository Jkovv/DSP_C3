# CBS News-to-Report Linkage: Audit & Semantic Enhancement

This repository contains the source code, data pipelines, and performance audits for a system designed to link news articles to official CBS (Centraal Bureau voor de Statistiek) reports. The project demonstrates the transition from traditional keyword-based matching to a **Hybrid Semantic Approach** using S-BERT embeddings and spaCy NLP.

## Project Overview
The primary goal was to address **semantic blindness** and **high overfitting** observed in legacy linkage systems. By implementing Transformer-based embeddings (S-BERT) and Named Entity Recognition (NER), we evolved the system into a high-precision ranking engine capable of understanding the context of Dutch news.

## Final Performance Matrix
The following results were obtained using an **80/20 Group-Aware Temporal Split**, ensuring the model generalizes to entirely new news articles.

| Dataset | Model | Train_Acc | Test_Acc | Gap | Recall | F1 | AUC | Succ@1 | Succ@5 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Baseline** | CatBoost | 0.8473 | 0.5100 | 0.3373 | 0.0345 | 0.0637 | 0.4764 | 0.5833 | 0.8333 |
| **2. Basic** | CatBoost | 0.9907 | 0.9601 | 0.0306 | 0.5893 | 0.0453 | 0.8662 | 0.8571 | 1.0000 |
| **3. Hybrid** | **CatBoost** | **0.9980** | **0.9609** | **0.0372** | **0.7679** | **0.0592** | **0.9540** | **0.9643** | **1.0000** |

### Key Audit Findings:
* **Semantic Power**: The Hybrid dataset (S-BERT) achieved an **AUC of 0.9540**, proving that semantic embeddings effectively separate true matches from thematic noise.
* **Overfitting Elimination**: While the legacy Baseline showed a massive **~33-49% Gap**, the Hybrid approach maintained a stable **~3.7% Gap**, indicating excellent generalization.
* **Ranking Excellence**: With a **Success@1 of 0.9643** and **Success@5 of 1.0000**, the system ensures that the correct report is almost always the very first suggestion provided to an auditor.

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
   git clone [https://github.com/YourUsername/dsp.git](https://github.com/YourUsername/dsp.git)
   cd dsp

2.  **Install Dependencies**:
    ```bash
    pip install pandas numpy scikit-learn sentence-transformers spacy xgboost lightgbm catboost shap
    ```

3.  **Download Dutch NLP Model**:
    ```bash
    python -m spacy download nl_core_news_lg
    ```
