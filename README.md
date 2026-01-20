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

### Datasets (still gotta check, but looks alright at first glance)

#### Legacy feature-only trainset (no pair IDs)
- `final_trainset.csv`
  - Feature-only dataset used for model training/evaluation in the legacy pipeline.
  - Each row represents one candidate pair, but this file does not include the original item identifiers (only the label + features).
  - Columns:
    - `match` (binary ground truth: 1 = true match, 0 = non-match)
    - `jac_total` (overall Jaccard similarity feature)
    - `title_similarity` (title-level similarity feature)
    - `content_similarity` (content/body-level similarity feature)
    - `numbers_lenmatches ` (numeric overlap / number-match feature; note the trailing space in the column name)

#### Baseline dataset (fixed lexical + overlap features, with pair IDs)
- `final_basic_trainset_fixed.csv`
  - Baseline candidate-pair dataset produced by the fixed legacy-style preprocessing pipeline.
  - Each row represents one candidate link between two items (news article vs CBS report), including pair identifiers, the ground-truth label, and baseline overlap features.
  - Columns:
    - `child_id` (identifier for one side of the candidate pair)
    - `parent_id` (identifier for the other side of the candidate pair)
    - `match` (binary ground truth: 1 = true match, 0 = non-match)
    - `word_jaccard_sim` (Jaccard similarity over word sets)
    - `tax_overlap_count` (overlap count for taxonomy/category terms, if present in the text/metadata)
    - `num_overlap_count` (overlap count of numbers found in both items)
    - `common_word_count` (count of shared words between the two items)

#### Hybrid semantic dataset (S-BERT + spaCy features, with pair IDs)
- `final_hybrid_sbert_trainset_100pct.csv`
  - Hybrid candidate-pair dataset produced by the semantic preprocessing pipeline.
  - Each row represents one candidate link between two items (news article vs CBS report), including pair identifiers, the ground-truth label, and semantic + NLP overlap features.
  - Columns:
    - `child_id` (identifier for one side of the candidate pair)
    - `parent_id` (identifier for the other side of the candidate pair)
    - `match` (binary ground truth: 1 = true match, 0 = non-match)
    - `sbert_sim` (Sentence-BERT embedding cosine similarity)
    - `spacy_sim` (spaCy-based similarity feature)
    - `tax_matches` (taxonomy/category match feature)
    - `ner_overlap` (named-entity overlap feature)
    - `num_matches` (numeric match feature)

#### Topic assignment error export
- `topic_error.csv`
  - Small audit/export file used to inspect topic assignment quality for specific examples.
  - Each row represents one inspected example with its assigned topic and the expected topic label.
  - Columns:
    - `status` (e.g., CORRECT / INCORRECT)
    - `assigned_topic` (topic predicted/assigned by the system)
    - `actual_topic` (reference/ground-truth topic)
    - `text` (the underlying text snippet or record used for topic assignment)

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
