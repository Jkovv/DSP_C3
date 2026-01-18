# CBS News-to-Report Linkage: Audit & Semantic Enhancement

This repository contains the source code and datasets for an audit and enhancement of the CBS (Centraal Bureau voor de Statistiek) news-linkage system. The project moves beyond simple keyword matching to a **Hybrid Semantic Approach** using S-BERT and spaCy.

## Project Overview
The goal of this project was to solve the critical issue of **overfitting** and **semantic blindness** in the original CBS linkage system. By introducing Transformer-based embeddings (S-BERT) and Named Entity Recognition (NER), we transformed the system from a basic word-counter into a semantic ranking engine.

## File Descriptions

### Datasets
* **`final_hybrid_sbert_trainset_100pct.csv`**: The "Champion" dataset. It includes S-BERT semantic embeddings, spaCy NER overlaps, and numerical matches. This dataset provides the strongest signal for high-precision ranking.
* **`final_basic_trainset.csv`**: A "Fixed Baseline" dataset. It follows the original CBS logic but fixes mathematical bugs (e.g., implementing the actual Jaccard Index) and data mapping errors. It does **not** use S-BERT or spaCy.
* **`final_trainset.csv`**: The **Legacy Baseline**. This is the original data provided by CBS. Our audit proves this data leads to severe overfitting (Gap > 50%) due to noisy features and incorrect similarity metrics.

### Preprocessing Pipelines
* **`preprocessing.py`**: The advanced pipeline. It uses `SentenceTransformer` (S-BERT) and `spaCy` (`nl_core_news_lg`) to extract deep semantic features. It is optimized for large-scale data (340k+ articles) using batching and disk-streaming.
* **`preprocessing_fixed_basic.py`**: A refined version of the original CBS preprocessing. It removes heavy AI dependencies, utilizing optimized set operations and a mathematically correct Jaccard Similarity calculation.

### Model & Evaluation
* **`model_comparison.py`**: The core audit script. It trains and evaluates multiple classifiers (Random Forest, XGBoost, LightGBM, CatBoost) across all datasets to produce a comparative performance matrix.

## Key Metrics Explained

Our analysis evaluates the system using two distinct lenses:

### 1. Generalization (The "Gap")
The difference between Training and Testing accuracy. 
* **Legacy Baseline:** Showed a **~50% Gap**, indicating the model was simply memorizing words.
* **New RF / Hybrid Model:** Achieved a **<0.2% Gap**, proving the model actually understands the context.

### 2. Ranking Success (Success@K)
* **Success@1**: Tells us if the correct CBS report is the **very first** suggestion. The Hybrid model achieves near **1.0000 (100%)** accuracy here, effectively automating the linkage process.
* **Success@5**: Checks if the match is in the top 5 suggestions. This represents the "search assistant" capability.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Jkovv/dsp.git
    cd dsp
    ```

2.  **Install Dependencies**:
    ```bash
    pip install pandas numpy scikit-learn sentence-transformers spacy xgboost lightgbm catboost
    ```

3.  **Download Dutch NLP Model**:
    ```bash
    python -m spacy download nl_core_news_lg
    ```
