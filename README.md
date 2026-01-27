# CBS News-to-Report Linkage: Audit & Semantic Enhancement

This repository contains the source code, data pipelines, and performance audits for a system designed to link news articles to official CBS (Centraal Bureau voor de Statistiek) reports. The project demonstrates the transition from traditional keyword-based matching to a **Hybrid Semantic Approach** using S-BERT embeddings and spaCy NLP.

---

## Project Overview - Algorithm Improvement
The primary goal was to address **semantic blindness** and **high overfitting** observed in legacy linkage systems. By implementing Transformer-based embeddings (S-BERT) and Named Entity Recognition (NER), we evolved the system into a high-precision ranking engine capable of understanding the context of Dutch news.

---

## Final Performance Matrix
The following results were obtained using an **80/20 Group-Aware Split**, ensuring the model generalizes to entirely new news articles.

| Dataset | Model | Tr_Acc | Ts_Acc | Gap | F1 | AUC | Recall | Succ@1 | Succ@2 | Succ@3 | Succ@4 | Succ@5 | Time(s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Baseline** | Original_RF | 1.000 | 0.483 | 0.517 | 0.482 | 0.484 | 0.503 | 0.642 | - | - | - | - | 0.49 |
| **1. Baseline** | Balanced_RF | 1.000 | 0.487 | 0.513 | 0.473 | 0.482 | 0.483 | 0.633 | - | - | - | - | 0.45 |
| **1. Baseline** | CatBoost | 0.859 | 0.513 | 0.346 | 0.500 | 0.469 | 0.510 | 0.661 | - | - | - | - | 1.10 |
| **1. Baseline** | XGBoost | 0.999 | 0.507 | 0.492 | 0.493 | 0.478 | 0.503 | 0.624 | - | - | - | - | 0.21 |
| **1. Baseline** | AdaBoost | 0.582 | 0.480 | 0.102 | 0.466 | 0.437 | 0.476 | 0.606 | - | - | - | - | 0.27 |
| **2. Basic** | Original_RF | 1.000 | 0.967 | 0.033 | 0.582 | 0.867 | 0.582 | 0.600 | 0.709 | 0.745 | 0.800 | 0.836 | 0.89 |
| **2. Basic** | Balanced_RF | 1.000 | 0.964 | 0.036 | 0.554 | 0.860 | 0.564 | 0.618 | 0.655 | 0.691 | 0.764 | 0.782 | 0.80 |
| **2. Basic** | CatBoost | 0.965 | 0.961 | 0.005 | 0.509 | 0.855 | 0.509 | 0.636 | 0.673 | 0.745 | 0.764 | 0.782 | 0.91 |
| **2. Basic** | XGBoost | 0.994 | 0.962 | 0.032 | 0.527 | 0.859 | 0.527 | 0.636 | 0.709 | 0.782 | 0.800 | 0.818 | 0.12 |
| **2. Basic** | AdaBoost | 0.971 | 0.959 | 0.012 | 0.486 | 0.797 | 0.491 | 0.618 | 0.727 | 0.764 | 0.800 | 0.836 | 0.45 |
| **3. Hybrid** | Original_RF | 1.000 | 0.969 | 0.031 | 0.618 | 0.872 | 0.618 | 0.655 | 0.727 | 0.800 | 0.800 | 0.818 | 0.92 |
| **3. Hybrid** | Balanced_RF | 1.000 | 0.969 | 0.031 | 0.618 | 0.883 | 0.618 | 0.655 | 0.727 | 0.800 | 0.855 | 0.855 | 0.86 |
| **3. Hybrid** | **CatBoost** | **0.977** | **0.972** | **0.004** | **0.655** | **0.908** | **0.655** | **0.673** | **0.782** | **0.873** | **0.909** | **0.927** | **1.14** |
| **3. Hybrid** | XGBoost | 0.999 | 0.969 | 0.030 | 0.618 | 0.863 | 0.618 | 0.691 | 0.745 | 0.800 | 0.800 | 0.818 | 0.13 |
| **3. Hybrid** | AdaBoost | 0.972 | 0.966 | 0.006 | 0.577 | 0.877 | 0.582 | 0.691 | 0.764 | 0.800 | 0.873 | 0.891 | 0.59 |

> **Note on Baseline Metrics:** Success@2-5 metrics for the Baseline dataset are marked as `-` because the original 1:1 data structure makes these rankings mathematically trivial and incomparable to the 1:24 ranking challenge used in the newer datasets.

### Key Audit Findings:
* **Semantic Power**: The Hybrid dataset (CatBoost) achieved a peak **AUC of 0.908**, proving that semantic embeddings effectively separate true matches from noise in a complex 1:24 environment.
* **Overfitting Elimination**: While the legacy Baseline showed a massive **~51% Gap**, the Hybrid approach (CatBoost) maintained a stable **0.4% Gap**, indicating excellent generalization.
* **Ranking Excellence**: The system achieved a **Success@5 of 0.927** on Hybrid data, confirming its utility as a high-precision recommendation tool for auditors.

---

## Qualitative Audit (Classification Examples)

Below are representative examples from the audit, showing how the Hybrid model interprets news context compared to official CBS labels.

| Status | Assigned Topic | Actual Topic | Article Snippet |
| :--- | :--- | :--- | :--- |
| **CORRECT** | Government & Politics | Government & Politics | "In 2021, nearly 69,000 new-build homes were completed... the housing stock grew by 0.9% to over 8 million homes." |
| **ERROR** | Health & Welfare | Government & Politics | "In 2021, nearly 171,000 people died, 16,000 more than expected... excess mortality was higher in the 50-80 age groups." |

**Audit Note:** *Errors* often occur due to "Label Ambiguity"-the model logically links death statistics to *Health*, while the CBS ground truth categorizes them under *Government/Politics*.

---

## Semantic Similarity Proof

This audit examines the **Semantic Neighborhood** of news articles within the S-BERT vector space. By comparing a **Seed** (a human-validated link) to its closest **Neighbor**, we can identify where the model finds "Event Twins," where it matches perfectly, and where it gets distracted by institutional "noise."

### Qualitative Audit Table

| Pair ID | Result | Actual Topic | Predicted Topic | Seed Excerpt (Translated) |
| :--- | :--- | :--- | :--- | :--- |
| **958850** | **MATCH** | Gov & Politics | Gov & Politics | "Construction of new houses increases rapidly... almost 69,000 new homes in 2021." |
| **958806** | **MATCH** | Gov & Politics | Gov & Politics | "Nearly 6,000 people died in Drenthe in 2021... 10% excess mortality nationwide." |
| **1037677** | **MATCH** | Agriculture | Agriculture | "Nitrogen excretion is falling, while phosphate is rising in 2022." |
| **958761** | **EVENT TWIN** | Health | Population | "In 2021, 16,000 more people died than expected... mortality was increased in every age group." |
| **958921** | **BIAS CLASH** | Housing | Agriculture | "Over 77,000 houses added in 2021... the housing stock grew by 0.9%." |

---

### Understanding the Results

The audit reveals three specific patterns in how the model processes the CBS dataset:

#### 1. The "Golden" Matches (Success)
In cases like **958850** and **1037677**, the model is extremely reliable. The semantic distance between a news clip and its neighbor is nearly zero because they share specific technical vocabulary (e.g., "housing stock," "nitrogen excretion").
* **Insight**: When categories are consistently labeled in the database, the S-BERT embedding space perfectly clusters related news.

#### 2. The "Event Twin" Discovery (Hidden Truth)
Pair **958761** shows a semantic similarity of **0.983**, yet the human labels are different ("Health" vs. "Population").
* **Insight**: These articles are functionally identical. The "Actual Topic" mismatch is a result of inconsistent human filing or the legacy algorithm's failure to sync. The model correctly identifies these as the same story, proving it is often more accurate than the original database labels.



#### 3. The Boilerplate Bias (Institutional Noise)
In Pair **958921**, the model links a Housing article to a Nitrogen article with **0.989** similarity.
* **Insight**: This is the "Boilerplate Bias." Both articles contain heavy CBS footers and standardized introductory phrasing (e.g., *"Dit meldt het CBS op basis van nieuwe cijfers"*). This "institutional noise" creates a false gravitational pull in the vector space.
* **Solution**: This confirms why the **Hybrid CatBoost Model** is necessary. By adding additional metadata features and confidence weighting, the hybrid model can "see through" this stylistic similarity to focus on the unique thematic content.



### Audit Conclusion
While raw semantic similarity can be misled by shared institutional style, the **92.7% Success@5** rate proves the system is robust. By identifying "Event Twins," the model acts as a powerful auditing tool to help CBS clean up its fragmented database and unify inconsistent labeling across departments.

---

## Methodology
To ensure scientific validity, the evaluation suite implements:
1.  **Group-Aware Splitting**: Prevents "Data Leakage" by ensuring all rows related to a single news article (`child_id`) are kept together in either the training or test set.
2.  **Quantile Thresholding**: Due to the 1:24 class imbalance, we use a 96th percentile threshold to force the model to rank candidates effectively, overcoming "model conservatism."
3.  **Success@K Metrics**: Focuses on the system's utility as a recommendation engine where a prediction is "correct" if the ground truth appears in the Top-K candidates.

---

## Installation

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/Jkovv/DSP_C3.git](https://github.com/Jkovv/DSP_C3.git)
    cd DSP_C3
    ```

2.  **Install Dependencies**:
    ```bash
    pip install pandas numpy scikit-learn sentence-transformers spacy xgboost catboost
    ```

3.  **Download Dutch NLP Model**:
    ```bash
    python -m spacy download nl_core_news_lg
    ```

---

## Project Overview - Dashboard Improvement
(short intro.)

---

## High-level Technical Userflow

Dashboard
   ↓
Article Selection (single or bulk)
   ↓
Verification Page ⇄ Article Detail Modals
   ↓
Verification Decision
        ↓
[Confirm | Reclassify | Defer | Close]
        ↓
Check for Similar Articles
      ↙          ↘
Similar Found   None Found
     ↓              ↓
Bulk Action     Single Update
     ↓              ↓
Update Database
        ↓
Update Metrics & Counters
        ↓
Return to Dashboard

---

## UAT (User Testing)
(incl. methodolody)

---

## Results - Dashboard Improvement
(1. the instruction of how to use the sample code for frontend dashboard dev;
 2. result from user testing.)

(... to be edited)
