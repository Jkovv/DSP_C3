# CBS News-to-Report Linkage: Audit & Semantic Enhancement

This repository contains the source code, data pipelines, and performance audits for a system designed to link news articles to official CBS (Centraal Bureau voor de Statistiek) reports. The project demonstrates the transition from traditional keyword-based matching to a **Hybrid Semantic Approach** using S-BERT embeddings and spaCy NLP, as well as an improved User- and Workflow through a redesigned dashboard.

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
This is the high-level technical flow of the CBS Matching Platform, illustrating the end-to-end validation process from dashboard selection through verification, optional bulk reclassification, database updates, and return to the dashboard.
```
Dashboard
   ↓
Article Selection (single or bulk)
   ↓
Verification Page ⇄ Article Detail Modals
   ↓
Verification Decision
        ↓
[ Confirm | Reclassify | Defer | Close ]
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
```

---
## User Evaluation Session

To evaluate the usability and workflow design of the developed dashboard prototype, a user validation session was conducted with CBS analysts who actively use the current Matching Platform in practice.

### Setup and Methodology

The session was held remotely via Microsoft Teams and followed a predefined timetable and testing script. Participants were asked to share their screen and complete a series of predefined tasks while verbalizing their thoughts using a Think Aloud approach.

The evaluation combined the following usability methods:

- **Hallway Usability Testing** – short goal-based tasks to quickly identify navigation and workflow issues  
- **Think Aloud Study** – capturing user reasoning and interaction behavior  
- **A/B Interface Comparison** – comparing the current CBS system with the developed dashboard prototype  

The session consisted of two main parts:

1. **Navigation testing** within the developed dashboard  
2. **Workflow comparison**, where identical matching tasks were executed in both the current system and the prototype  

Task completion time and task success were recorded for each task.

---

### Task Performance Comparison

The table below shows the results for Part 2 (workflow tasks) in both systems, with task time presented in seconds.

| Task | Description | Current System Success | Current System Time | Prototype Success | Prototype Time |
|------|------------|----------------------|--------------------|------------------|---------------|
| 1 | Search CBS article and open page | Yes | 7 | Yes | 45 |
| 2 | Filter media articles | – | – | Yes | 10 |
| 3 | Match three media articles | Yes | 21 | Yes | 110 |
| 4 | Open CBS article and citing media article | – | – | – | – |

---

### Key Findings

#### Navigation and Interface

- Users were able to complete most navigation tasks successfully  
- Some confusion occurred between CBS articles (parents) and media articles (children) due to layout emphasis  
- The visual focus on CBS articles caused users to initially overlook the media article list  
- The “Matchen” button label was interpreted as directly confirming a match instead of opening the matching page  

Users suggested switching the parent-child layout to better reflect their usual workflow, which starts from CBS publications.

#### Workflow Comparison

**Current system limitations:**

- Limited visibility of CBS article content during matching  
- Frequent switching between the matching platform and media portal  
- No bulk matching functionality  

**Prototype improvements observed:**

- Matching tasks could be performed within a single interface  
- Confidence scores helped prioritize easy vs. complex matches  
- Highlighted keywords made match reasoning more transparent  
- Bulk verification enabled multiple high-confidence matches to be confirmed at once  

Some usability issues were noted in the prototype, including:

- Lack of scrolling in certain modal views  
- Desire to automatically open the next article after completing a match  
- Clearer handling of unmatched (orphan) articles  

#### Overall Feedback

Participants described the developed dashboard as:

- Clean and visually calm  
- Easy to use after a short learning period  
- Well structured for matching-focused tasks  

Positive elements highlighted included:

- Confidence-based grouping of matches  
- Transparent keyword highlighting  
- Centralized workflow without switching systems  

There were no major usability blockers identified.

---

### Key Takeaways Usability Evalutation

The user evaluation indicates that the redesigned dashboard better supports the matching workflow compared to the current system. The prototype aligns more closely with analyst working patterns by focusing on task-oriented validation, transparency, and reduced manual interactions.

While some layout and interaction refinements are needed, the results suggest that the user-centered design approach improves usability and workflow clarity.


---

## Results – Dashboard Improvement

This section presents (1) instructions for using the provided sample frontend code for dashboard development and (2) the main features of the final dashboard iteration after incorporating feedback from the user evaluation session. The dasboard prototype can be interacted with through its [Figma Site](https://bear-thumb-72325869.figma.site/) and a walkthrough of it can be watched [here](https://youtu.be/JFvyypkzkgs)

### Using the Prototype Frontend Code

The repository includes sample frontend code to demonstrate the structure and interaction logic of the developed dashboard prototype. This code serves as a reference implementation for further development and experimentation.

To use the sample code:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Jkovv/DSP_C3.git
   cd DSP_C3
   ```

2.  **Install Dependencies**:
    ```bash
    npm install
    ```

3.  **Run the development server:**:
    ```bash
    npm start
    ```

    ## Dashboard Prototype Setup and Usage

The interactive dashboard prototype is built with **TypeScript and React**. It serves as a functional front-end concept for the redesigned CBS Matching Platform, focusing on visualising validation workflows and usability improvements. 

### Technical Environment

To replicate or run the prototype, a standard **React development environment** is required.

### Reconstruction Steps

The dashboard environment can be reconstructed by extracting the following archives into your React project and using the `App.tsx` entry point:

1. **App.tsx**: The main application entry point that orchestrates layout, routing logic, and state management.
2. **components.zip**: Contains all React components used in the interface, including TypeScript definitions for type safety.
3. **data.zip**: Includes sample datasets representing the structure of CBS articles and media articles, allowing for functional testing without requiring access to classified company data.
4. **styles.zip**: Contains all styling assets, including Tailwind CSS configurations and the global CSS file implementing CBS corporate styling.

To use the sample code:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Jkovv/DSP_C3.git
   cd DSP_C3
   ```

2. **Extract Prototype Assets**:
   ```bash
    unzip components.zip -d src/components
    unzip data.zip -d src/data
    unzip styles.zip -d src/styles
   ```
   Note: Ensure the target directories (src/components, etc.)     match your specific React project structure.
   
4. **Install Dependencies**:
    ```bash
    npm install
    ```

5.  **Run the development server:**:
    ```bash
    npm start
    ```


## Final Dashboard Features (After Usability Feedback)

After processing feedback from the user evaluation session, the dashboard prototype was refined into a more workflow-focused and user-centered interface. The main improvements and features include:

---

### Improved User Flow and Interaction Design
* **Smoother Navigation:** More intuitive layout designed to reduce cognitive load.
* **CBS Corporate Identity:** Use of official internal CBS house-style for increased familiarity and trust.

### Bulk Verification for High-Confidence Matches
* **One-Click Verification:** Designed for matches with confidence scores between **85–99%**.
* **Default Selection:** All related media articles are selected by default to speed up batch processing.
* **Hover Snippets:** Quick content previews available on hover to minimize page transitions.

### Minimalist and Structured Search
* **Dual Search Toggle:** Easily switch between CBS articles and media articles.
* **Breadcrumb Navigation:** Enhanced location awareness within the application hierarchy.
* **Advanced Filtering:** Granular options for structured searching across various metadata.

### Compact Article Pages
* **Decision-Focused UI:** Only essential information is displayed to streamline matching decisions.
* **Hierarchical View:** Parent (CBS) articles are pinned at the top of child articles.
* **Citation Lists:** Parent articles list all currently citing or matched children.
* **Direct Access:** Directly clickable source links for immediate external verification.

### Clear Metrics and Task Overview
* **Centralized Dashboard:** Real-time counters for processed, pending, and flagged items.
* **To-Do List Structure:** Organized categories based on matching priority and status.

### Manual Validation for Low-Confidence Matches
* **Focused Review:** Dedicated overview for matches with confidence scores **below 85%**.
* **Structured Interface:** Built specifically for careful, manual review of edge cases.

### Transparent Matching Logic
* **Keyword Highlighting:** Clear visual indicators of terms used in the automated matching process.
* **Evidence-Based Support:** Decision support backed by visible data points for user accountability.

### Recommended Parents for Matching
* **Top 5 Recommendations:** Ranked by confidence score to aid the user in manual selection.
* **Transparency:** Display of matching terms for every recommendation to explain the "why."

### Cluster Matching
* **Semantic Grouping:** Ability to match multiple similar articles to the same parent simultaneously based on semantic similarity.

### Additional Features
* **Toast Notifications:** Immediate visual feedback on all completed actions (e.g., "Match Verified").
* **Orphan Article Handling:** A separate overview for articles that could not be matched automatically.
* **Dark Mode:** Optional theme for improved visual comfort during extended analysis sessions.

---

### Summary
The final dashboard iteration reflects both the functional requirements and the insights gathered during the user evaluation session. Key improvements focus on **workflow efficiency**, **transparency of automated matching**, and the **reduction of repetitive manual actions**. The resulting prototype provides a clear and task-oriented interface tailored specifically to the CBS matching process.
