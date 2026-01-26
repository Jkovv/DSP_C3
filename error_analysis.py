import pandas as pd
import numpy as np
import os
import zipfile
import re
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostClassifier

HYBRID_PATH = 'final_hybrid_sbert_trainset_100pct.csv'
TAX_PATH = 'taxonomie_df.csv'
ZIP_PATH = 'data.zip'

def resolve_cbs_theme(text, df_tax):
    if not isinstance(text, str) or len(text) < 10: return "999"
    # normalize casing so term matching is case-insensitive
    text_clean = text.lower()

    # counts taxonomy term hits and returns the most frequent topic code
    scores = {}
    for term, row in df_tax.iterrows():
        term_str = str(term).lower().strip()
        topic = str(row['TT']).strip()
        # ignore extremely short terms to reduce noise
        if len(term_str) > 3 and term_str in text_clean:
            scores[topic] = scores.get(topic, 0) + 1
    
    valid_hits = {k: v for k, v in scores.items() if k not in ['999', '999.0', 'None', 'nan']}
    return max(valid_hits, key=valid_hits.get) if valid_hits else "999"

def generate_example_csv():
    if not os.path.exists(HYBRID_PATH):
        print("Error: Missing Hybrid File"); return
    
    df = pd.read_csv(HYBRID_PATH).fillna(0)
    df_tax = pd.read_csv(TAX_PATH, index_col=0)

    # split on unique child_id so all candidates for one article stay in the same split
    unique_ids = sorted(df['child_id'].unique())
    split_idx = int(len(unique_ids) * 0.8)
    train_ids, test_ids = unique_ids[:split_idx], unique_ids[split_idx:]
    
    train_df = df[df['child_id'].isin(train_ids)].copy()
    test_df = df[df['child_id'].isin(test_ids)].copy()

    features = [c for c in train_df.columns if c not in ['match', 'child_id', 'parent_id', 'probs', 'preds']]
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train_df[features])
    X_test = scaler.transform(test_df[features])
    
    model = CatBoostClassifier(iterations=200, scale_pos_weight=24.0, verbose=0, random_state=42)
    model.fit(X_train, train_df['match'])
    
    test_df['probs'] = model.predict_proba(X_test)[:, 1]
    # for now
    test_df['preds'] = (test_df['probs'] > 0.5).astype(int)

    examples = []
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        zip_files = {f.split('/')[-1]: f for f in z.namelist()}
        
        p_file = next((f for f in z.namelist() if f.endswith('all_parents.csv')), None)
        with z.open(p_file) as f:
            p_map = pd.read_csv(f).set_index('id')['content'].to_dict()

        # true positives: top 10 highest-confidence correct predictions to sanity-check
        correct_matches = test_df[(test_df['match'] == 1) & (test_df['preds'] == 1)].sort_values('probs', ascending=False).head(10)
        for _, row in correct_matches.iterrows():
            fname = f"c_{int(row['child_id'])}.csv"
            if fname in zip_files:
                with z.open(zip_files[fname]) as file:
                    txt = " ".join(pd.read_csv(file).iloc[:, 1:].fillna("").astype(str).values.flatten()).strip()
                topic = resolve_cbs_theme(txt, df_tax)
                examples.append({"type": "CORRECT", "assigned": topic, "actual": topic, "article": txt[:500] + "..."})

        # false positives
        incorrect_matches = test_df[(test_df['match'] == 0) & (test_df['preds'] == 1)].sort_values('probs', ascending=False)
        
        added = 0
        for _, row in incorrect_matches.iterrows():
            if added >= 10: break
            
            fname = f"c_{int(row['child_id'])}.csv"
            if fname in zip_files:
                with z.open(zip_files[fname]) as file:
                    txt = " ".join(pd.read_csv(file).iloc[:, 1:].fillna("").astype(str).values.flatten()).strip()
                
                actual_t = resolve_cbs_theme(txt, df_tax)
                assigned_t = resolve_cbs_theme(p_map.get(row['parent_id'], ""), df_tax)
                
                if actual_t != assigned_t:
                    examples.append({"type": "ERROR", "assigned": assigned_t, "actual": actual_t, "article": txt[:500] + "..."})
                    added += 1

    out_df = pd.DataFrame(examples)
    out_df.to_csv("classification_examples.csv", index=False)
    print("Success: classification_examples.csv generated.")

if __name__ == "__main__":
    generate_example_csv()
