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
    text_clean = text.lower()
    scores = {}
    for term, row in df_tax.iterrows():
        term_str = str(term).lower().strip()
        topic = str(row['TT']).strip()
        if len(term_str) > 3 and term_str in text_clean:
            scores[topic] = scores.get(topic, 0) + 1
    
    valid_hits = {k: v for k, v in scores.items() if k not in ['999', '999.0', 'None', 'nan']}
    return max(valid_hits, key=valid_hits.get) if valid_hits else "999"

def get_news_body(zip_ref, child_id):
    for f in zip_ref.namelist():
        if f.endswith(f"c_{child_id}.csv"):
            with zip_ref.open(f) as file:
                df = pd.read_csv(file)
                return " ".join(df.iloc[:, 1:].fillna("").astype(str).values.flatten()).strip()
    return ""

def generate_example_csv():
    if not os.path.exists(HYBRID_PATH):
        print("Error"); return
    
    df = pd.read_csv(HYBRID_PATH).fillna(0)
    df_tax = pd.read_csv(TAX_PATH, index_col=0)
    
    # 80/20 split
    unique_ids = sorted(df['child_id'].unique())
    split_idx = int(len(unique_ids) * 0.8)
    train_ids, test_ids = unique_ids[:split_idx], unique_ids[split_idx:]
    
    train_df = df[df['child_id'].isin(train_ids)].copy()
    test_df = df[df['child_id'].isin(test_ids)].copy()

    features = [c for c in train_df.columns if c not in ['match', 'child_id', 'parent_id', 'probs', 'preds']]
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train_df[features])
    X_test = scaler.transform(test_df[features])
    
    model = CatBoostClassifier(iterations=100, scale_pos_weight=7.0, verbose=0, random_state=42)
    model.fit(X_train, train_df['match'])
    
    test_df['probs'] = model.predict_proba(X_test)[:, 1]
    test_df['preds'] = (test_df['probs'] > 0.5).astype(int)

    examples = []
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        p_file = next((f for f in z.namelist() if f.endswith('all_parents.csv')), None)
        with z.open(p_file) as f:
            p_map = pd.read_csv(f).set_index('id')['content'].to_dict()

        # CORRECT matches 
        correct_matches = test_df[(test_df['match'] == 1) & (test_df['preds'] == 1)].head(10)
        for _, row in correct_matches.iterrows():
            txt = get_news_body(z, int(row['child_id']))
            topic = resolve_cbs_theme(txt, df_tax)
            examples.append({"assigned": topic, "actual": topic, "article": txt})

        # INCORRECT matches
        incorrect_matches = test_df[(test_df['match'] == 0) & (test_df['preds'] == 1)].sort_values('probs', ascending=False)
        count = 0
        for _, row in incorrect_matches.iterrows():
            txt = get_news_body(z, int(row['child_id']))
            actual_t = resolve_cbs_theme(txt, df_tax)
            assigned_t = resolve_cbs_theme(p_map.get(row['parent_id'], ""), df_tax)
            
            if actual_t != assigned_t and count < 10:
                examples.append({"assigned": assigned_t, "actual": actual_t, "article": txt})
                count += 1

    out_df = pd.DataFrame(examples)
    out_df.to_csv("classification_examples.csv", index=False)
    print("Succes: classification_examples.csv was generated.")

if __name__ == "__main__":
    generate_example_csv()
