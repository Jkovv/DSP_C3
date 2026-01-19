import pandas as pd
import numpy as np
import os
import zipfile
import re
from sklearn.model_selection import GroupShuffleSplit
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

def run_cbs_audit():
    df = pd.read_csv(HYBRID_PATH).fillna(0)
    df_tax = pd.read_csv(TAX_PATH, index_col=0) # Index = Term
    id_col = 'child_id'
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df[id_col]))
    train_df, test_df = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
    features = [c for c in train_df.columns if c not in ['match', id_col, 'parent_id', 'probs', 'preds']]
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train_df[features])
    X_test = scaler.transform(test_df[features])
    model = CatBoostClassifier(iterations=100, scale_pos_weight=7.0, verbose=0, random_state=42)
    model.fit(X_train, train_df['match'])
    test_df['probs'] = model.predict_proba(X_test)[:, 1]

    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        p_file = next((f for f in z.namelist() if f.endswith('all_parents.csv')), None)
        with z.open(p_file) as f:
            p_map = pd.read_csv(f).set_index('id')['content'].to_dict()
        print(f"\n{'STATUS':<12} | {'ASSIGNED TOPIC (ALGO)':<25} | {'ACTUAL TOPIC (TRUTH)':<25} | {'SENTENCE'}")

        # CORRECT
        tp = test_df[test_df['match'] == 1].sort_values('probs', ascending=False).iloc[0]
        tp_txt = get_news_body(z, int(tp[id_col]))
        tp_topic = resolve_cbs_theme(tp_txt, df_tax)
        print(f"{'CORRECT':<12} | {tp_topic:<25} | {tp_topic:<25} | {tp_txt[:85]}...")

        # INCORRECT  
        fp_candidates = test_df[test_df['match'] == 0].sort_values('probs', ascending=False)
        for _, row in fp_candidates.iterrows():
            news_txt = get_news_body(z, int(row[id_col]))
            actual_t = resolve_cbs_theme(news_txt, df_tax)
            p_id = row.get('parent_id')
            assigned_t = resolve_cbs_theme(p_map.get(p_id, ""), df_tax) if p_id in p_map else "999"
            if assigned_t != actual_t:
                print(f"{'INCORRECT':<12} | {assigned_t:<25} | {actual_t:<25} | {news_txt[:85]}...")
                break

if __name__ == "__main__":
    run_cbs_audit()
