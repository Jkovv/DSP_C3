import pandas as pd
import numpy as np
import os
import zipfile
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
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
    valid_hits = {k: v for k, v in scores.items() if k not in ['999', 'nan']}
    return max(valid_hits, key=valid_hits.get) if valid_hits else "999"

def generate_semantic_pairs():
    df = pd.read_csv(HYBRID_PATH).fillna(0)
    df_tax = pd.read_csv(TAX_PATH, index_col=0)
    
    # 80/20 split
    unique_ids = sorted(df['child_id'].unique())
    test_ids = unique_ids[int(len(unique_ids) * 0.8):]
    train_df = df[~df['child_id'].isin(test_ids)].copy()
    test_df = df[df['child_id'].isin(test_ids)].copy()

    features = [c for c in train_df.columns if c not in ['match', 'child_id', 'parent_id']]
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train_df[features])
    X_test = scaler.transform(test_df[features])
    
    model = CatBoostClassifier(iterations=200, scale_pos_weight=24.0, verbose=0, random_state=42)
    model.fit(X_train, train_df['match'])
    
    test_df['probs'] = model.predict_proba(X_test)[:, 1]
    
    seeds = test_df[(test_df['match'] == 1) & (test_df['probs'] > 0.8)].head(5)
    
    output_rows = []
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        zip_files = {f.split('/')[-1]: f for f in z.namelist()}
        
        for idx, seed_row in seeds.iterrows():
            seed_id = int(seed_row['child_id'])
            seed_txt = ""
            if f"c_{seed_id}.csv" in zip_files:
                with z.open(zip_files[f"c_{seed_id}.csv"]) as f:
                    seed_txt = " ".join(pd.read_csv(f).iloc[:, 1:].fillna("").astype(str).values.flatten()).strip()
            
            seed_topic = resolve_cbs_theme(seed_txt, df_tax)
            
            # cosine similarity with other test samples
            seed_vec = X_test[test_df.index.get_loc(idx)].reshape(1, -1)
            similarities = cosine_similarity(seed_vec, X_test).flatten()
            
            similarities[test_df.index.get_loc(idx)] = -1
            neighbor_idx_local = np.argmax(similarities)
            neighbor_row = test_df.iloc[neighbor_idx_local]
            neighbor_sim = similarities[neighbor_idx_local]
            
            neighbor_id = int(neighbor_row['child_id'])
            neighbor_txt = ""
            if f"c_{neighbor_id}.csv" in zip_files:
                with z.open(zip_files[f"c_{neighbor_id}.csv"]) as f:
                    neighbor_txt = " ".join(pd.read_csv(f).iloc[:, 1:].fillna("").astype(str).values.flatten()).strip()
            
            neighbor_topic = resolve_cbs_theme(neighbor_txt, df_tax)
            
            output_rows.append({
                "Pair_ID": seed_id, "Role": "SEED (Correct)", 
                "Similarity": 1.0, "Category": seed_topic, "Text": seed_txt[:500]
            })
            output_rows.append({
                "Pair_ID": seed_id, "Role": f"NEIGHBOR (Sim: {neighbor_sim:.3f})", 
                "Similarity": neighbor_sim, "Category": neighbor_topic, "Text": neighbor_txt[:500]
            })
            output_rows.append({"Pair_ID": "", "Role": "---", "Similarity": "", "Category": "", "Text": ""})

    pd.DataFrame(output_rows).to_csv("semantic_similarity_proof.csv", index=False)
    print("Success: semantic_similarity_proof.csv generated.")

if __name__ == "__main__":
    generate_semantic_pairs()
