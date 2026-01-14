import pandas as pd
import numpy as np
import os
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.preprocessing import MinMaxScaler


DATASETS = {
    "1. Baseline (Old)": "final_trainset.csv",
    "2. Enhanced (Full)": "actual_final_trainset_100pct_enhanced.csv",
    "3. Hybrid (S-BERT)": "final_hybrid_sbert_trainset.csv" # i think this is 30% -> fix
}

def calculate_ranking_success(df_test, score_col, k=5, id_col='child_id'):
    """Calculates Success@K: Is the correct match in the Top K ranked results?"""
    if id_col not in df_test.columns:
        return np.nan
    
    grouped = df_test.groupby(id_col)
    hits = 0
    total = 0
    for _, group in grouped:
        if group['match'].sum() > 0:
            total += 1
            top_k = group.sort_values(score_col, ascending=False).head(k)
            if top_k['match'].sum() > 0:
                hits += 1
    return hits / total if total > 0 else 0

def run_performance_audit():
    unified_results = []
    scaler = MinMaxScaler()

    for d_name, d_path in DATASETS.items():
        if not os.path.exists(d_path):
            print(f"Skipping {d_name}: File not found at {d_path}")
            continue
        
        print(f"Processing {d_name}...")
        df = pd.read_csv(d_path).fillna(0)
        
        id_col = next((c for c in ['child_id', 'id_child', 'id', 'c'] if c in df.columns), None)
        if not id_col:
            # for the baseline, we assume blocks of 50 candidates per news item to enable ranking
            df['temp_id'] = np.arange(len(df)) // 50
            id_col = 'temp_id'

        # replicated legacy scoring heuristic (they called it jac_total but it's not jaccard)
        if 'jac_total' in df.columns:
            # baseline -> using the actual original jac_total column
            df['legacy_score'] = df['jac_total']
        else:
            # simulating legacy score 
            sim_cols = [c for c in ['sbert_sim', 'title_similarity', 'vector_similarity'] if c in df.columns]
            count_cols = [c for c in ['num_matches', 'tax_matches', 'numbers_lenmatches', 'taxonomy_lenmatches'] if c in df.columns]
            
            # similarities (0-1) + scaled counts (0-1)
            score_parts = df[sim_cols].sum(axis=1)
            if count_cols:
                scaled_counts = scaler.fit_transform(df[count_cols].sum(axis=1).values.reshape(-1, 1))
                score_parts += scaled_counts.flatten()
            df['legacy_score'] = score_parts

        # 70/30 data split 
        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        train_idx, test_idx = next(gss.split(df, groups=df[id_col]))
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx].copy()
        
        # features (exclude target, IDs, and the heuristic scores)
        exclude = ['match', 'child_id', 'parent_id', 'id', 'id_child', 'c', 'p', 'legacy_score', 'temp_id', 'jac_total']
        features = [c for c in train_df.columns if c not in exclude]
        
        # model training - handling class imbalance
        ratio = len(train_df[train_df['match']==0]) / len(train_df[train_df['match']==1])
        
        models = {
            "RF": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "XGBoost": xgb.XGBClassifier(scale_pos_weight=ratio, random_state=42, eval_metric='logloss'),
            "LightGBM": lgb.LGBMClassifier(scale_pos_weight=ratio, random_state=42, verbose=-1),
            "CatBoost": CatBoostClassifier(scale_pos_weight=ratio, verbose=0, random_state=42)
        }

        for m_name, model in models.items():
            model.fit(train_df[features], train_df['match'])
            
            probs = model.predict_proba(test_df[features])[:, 1]
            preds = (probs > 0.5).astype(int)
            test_df['current_probs'] = probs

            # metrics
            tr_acc = accuracy_score(train_df['match'], model.predict(train_df[features]))
            te_acc = accuracy_score(test_df['match'], preds)
            rec = recall_score(test_df['match'], preds, zero_division=0)
            prec = precision_score(test_df['match'], preds, zero_division=0)
            
            s1 = calculate_ranking_success(test_df, 'current_probs', k=1, id_col=id_col)
            s5 = calculate_ranking_success(test_df, 'current_probs', k=5, id_col=id_col)
            cbs5 = calculate_ranking_success(test_df, 'legacy_score', k=5, id_col=id_col)

            unified_results.append({
                "Dataset": d_name,
                "Model": m_name,
                "Tr_Acc": f"{tr_acc:.4f}",
                "Te_Acc": f"{te_acc:.4f}",
                "Gap": f"{tr_acc - te_acc:.4f}",
                "Recall": f"{rec:.4f}",
                "Prec": f"{prec:.4f}",
                "Succ@1": f"{s1:.4f}",
                "Succ@5": f"{s5:.4f}",
                "CBS@5": f"{cbs5:.4f}"
            })

    print("model perfonmance audit results:")
    final_df = pd.DataFrame(unified_results)
    print(final_df.to_string(index=False))

if __name__ == "__main__":
    run_performance_audit()
