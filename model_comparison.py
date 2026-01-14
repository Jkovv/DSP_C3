import pandas as pd
import numpy as np
import os
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.preprocessing import MinMaxScaler

# --- CONFIGURATION ---
DATASETS = {
    "1. Baseline (Old)": "final_trainset.csv",
    "2. Enhanced (Full)": "actual_final_trainset_100pct_enhanced.csv",
    "3. Hybrid (S-BERT)": "final_hybrid_sbert_trainset.csv" 
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
            # Sort by score descending to find the top results
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
        
        print(f"\nProcessing {d_name}...")
        df = pd.read_csv(d_path).fillna(0)
        
        # --- IMPROVED PSEUDO-ID LOGIC FOR BASELINE ---
        id_col = next((c for c in ['child_id', 'id_child', 'id', 'c'] if c in df.columns), None)
        
        if not id_col:
            # Dynamic block size: Total Rows / Total Matches
            num_matches = df['match'].sum()
            if num_matches > 0:
                avg_block_size = int(len(df) / num_matches)
                print(f"  -> Detected {num_matches} matches. Using dynamic block size: {avg_block_size}")
                df['temp_id'] = np.arange(len(df)) // avg_block_size
            else:
                print("  -> Warning: No matches found in baseline. Using fallback block size 50.")
                df['temp_id'] = np.arange(len(df)) // 50
            id_col = 'temp_id'

        # Legacy score logic (Original jac_total vs. Simulated Sum)
        if 'jac_total' in df.columns:
            df['legacy_score'] = df['jac_total']
        else:
            sim_cols = [c for c in ['sbert_sim', 'title_similarity', 'vector_similarity'] if c in df.columns]
            count_cols = [c for c in ['num_matches', 'tax_matches', 'numbers_lenmatches', 'taxonomy_lenmatches'] if c in df.columns]
            
            score_parts = df[sim_cols].sum(axis=1)
            if count_cols:
                scaled_counts = scaler.fit_transform(df[count_cols].sum(axis=1).values.reshape(-1, 1))
                score_parts += scaled_counts.flatten()
            df['legacy_score'] = score_parts

        # 70/30 Split (Randomized but Group-aware to avoid leakage)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        train_idx, test_idx = next(gss.split(df, groups=df[id_col]))
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx].copy()
        
        exclude = ['match', 'child_id', 'parent_id', 'id', 'id_child', 'c', 'p', 'legacy_score', 'temp_id', 'jac_total']
        features = [c for c in train_df.columns if c not in exclude]
        
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

            # Statistical Metrics
            tr_acc = accuracy_score(train_df['match'], model.predict(train_df[features]))
            te_acc = accuracy_score(test_df['match'], preds)
            rec = recall_score(test_df['match'], preds, zero_division=0)
            prec = precision_score(test_df['match'], preds, zero_division=0)
            
            # Success Metrics
            s1 = calculate_ranking_success(test_df, 'current_probs', k=1, id_col=id_col)
            s5 = calculate_ranking_success(test_df, 'current_probs', k=5, id_col=id_col)
            cbs5 = calculate_ranking_success(test_df, 'legacy_score', k=5, id_col=id_col)

            unified_results.append({
                "Dataset": d_name, "Model": m_name,
                "Tr_Acc": f"{tr_acc:.4f}", "Te_Acc": f"{te_acc:.4f}",
                "Gap": f"{tr_acc - te_acc:.4f}", "Recall": f"{rec:.4f}", "Prec": f"{prec:.4f}",
                "Succ@1": f"{s1:.4f}", "Succ@5": f"{s5:.4f}", "CBS@5": f"{cbs5:.4f}"
            })

    print("\n" + "="*125)
    print("FINAL INTEGRATED PERFORMANCE REPORT (DYNAMIC GROUPING)")
    print("="*125)
    print(pd.DataFrame(unified_results).to_string(index=False))

if __name__ == "__main__":
    run_performance_audit()
