import pandas as pd
import numpy as np
import os
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import MinMaxScaler

DATASETS = {
    "1. Baseline (Old)": "final_trainset.csv",
    "2. Fixed (Full)": "final_basic_trainset_fixed.csv",
    "3. Hybrid (S-BERT)": "final_hybrid_sbert_trainset_100pct.csv" 
}

def cbs_date_comparison(diff, offset=2, scale=7):  
    """Original CBS Time Decay function."""
    return 2**(-(abs(diff) - offset) / scale)

def calculate_ranking_success(test_df, score_col, k=5, id_col='child_id'):
    """ Success@K: Is the correct match in the Top K ranked results?"""
    if id_col not in test_df.columns: return 0.0
    
    grouped = test_df.groupby(id_col)
    total_relevant_queries = 0
    hits = 0
    
    for _, group in grouped:
        if group['match'].sum() > 0: 
            total_relevant_queries += 1
            top_k = group.sort_values(score_col, ascending=False).head(k)
            if top_k['match'].sum() > 0:
                hits += 1
                
    return hits / total_relevant_queries if total_relevant_queries > 0 else 0

def run_final_audit():
    final_results = []
    scaler = MinMaxScaler()

    for d_name, d_path in DATASETS.items():
        if not os.path.exists(d_path):
            print(f"Skipping {d_name}: File not found.")
            continue
            
        df = pd.read_csv(d_path).fillna(0)
        
        id_col = next((c for c in ['child_id', 'id_child', 'id', 'c'] if c in df.columns), None)
        if not id_col:
            num_matches = df['match'].sum()
            avg_block = int(len(df) / num_matches) if num_matches > 0 else 50
            df['child_id'] = np.arange(len(df)) // avg_block
            id_col = 'child_id'

        # legacy score (CBS_Legacy@5)
        if 'jac_total' in df.columns:
            df['legacy_score'] = df['jac_total']
        else:
            # simulating the old CBS scoring if not present
            sim_cols = [c for c in ['sbert_sim', 'title_similarity', 'tax_matches', 'num_matches'] if c in df.columns]
            df['legacy_score'] = df[sim_cols].sum(axis=1)

        # group aware split
        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        train_idx, test_idx = next(gss.split(df, groups=df[id_col]))
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx].copy()
        
        # features
        exclude = ['match', 'child_id', 'parent_id', 'id', 'id_child', 'legacy_score', 'c', 'p']
        features = [c for c in train_df.columns if c not in exclude]
        
        y_train = train_df['match']
        y_test = test_df['match']
        
        dynamic_ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1]) if len(y_train[y_train == 1]) > 0 else 1
        print(f"Analyzing: {d_name} (Features: {len(features)})")

        models = {
            "Old RF (Baseline)": RandomForestClassifier(n_estimators=100, random_state=42),
            "New RF (Balanced)": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
            "XGBoost (Balanced)": xgb.XGBClassifier(n_estimators=100, scale_pos_weight=dynamic_ratio, random_state=42, eval_metric='logloss'),
            "LightGBM (Balanced)": lgb.LGBMClassifier(n_estimators=100, scale_pos_weight=dynamic_ratio, random_state=42, verbose=-1),
            "CatBoost (Balanced)": CatBoostClassifier(iterations=100, scale_pos_weight=dynamic_ratio, verbose=0, random_state=42)
        }

        for m_name, model in models.items():
            model.fit(train_df[features], y_train)
            
            # preds
            test_probs = model.predict_proba(test_df[features])[:, 1]
            test_preds = (test_probs > 0.5).astype(int)
            train_preds = model.predict(train_df[features])
            
            test_df['probs'] = test_probs
            
            # metrics 
            tr_acc = accuracy_score(y_train, train_preds)
            te_acc = accuracy_score(y_test, test_preds)
            rec = recall_score(y_test, test_preds, zero_division=0)
            prec = precision_score(y_test, test_preds, zero_division=0)
            f1 = f1_score(y_test, test_preds, zero_division=0)
            
            s1 = calculate_ranking_success(test_df, 'probs', k=1, id_col=id_col)
            s5 = calculate_ranking_success(test_df, 'probs', k=5, id_col=id_col)
            leg5 = calculate_ranking_success(test_df, 'legacy_score', k=5, id_col=id_col)

            final_results.append({
                "Dataset": d_name,
                "Model": m_name,
                "Train_Acc": f"{tr_acc:.4f}",
                "Test_Acc": f"{te_acc:.4f}",
                "Gap": f"{tr_acc - te_acc:.4f}",
                "Recall": f"{rec:.4f}",
                "Precision": f"{prec:.4f}",
                "F1-Score": f"{f1:.4f}",
                "Succ@1": f"{s1:.4f}", 
                # "Succ@5": f"{s5:.4f}", 
                # "CBS_Legacy@5": f"{leg5:.4f}" 
            })

    # results
    report_df = pd.DataFrame(final_results)
    print("\nFINAL PERFORMANCE MATRIX")
    print(report_df.to_string(index=False))

if __name__ == "__main__":
    run_final_audit()
