import pandas as pd
import numpy as np
import os
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score

DATASETS = {
    "1. Baseline (Old)": "final_trainset.csv",
    "2. Enhanced (Full)": "actual_final_trainset_100pct_enhanced.csv",
    "3. Hybrid (S-BERT)": "final_hybrid_sbert_trainset_100pct.csv" 
}

def cbs_date_comparison(diff, offset=2, scale=7):  
    """Original CBS Time Decay function."""
    return 2**(-(abs(diff) - offset) / scale)

def calculate_cbs_search_success(test_df, k=5):
    """
    Search Success: Groups by news item (Child) and checks 
    if the correct report is in the top 5 suggestions.
    """
    id_col = next((c for c in ['child_id', 'id_child', 'id', 'c'] if c in test_df.columns), None)
    if not id_col: return 0.0

    grouped = test_df.groupby(id_col)
    total_relevant = 0
    hits = 0
    
    for _, group in grouped:
        if group['match'].sum() > 0: 
            total_relevant += 1
            top_k = group.sort_values('probs', ascending=False).head(k)
            if top_k['match'].sum() > 0:
                hits += 1
                
    return hits / total_relevant if total_relevant > 0 else 0

def run_final_audit():
    final_results = []

    for d_name, d_path in DATASETS.items():
        if not os.path.exists(d_path): continue
        df = pd.read_csv(d_path).fillna(0)
        
        if 'date_diff_days' in df.columns:
            df['cbs_date_score'] = df['date_diff_days'].apply(cbs_date_comparison)
        
        y = df['match']
        X = df.drop(columns=['match'], errors='ignore')
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Calculate dynamic imbalance ratio
        dynamic_ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1]) if len(y_train[y_train == 1]) > 0 else 1
        print(f"\nAnalyzing: {d_name} (Imbalance {dynamic_ratio:.2f}:1)")

        drop_cols = ['child_id', 'parent_id', 'id', 'id_child', 'id_parent', 'c', 'p']
        features = [c for c in X_train.columns if c not in drop_cols]

        models = {
            "Old RF (Baseline)": RandomForestClassifier(n_estimators=100, random_state=42),
            "New RF (Balanced)": RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42),
            "XGBoost (Balanced)": xgb.XGBClassifier(n_estimators=300, scale_pos_weight=dynamic_ratio, random_state=42),
            "LightGBM (Balanced)": lgb.LGBMClassifier(n_estimators=300, scale_pos_weight=dynamic_ratio, random_state=42, verbose=-1),
            "CatBoost (Balanced)": CatBoostClassifier(iterations=300, scale_pos_weight=dynamic_ratio, verbose=0, random_state=42)
        }

        for m_name, model in models.items():
            model.fit(X_train[features], y_train)
            
            # Predictions
            train_preds = model.predict(X_train[features])
            test_preds = model.predict(X_test[features])
            test_probs = model.predict_proba(X_test[features])[:, 1]
            
            # Dataframe for Search Success metric
            eval_df = X_test.copy()
            eval_df['match'] = y_test.values
            eval_df['probs'] = test_probs
            
            # Calculating Metrics
            train_acc = accuracy_score(y_train, train_preds)
            test_acc = accuracy_score(y_test, test_preds)
            search_success = calculate_cbs_search_success(eval_df, k=5)
            recall = recall_score(y_test, test_preds, zero_division=0)
            gap = train_acc - test_acc

            final_results.append({
                "Dataset": d_name, "Model": m_name, 
                "Train_Acc": f"{train_acc:.4f}",
                "Test_Acc": f"{test_acc:.4f}", 
                "Search_SS@5": f"{search_success:.4f}", 
                "Recall(C1)": f"{recall:.4f}", 
                "Gap": f"{gap:.4f}"
            })

    # Display results
    report_df = pd.DataFrame(final_results)
    print("\nFINAL PERFORMANCE MATRIX")
    print(report_df.to_string(index=False))

if __name__ == "__main__":
    run_final_audit()
