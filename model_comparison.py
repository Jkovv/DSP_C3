import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (accuracy_score, recall_score, f1_score, 
                             roc_auc_score, roc_curve, confusion_matrix)
from sklearn.preprocessing import MinMaxScaler

def try_import(module_name):
    try: return __import__(module_name)
    except ImportError: return None

xgb = try_import("xgboost")
lgb = try_import("lightgbm") 
cb = try_import("catboost")
shap = try_import("shap")

DATASETS = {
    "1. Baseline": "final_trainset.csv",
    "2. Basic": "final_basic_trainset_fixed.csv",
    "3. Hybrid": "final_hybrid_sbert_trainset_100pct.csv" 
}

def calculate_success_k(test_df, score_col, k_values=[1, 2, 3, 4, 5], id_col='child_id'):
    temp_df = test_df.copy()
    temp_df[score_col] = temp_df[score_col] + np.random.uniform(0, 1e-10, size=len(temp_df))
    
    grouped = temp_df.groupby(id_col)
    total_queries, hits = 0, {k: 0 for k in k_values}
    
    for _, group in grouped:
        if group['match'].sum() > 0: 
            total_queries += 1
            sorted_group = group.sort_values(score_col, ascending=False)
            for k in k_values:
                if sorted_group.head(k)['match'].sum() > 0: 
                    hits[k] += 1
    return {f"Succ@{k}": hits[k] / total_queries if total_queries > 0 else 0 for k in k_values}

def run_master_evaluation():
    all_metrics = []
    scaler = MinMaxScaler()

    for d_name, d_path in DATASETS.items():
        if not os.path.exists(d_path): continue
        print(f"\nAUDITING DATASET: {d_name}")
        df = pd.read_csv(d_path).fillna(0)
        
        id_col = next((c for c in ['child_id', 'id_child', 'id'] if c in df.columns), None)
        if not id_col:
            df['child_id'] = np.arange(len(df)) // 25 
            id_col = 'child_id'

        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        train_idx, test_idx = next(gss.split(df, groups=df[id_col]))
        train_df, test_df = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
        
        exclude = ['match', id_col, 'parent_id', 'legacy_score']
        features = [c for c in train_df.columns if c in train_df.columns and c not in exclude]
        
        X_train = scaler.fit_transform(train_df[features])
        X_test = scaler.transform(test_df[features])
        y_train, y_test = train_df['match'].values, test_df['match'].values
        
        ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1]) if 1 in y_train else 10

        models = {
            "Balanced_RF": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
            "CatBoost": cb.CatBoostClassifier(iterations=200, scale_pos_weight=ratio, verbose=0, random_state=42) if cb else None
        }
        if xgb: models["XGBoost"] = xgb.XGBClassifier(scale_pos_weight=ratio, random_state=42, eval_metric='logloss')
        if lgb: models["LightGBM"] = lgb.LGBMClassifier(scale_pos_weight=ratio, random_state=42, verbose=-1)

        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))

        for m_name, model in models.items():
            if model is None: continue
            print(f"   Training {m_name}...")
            model.fit(X_train, y_train)
            
            # pred
            tr_probs = model.predict_proba(X_train)[:, 1]
            ts_probs = model.predict_proba(X_test)[:, 1]
            
            # top-4% quantile threshold -> arbitrary choice for positive class
            threshold = np.quantile(ts_probs, 0.96)
            ts_preds = (ts_probs >= threshold).astype(int)
            
            test_df['probs'] = ts_probs
            
            # metrics
            tr_acc = accuracy_score(y_train, (tr_probs > 0.5).astype(int))
            ts_acc = accuracy_score(y_test, ts_preds)
            gap = tr_acc - ts_acc
            auc = roc_auc_score(y_test, ts_probs) if len(np.unique(y_test)) > 1 else 0
            rec = recall_score(y_test, ts_preds, zero_division=0)
            # f1 = f1_score(y_test, ts_preds, zero_division=0)
            succ = calculate_success_k(test_df, 'probs', id_col=id_col)
            
            res = {
                "Dataset": d_name, "Model": m_name, 
                "Train_Acc": f"{tr_acc:.4f}", "Test_Acc": f"{ts_acc:.4f}", "Gap": f"{gap:.4f}",
                "Recall": f"{rec:.4f}", "AUC": f"{auc:.4f}"
            }
            res.update({k: f"{v:.4f}" for k, v in succ.items()})
            all_metrics.append(res)

            # Confusion matrix
            plt.figure(figsize=(4, 3))
            sns.heatmap(confusion_matrix(y_test, ts_preds), annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.title(f"CM_{d_name[0]}_{m_name}")
            plt.savefig(f"CM_{d_name[0]}_{m_name}.png", bbox_inches='tight'); plt.close()

            # ROC Data
            fpr, tpr, _ = roc_curve(y_test, ts_probs)
            ax_roc.plot(fpr, tpr, label=f"{m_name} (AUC={auc:.3f})")

            # SHAP Summary
            if shap and m_name in ["Balanced_RF", "CatBoost", "LightGBM"]:
                try:
                    explainer = shap.Explainer(model)
                    sv = explainer(X_test[:100]).values
                    if len(sv.shape) == 3: sv = sv[:,:,1]
                    plt.figure(); shap.summary_plot(sv, pd.DataFrame(X_test[:100], columns=features), show=False)
                    plt.savefig(f"SHAP_{d_name[0]}_{m_name}.png", bbox_inches='tight'); plt.close()
                except: pass
        
        ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5); ax_roc.legend()
        ax_roc.set_title(f"ROC Curves - {d_name}")
        fig_roc.savefig(f"ROC_{d_name[0]}.png", bbox_inches='tight'); plt.close(fig_roc)

    order = ["Dataset", "Model", "Train_Acc", "Test_Acc", "Gap", "Recall", "AUC", 
             "Succ@1", "Succ@2", "Succ@3", "Succ@4", "Succ@5"]
    print(pd.DataFrame(all_metrics)[order].to_string(index=False))

if __name__ == "__main__":
    run_master_evaluation()
