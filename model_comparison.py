import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (accuracy_score, recall_score, precision_score, 
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)
from sklearn.preprocessing import MinMaxScaler

def try_import(module_name):
    try: return __import__(module_name)
    except ImportError: return None

xgb = try_import("xgboost")
lgb = try_import("lightgbm") 
cb = try_import("catboost")
shap = try_import("shap")

DATASETS = {
    "1. Baseline (Old)": "final_trainset.csv",
    "2. Fixed (Full)": "final_basic_trainset_fixed.csv",
    "3. Hybrid (S-BERT)": "final_hybrid_sbert_trainset_100pct.csv" 
}

def calculate_success_k(test_df, score_col, k_values=[1, 2, 3, 4, 5], id_col='child_id'):
    grouped = test_df.groupby(id_col)
    total_queries, hits = 0, {k: 0 for k in k_values}
    for _, group in grouped:
        if group['match'].sum() > 0: 
            total_queries += 1
            sorted_group = group.sort_values(score_col, ascending=False)
            for k in k_values:
                if sorted_group.head(k)['match'].sum() > 0: hits[k] += 1
    return {f"Succ@{k}": hits[k] / total_queries if total_queries > 0 else 0 for k in k_values}

def run_audit():
    final_results = []
    scaler = MinMaxScaler()

    for d_name, d_path in DATASETS.items():
        if not os.path.exists(d_path): continue
            
        print(f"\nPROCESSING: {d_name}")
        df = pd.read_csv(d_path).fillna(0)
        
        # Permanent ID generation
        id_col = next((c for c in ['child_id', 'id_child', 'id'] if c in df.columns), None)
        if not id_col:
            df['child_id'] = np.arange(len(df)) // 25 
            id_col = 'child_id'

        # 80/20 Group-Aware Split
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        train_idx, test_idx = next(gss.split(df, groups=df[id_col]))
        train_df, test_df = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
        
        exclude = ['match', id_col, 'parent_id', 'legacy_score', 'c', 'p']
        features = [c for c in train_df.columns if c not in exclude]
        
        X_train = scaler.fit_transform(train_df[features])
        X_test = scaler.transform(test_df[features])
        y_train, y_test = train_df['match'].values, test_df['match'].values
        ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1]) if 1 in y_train else 1

        models = {"Balanced_RF": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)}
        
        if xgb: 
            models["XGBoost"] = xgb.XGBClassifier(scale_pos_weight=ratio, random_state=42, eval_metric='logloss')
        
        if lgb: 
            models["LightGBM"] = lgb.LGBMClassifier(scale_pos_weight=ratio, random_state=42, verbose=-1)
        
        if cb:  
            models["CatBoost"] = cb.CatBoostClassifier(iterations=100, scale_pos_weight=ratio, verbose=0, random_state=42)

        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))

        for m_name, model in models.items():
            print(f"   Training {m_name}...")
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]
            preds = (probs > 0.5).astype(int)
            test_df['probs'] = probs
            
            # metrics 
            auc = roc_auc_score(y_test, probs)
            rec = recall_score(y_test, preds, zero_division=0)
            f1 = f1_score(y_test, preds, zero_division=0)
            succ = calculate_success_k(test_df, 'probs', id_col=id_col)
            
            res = {"Dataset": d_name, "Model": m_name, "Recall": f"{rec:.4f}", "F1": f"{f1:.4f}", "AUC": f"{auc:.4f}", 
                   "Gap": f"{accuracy_score(y_train, model.predict(X_train)) - accuracy_score(y_test, preds):.4f}"}
            res.update({k: f"{v:.4f}" for k, v in succ.items()})
            final_results.append(res)

            # confusion matrix
            plt.figure(figsize=(4, 3))
            sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.title(f"CM_{m_name}_{d_name[0]}")
            plt.savefig(f"CM_{d_name[0]}_{m_name}.png"); plt.close()

            # SHAP analysis for misclassifications
            if shap and m_name in ["Balanced_RF", "CatBoost", "LightGBM"]:
                try:
                    explainer = shap.Explainer(model)
                    s_vals = explainer(X_test).values
                    if len(s_vals.shape) == 3: s_vals = s_vals[:,:,1]
                    
                    fn_idx = np.where((y_test == 1) & (preds == 0))[0]
                    if len(fn_idx) > 0:
                        plt.figure()
                        shap.summary_plot(s_vals[fn_idx], pd.DataFrame(X_test[fn_idx], columns=features), show=False)
                        plt.title(f"Why we MISSED: {m_name}")
                        plt.savefig(f"SHAP_Missed_{d_name[0]}_{m_name}.png", bbox_inches='tight'); plt.close()
                except: pass

            fpr, tpr, _ = roc_curve(y_test, probs)
            ax_roc.plot(fpr, tpr, label=f"{m_name} (AUC={auc:.3f})")

        ax_roc.plot([0, 1], [0, 1], 'k--'); ax_roc.legend(); ax_roc.set_title(f"ROC: {d_name}")
        fig_roc.savefig(f"ROC_{d_name[0]}.png"); plt.close(fig_roc)

    print("\nFINAL AUDIT MATRIX")
    order = ["Dataset", "Model", "Recall", "F1", "AUC", "Gap", "Succ@1", "Succ@2", "Succ@3", "Succ@4", "Succ@5"]
    df_final = pd.DataFrame(final_results)
    print(df_final[order].to_string(index=False))

if __name__ == "__main__":
    run_audit()

