
import argparse, os, yaml, joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

def time_split(df, train_ratio=0.7, val_ratio=0.15):
    n = len(df)
    n_train = int(n*train_ratio)
    n_val = int(n*(train_ratio+val_ratio))
    return df.iloc[:n_train], df.iloc[n_train:n_val], df.iloc[n_val:]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open('config.yaml'))
    feat_dir = cfg['feature_dir']
    model_dir = cfg['model_dir']
    os.makedirs(model_dir, exist_ok=True)

    csv_path = os.path.join(feat_dir, f"{args.symbol}_features.csv")
    df = pd.read_csv(csv_path, parse_dates=['time'])

    feature_cols = [c for c in df.columns if c not in ('time','y')]
    X = df[feature_cols].values
    y = df['y'].values

    # time-aware split
    train, val, test = time_split(df)
    X_train = train[feature_cols].values; y_train = train['y'].values
    X_val   = val[feature_cols].values;   y_val   = val['y'].values
    X_test  = test[feature_cols].values;  y_test  = test['y'].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight='balanced', random_state=42)
    rf.fit(X_train_s, y_train)

    def eval_split(name, Xs, ys):
        proba = rf.predict_proba(Xs)[:,1]
        pred  = (proba >= 0.5).astype(int)
        auc = roc_auc_score(ys, proba)
        acc = accuracy_score(ys, pred)
        return auc, acc

    for name, Xs, ys in [('train', X_train_s, y_train), ('val', X_val_s, y_val), ('test', X_test_s, y_test)]:
        auc, acc = eval_split(name, Xs, ys)
        print(f"{name}: AUC={auc:.3f} ACC={acc:.3f}")

    joblib.dump(rf, os.path.join(model_dir, f"{args.symbol}_rf.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    print(f"Saved model to {model_dir}")

if __name__ == '__main__':
    main()
