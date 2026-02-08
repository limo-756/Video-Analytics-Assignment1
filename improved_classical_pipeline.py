import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import joblib
import time
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Try importing cuML, fall back to sklearn
try:
    from cuml.svm import SVC
    USE_CUML = True
    print("Using cuML for SVM")
except ImportError:
    from sklearn.svm import SVC
    USE_CUML = False
    print("Using sklearn for SVM")

from feature_extractor import VideoExtractorFeature, CONFIG

# Setup Directories
OUTPUT_DIR = "results_classical"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    train = pd.read_csv("./dataset/splits/train.csv", index_col='index')
    val = pd.read_csv("./dataset/splits/validation.csv", index_col='index')
    test = pd.read_csv("./dataset/splits/test.csv", index_col='index')
    # Stratified subset for debugging
    # return train.groupby('label').head(5), val.groupby('label').head(5), test.groupby('label').head(5)
    
    # Returning full data now but kept snippet for future debug
    return train, val, test

def train_svm_optuna(X_train, y_train, X_val, y_val, trials=20):
    def objective(trial):
        params = {
            'C': trial.suggest_float('C', 1e-2, 1e2, log=True),
            'gamma': trial.suggest_float('gamma', 1e-3, 1e1, log=True),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
        }
        
        clf = SVC(**params, probability=True) 
        clf.fit(X_train, y_train)
        
        preds = clf.predict(X_val)
        acc = accuracy_score(y_val, preds)
        return acc

    print("Optimizing SVM...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=trials)
    
    print("Best params (SVM):", study.best_params)
    return study.best_params

def train_rf_optuna(X_train, y_train, X_val, y_val, trials=20):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        }
        
        clf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        
        preds = clf.predict(X_val)
        acc = accuracy_score(y_val, preds)
        return acc

    print("Optimizing Random Forest...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=trials)
    
    print("Best params (RF):", study.best_params)
    return study.best_params

def train_knn_optuna(X_train, y_train, X_val, y_val, trials=20):
    def objective(trial):
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', 3, 20),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan']),
        }
        
        clf = KNeighborsClassifier(**params, n_jobs=-1)
        clf.fit(X_train, y_train)
        
        preds = clf.predict(X_val)
        acc = accuracy_score(y_val, preds)
        return acc

    print("Optimizing KNN...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=trials)
    
    print("Best params (KNN):", study.best_params)
    return study.best_params
def plot_confusion_matrix(y_true, y_pred, labels, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def plot_roc_curve(clf, X_test, y_test, label_encoder, filename):
    # Only works if probability=True or decision_function available
    # cuML SVC might not support predict_proba easily for all kernels?
    # sklearn SVC needs probability=True.
    
    try:
        y_score = clf.predict_proba(X_test)
    except:
        try:
             y_score = clf.decision_function(X_test)
        except:
            print("Model does not support probability/decision function. Skipping ROC.")
            return

    n_classes = len(label_encoder.classes_)
    
    # Binarize output
    from sklearn.preprocessing import label_binarize
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    
    plt.figure()
    colors = ['blue', 'red', 'green']
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'Class {label_encoder.classes_[i]} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def run_experiment(name, config_update):
    print(f"\n--- Running Experiment: {name} ---")
    
    # Reload Config
    cfg = CONFIG.copy()
    cfg.update(config_update)
    
    extractor = VideoExtractorFeature(cfg)
    
    train_df, val_df, test_df = load_data()
    
    # Process
    t0 = time.time()
    print("Extracting features...")
    X_train_frames = extractor.process_dataset(train_df)
    X_val_frames = extractor.process_dataset(val_df)
    
    # Scaling
    X_train_frames = extractor.fit_transform_scaler(X_train_frames)
    X_val_frames = extractor.fit_transform_scaler(X_val_frames, is_test=True)
    
    print(f"Extraction time: {time.time()-t0:.2f}s")
    
    # Merge Labels
    # We need to map back to labels.
    # process_dataset returns df with 'index'
    # train_df index is 'index'
    
    train_merged = pd.merge(train_df.reset_index(), X_train_frames, on='index')
    val_merged = pd.merge(val_df.reset_index(), X_val_frames, on='index')
    
    # Encode Labels
    le = LabelEncoder()
    # Fit on all possible labels to avoid unseen label errors
    all_labels = pd.concat([train_merged['label'], val_merged['label']]).unique()
    le.fit(all_labels)
    
    y_train_enc = le.transform(train_merged['label'])
    y_val_enc = le.transform(val_merged['label'])
    
    # Mode 1: Frame Level Classification + Voting
    print("Training Frame-Level Model...")
    feature_cols = [c for c in X_train_frames.columns if c not in ['index', 'encoded_label']]
    
    X_train_f = train_merged[feature_cols].values
    X_val_f = val_merged[feature_cols].values
    
    best_params = train_svm_optuna(X_train_f, y_train_enc, X_val_f, y_val_enc, trials=10)
    
    best_clf = SVC(**best_params, probability=True)
    best_clf.fit(X_train_f, y_train_enc)
    
    # Evaluate Video Level (Voting)
    val_preds_frames = best_clf.predict(X_val_f)
    val_merged['pred'] = val_preds_frames
    
    y_true_vid = []
    y_pred_vid = []
    
    for vid, group in val_merged.groupby('index'):
        y_true_vid.append(le.transform([group['label'].iloc[0]])[0])
        # Mode of frame predictions
        mode_pred = group['pred'].mode().iloc[0]
        y_pred_vid.append(mode_pred)
        
    acc = accuracy_score(y_true_vid, y_pred_vid)
    print(f"Video Level Accuracy (Voting): {acc:.4f}")
    
    plot_confusion_matrix(y_true_vid, y_pred_vid, le.classes_, f"Confusion Matrix - {name} (Voting)", f"cm_{name}_voting.png")
    
    # Mode 2: Video Level Aggregation
    print("Training Video-Level Aggregated Model...")
    
    # Aggregate
    # Note: re-process from raw X_train_frames (already scaled)
    # Aggregating SCALED features is fine.
    
    X_train_vid_df = extractor.aggregate_features(X_train_frames)
    X_val_vid_df = extractor.aggregate_features(X_val_frames)
    
    # We lost labels in aggregation, need to re-merge
    # X_train_vid_df index is 'index' (video id)
    
    train_vid_merged = pd.merge(train_df, X_train_vid_df, left_index=True, right_index=True)
    val_vid_merged = pd.merge(val_df, X_val_vid_df, left_index=True, right_index=True)
    
    feat_cols_vid = [c for c in train_vid_merged.columns if c not in train_df.columns]
    
    X_train_v = train_vid_merged[feat_cols_vid].values
    y_train_v = le.transform(train_vid_merged['label'])
    
    X_val_v = val_vid_merged[feat_cols_vid].values
    y_val_v = le.transform(val_vid_merged['label'])
    
    # PCA (Optional - req 4.4)
    pca = PCA(n_components=0.95) # Keep 95% variance
    X_train_v_pca = pca.fit_transform(X_train_v)
    X_val_v_pca = pca.transform(X_val_v)
    print(f"PCA reduced dim from {X_train_v.shape[1]} to {X_train_v_pca.shape[1]}")
    
    # 1. SVM Video
    best_params_vid_svm = train_svm_optuna(X_train_v_pca, y_train_v, X_val_v_pca, y_val_v, trials=10)
    best_clf_vid_svm = SVC(**best_params_vid_svm, probability=True)
    best_clf_vid_svm.fit(X_train_v_pca, y_train_v)
    vid_preds_svm = best_clf_vid_svm.predict(X_val_v_pca)
    acc_vid_svm = accuracy_score(y_val_v, vid_preds_svm)
    print(f"Video Level Accuracy (SVM + PCA): {acc_vid_svm:.4f}")
    plot_confusion_matrix(y_val_v, vid_preds_svm, le.classes_, f"Confusion Matrix - {name} (SVM)", f"cm_{name}_svm.png")
    plot_roc_curve(best_clf_vid_svm, X_val_v_pca, y_val_v, le, f"roc_{name}_svm.png")
    
    # 2. Random Forest Video
    best_params_vid_rf = train_rf_optuna(X_train_v, y_train_v, X_val_v, y_val_v, trials=10)
    best_clf_vid_rf = RandomForestClassifier(**best_params_vid_rf, random_state=42)
    best_clf_vid_rf.fit(X_train_v, y_train_v)
    vid_preds_rf = best_clf_vid_rf.predict(X_val_v)
    acc_vid_rf = accuracy_score(y_val_v, vid_preds_rf)
    print(f"Video Level Accuracy (RF): {acc_vid_rf:.4f}")
    plot_confusion_matrix(y_val_v, vid_preds_rf, le.classes_, f"Confusion Matrix - {name} (RF)", f"cm_{name}_rf.png")
    plot_roc_curve(best_clf_vid_rf, X_val_v, y_val_v, le, f"roc_{name}_rf.png")

    # 3. KNN Video
    best_params_vid_knn = train_knn_optuna(X_train_v_pca, y_train_v, X_val_v_pca, y_val_v, trials=10)
    best_clf_vid_knn = KNeighborsClassifier(**best_params_vid_knn)
    best_clf_vid_knn.fit(X_train_v_pca, y_train_v)
    vid_preds_knn = best_clf_vid_knn.predict(X_val_v_pca)
    acc_vid_knn = accuracy_score(y_val_v, vid_preds_knn)
    print(f"Video Level Accuracy (KNN + PCA): {acc_vid_knn:.4f}")
    plot_confusion_matrix(y_val_v, vid_preds_knn, le.classes_, f"Confusion Matrix - {name} (KNN)", f"cm_{name}_knn.png")
    plot_roc_curve(best_clf_vid_knn, X_val_v_pca, y_val_v, le, f"roc_{name}_knn.png")
    
    return {
        'frame_voting_acc': acc,
        'video_agg_acc_svm': acc_vid_svm,
        'video_agg_acc_rf': acc_vid_rf,
        'video_agg_acc_knn': acc_vid_knn,
    }

if __name__ == "__main__":
    try:
        print("Starting main...")
        # 1. Baseline: Uniform Sampling, MinMax
        res_baseline = run_experiment('baseline_uniform_minmax', {
            'sampling': {'strategy': 'uniform', 'n_frames': 16},
            'normalization': 'minmax'
        })
        
        # 2. Improved: Uniform, StandardScaler (Req 3)
        res_std = run_experiment('uniform_stdscaler', {
            'sampling': {'strategy': 'uniform', 'n_frames': 16},
            'normalization': 'standard'
        })
        
        # 3. Improved: Dense Sampling (Req 1), standard scaler
        res_dense = run_experiment('dense_stdscaler', {
            'sampling': {'strategy': 'dense', 'frame_skip': 5},
            'normalization': 'standard'
        })
        
        print("\n--- Summary ---")
        print("Baseline (Voting):", res_baseline['frame_voting_acc'])
        print("Baseline (SVM Agg):", res_baseline['video_agg_acc_svm'])
        print("Baseline (RF Agg):", res_baseline['video_agg_acc_rf'])
        print("Baseline (KNN Agg):", res_baseline['video_agg_acc_knn'])
        
        print("\nStdScaler (Voting):", res_std['frame_voting_acc'])
        print("StdScaler (SVM Agg):", res_std['video_agg_acc_svm'])
        print("StdScaler (RF Agg):", res_std['video_agg_acc_rf'])
        print("StdScaler (KNN Agg):", res_std['video_agg_acc_knn'])
        
        print("\nDense (Voting):", res_dense['frame_voting_acc'])
        print("Dense (SVM Agg):", res_dense['video_agg_acc_svm'])
        print("Dense (RF Agg):", res_dense['video_agg_acc_rf'])
        print("Dense (KNN Agg):", res_dense['video_agg_acc_knn'])
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"CRITICAL ERROR: {e}")
