import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score
)
from member2_activity.config import ML_MODEL_PATH, ML_SCALER_PATH, BASE_DIR
from member2_activity.dataset_generator import generate_synthetic_dataset


def train_and_evaluate_model():
    """Trains ML classifier, computes evaluation metrics, and saves model artifacts."""
    print("Generating synthetic dataset for suspicious activity classification...")
    X, y, feature_names = generate_synthetic_dataset(num_samples_per_class=300, seed=42)

    # Train / Test split (75% train, 25% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model training (Random Forest)
    clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        min_samples_split=4,
        random_state=42
    )
    clf.fit(X_train_scaled, y_train)

    # Predictions & evaluation
    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    cls_report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    
    try:
        roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
    except Exception:
        roc_auc = 0.0

    # Save trained model and scaler artifacts
    os.makedirs(os.path.dirname(ML_MODEL_PATH), exist_ok=True)
    joblib.dump(clf, ML_MODEL_PATH)
    joblib.dump(scaler, ML_SCALER_PATH)

    report_content = []
    report_content.append("==================================================")
    report_content.append("MEMBER 2: ML MODEL EVALUATION REPORT")
    report_content.append("==================================================")
    report_content.append(f"Model: RandomForestClassifier (n_estimators=120, max_depth=12)")
    report_content.append(f"Dataset Size: {len(X)} samples (75% Train, 25% Test)")
    report_content.append(f"Accuracy: {acc * 100:.2f}%")
    report_content.append(f"Multiclass ROC-AUC (OvR): {roc_auc:.4f}\n")
    report_content.append("--- Classification Report ---")
    report_content.append(cls_report)
    report_content.append("--- Confusion Matrix ---")
    report_content.append(f"Classes: {list(clf.classes_)}")
    report_content.append(str(cm))
    report_content.append("\n--- Feature Importances ---")
    for name, imp in zip(feature_names, clf.feature_importances_):
        report_content.append(f"  - {name:25s}: {imp:.4f}")

    report_str = "\n".join(report_content)
    print(report_str)

    report_path = os.path.join(BASE_DIR, "ml_evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write(report_str)

    print(f"\n[Success] Model saved to: {ML_MODEL_PATH}")
    print(f"[Success] Scaler saved to: {ML_SCALER_PATH}")
    print(f"[Success] Evaluation report written to: {report_path}")

    return clf, scaler


if __name__ == "__main__":
    train_and_evaluate_model()
