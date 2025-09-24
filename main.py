import numpy as np
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt

def load_data():
    wine = load_wine()
    
    X = wine.data
    y = wine.target
    
    # Select only classes 0 and 1 (remove class 2) as per rubric
    mask = (y == 0) | (y == 1)
    X = X[mask]
    y = y[mask]
    
   # print(f"Dataset loaded:")
   # print(f"  X shape: {X.shape}")
   # print(f"  y shape: {y.shape}")
    return X, y

def split_data(X, y, test_size=0.2, random_state=1):
    """Split data into train/test sets."""
    np.random.seed(random_state)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    # Shuffle indices
    indices = np.random.permutation(n_samples)
    
    # Split indices
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    # Split data
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    from lr.gradient_descent import LogisticRegressionGD
    
    X, y = load_data()
    
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Normalize features 
    X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std  
    
    print("\n" + "="*60)
    print("LOGISTIC REGRESSION IMPLEMENTATION")
    print("="*60)
    
    # 1. Train Custom Logistic Regression
    print("\n1. CUSTOM LOGISTIC REGRESSION:")
    gd_model = LogisticRegressionGD(learning_rate=0.01, max_iter=200)
    gd_model.fit(X_train, y_train)
    
    print(f"   Converged in {gd_model.n_iter_} iterations")
    print(f"   Final cost: {gd_model.cost_history_[-1]:.6f}")
    print(f"   Coefficients shape: {gd_model.coef_.shape}")
    print(f"   Intercept: {gd_model.intercept_:.4f}")
    
    # 2. Train Scikit-learn Reference
    print("\n2. SCIKIT-LEARN REFERENCE:")
    sk_model = SklearnLogisticRegression(max_iter=1000, random_state=1)
    sk_model.fit(X_train, y_train)
    
    print(f"   Converged: {sk_model.n_iter_[0]} iterations")
    print(f"   Coefficients shape: {sk_model.coef_.shape}")
    print(f"   Intercept: {sk_model.intercept_[0]:.4f}")
    
    # 3. Generate Predictions
    gd_pred = gd_model.predict(X_test)
    sk_pred = sk_model.predict(X_test)
    
    # Generate probabilities for ROC curve
    gd_proba = gd_model.predict_proba(X_test)
    sk_proba = sk_model.predict_proba(X_test)[:, 1]  # Positive class probability
    
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    # 1. Confusion Matrices
    print("\n1. CONFUSION MATRICES:")
    gd_cm = confusion_matrix(y_test, gd_pred)
    sk_cm = confusion_matrix(y_test, sk_pred)
    
    print("   Custom Implementation:")
    print(f"     {gd_cm}")
    print("     [[TN, FP]")
    print("      [FN, TP]]")
    
    print("   Scikit-learn:")
    print(f"     {sk_cm}")
    
    # 2. Accuracy
    print("\n2. ACCURACY:")
    gd_accuracy = accuracy_score(y_test, gd_pred)
    sk_accuracy = accuracy_score(y_test, sk_pred)
    print(f"   Custom Implementation: {gd_accuracy:.4f}")
    print(f"   Scikit-learn:         {sk_accuracy:.4f}")
    print(f"   Difference:           {abs(gd_accuracy - sk_accuracy):.4f}")
    
    # 3. Precision
    print("\n3. PRECISION:")
    gd_precision = precision_score(y_test, gd_pred)
    sk_precision = precision_score(y_test, sk_pred)
    print(f"   Custom Implementation: {gd_precision:.4f}")
    print(f"   Scikit-learn:         {sk_precision:.4f}")
    print(f"   Difference:           {abs(gd_precision - sk_precision):.4f}")
    
    # 4. Recall
    print("\n4. RECALL:")
    gd_recall = recall_score(y_test, gd_pred)
    sk_recall = recall_score(y_test, sk_pred)
    print(f"   Custom Implementation: {gd_recall:.4f}")
    print(f"   Scikit-learn:         {sk_recall:.4f}")
    print(f"   Difference:           {abs(gd_recall - sk_recall):.4f}")
    
    # 5. ROC Curve and AUC
    print("\n5. ROC CURVE:")
    gd_fpr, gd_tpr, _ = roc_curve(y_test, gd_proba)
    sk_fpr, sk_tpr, _ = roc_curve(y_test, sk_proba)
    gd_auc = auc(gd_fpr, gd_tpr)
    sk_auc = auc(sk_fpr, sk_tpr)
    
    print(f"   Custom Implementation AUC: {gd_auc:.4f}")
    print(f"   Scikit-learn AUC:          {sk_auc:.4f}")
    print(f"   AUC Difference:            {abs(gd_auc - sk_auc):.4f}")
    
    # Summary Table
    print("\n6. SUMMARY TABLE:")
    print("   " + "="*50)
    print("   Metric       | Custom  | Sklearn | Diff")
    print("   " + "-"*50)
    print(f"   Accuracy     | {gd_accuracy:.4f}  | {sk_accuracy:.4f}  | {abs(gd_accuracy-sk_accuracy):.4f}")
    print(f"   Precision    | {gd_precision:.4f}  | {sk_precision:.4f}  | {abs(gd_precision-sk_precision):.4f}")
    print(f"   Recall       | {gd_recall:.4f}  | {sk_recall:.4f}  | {abs(gd_recall-sk_recall):.4f}")
    print(f"   ROC AUC      | {gd_auc:.4f}  | {sk_auc:.4f}  | {abs(gd_auc-sk_auc):.4f}")
    print("   " + "="*50)
    

