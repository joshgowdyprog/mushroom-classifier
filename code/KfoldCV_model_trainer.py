import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef


def KfoldCV_model_method(test_df, model, X, y, n_splits=5, random_state=42):
    """Trains model multiple times using K-fold Cross Validation and returns 
    the out-of-fold MCC validation scores and the out-of-fold predicted 
    probabilities on the test data for these models.

    Args: test_df: test data
          model: classifier object
          X: features
          y: target
          n_splits: number of splits for K-fold CV (default=5)
          random_state: random state (default=42)
          Returns: oof_probs: out-of-fold predicted probabilities 
                              on test_df
                   oof_mccs: out-of-fold MCC scores
    """

    skfold = StratifiedKFold(n_splits=n_splits, shuffle=True, 
                             random_state=random_state)
    
    oof_probs, oof_mccs = [], []
    print("="*80)
    print(f"Training {model.__class__.__name__}")
    print("="*80, end="\n")

    for fold, (train_idx, test_idx) in enumerate(skfold.split(X, y)):
        X_train, y_train = X.iloc[train_idx, :], y[train_idx]
        X_test, y_test = X.iloc[test_idx, :], y[test_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mcc = matthews_corrcoef(y_pred, y_test)
        oof_mccs.append(mcc)
        oof_probs.append(model.predict_proba(test_df))
        print(f"--- Fold {fold+1} MCC Score: {mcc:.6f}")
    print(f"\n---> Mean MCC Score: {np.mean(oof_mccs):.6f} 
          \xb1 {np.std(oof_mccs):.6f}\n\n")
    return oof_probs, oof_mccs