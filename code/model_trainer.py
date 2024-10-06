import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (matthews_corrcoef, f1_score, 
                                confusion_matrix, ConfusionMatrixDisplay)


def train_validate_model(classifier, X, y, validation_split=0.2):
    """Trains and validates a classifier on the given data. Validates using 
    their F1 and MCC scores and a confusion matrix.
    Args: classifier: classifier object
            X: features
            y: target
            validation_split: validation split ratio (default=0.2)
    """

    print("="*80)
    print(f"    Model: {classifier.__class__.__name__}")
    print("="*80)
    
    (X_train, X_test, 
     y_train, y_test) = train_test_split(X, y, train_size=validation_split, 
                                        shuffle=True, stratify=y, 
                                        random_state=42)
    
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    print(f"F1 Score : {f1.mean():.6f}")
    print(f"MCC Score: {mcc.mean():.6f}")
    
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot()
    plt.title("Confusion Matrix")
    plt.show()

    print()

    return classifier, f1, mcc

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

def train_baseline_ensemble_of_models(X, y):
    """
    Instatiates and trains and validates the baseline model
      - XGBoost, Random Forest, CatBoost or LightGBM.
    Args: X: features
          y: target
    prints: the F1 and MCC scores and the confusion matrix for each model
    returns: the four baseline classifiers
    """

    xgb_clf = XGBClassifier(enable_categorical=True, 
                            device="cuda", tree_method="hist")
    xbg_baseline_clf, _, _ = model_train_validate(xgb_clf, X, y)

    rf_clf = RandomForestClassifier(criterion="gini", random_state=42)
    rf_baseline_clf, _, _ = model_train_validate(rf_clf, X, y)
    
    cat_clf = CatBoostClassifier(verbose=False,
                                allow_writing_files=False,)
    cat_baseline_clf, _, _ = model_train_validate(cat_clf, X, y)

    lgbm_clf = LGBMClassifier(verbosity=-1)
    lgbm_baseline_clf, _, _ = model_train_validate(lgbm_clf, X, y)

    return xbg_baseline_clf, rf_baseline_clf, cat_baseline_clf, lgbm_baseline_clf
