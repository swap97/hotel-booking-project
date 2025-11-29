import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib

def evaluate_models(models, X_test, y_test):
    """
    Evaluates all trained models and prints metrics.
    """
    results = []
    
    for name, model in models.items():
        try:
            y_pred = model.predict(X_test)
            # Check if model supports predict_proba
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = [0] * len(y_pred)
            
            metrics = {
                'Model': name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1 Score': f1_score(y_test, y_pred),
                'ROC AUC': roc_auc_score(y_test, y_prob)
            }
            results.append(metrics)
            
            print(f"\nConfusion Matrix: {name}")
            print(confusion_matrix(y_test, y_pred))
            
        except Exception as e:
            print(f"Error evaluating {name}: {e}")

    results_df = pd.DataFrame(results)
    print("\n--- Model Performance Comparison ---")
    print(results_df)
    return results_df

def save_best_model(model, filename='best_hotel_booking_model.joblib'):
    """
    Saves the trained model to disk.
    """
    if model:
        joblib.dump(model, filename)
        print(f"Best model saved to {filename}")
    else:
        print("No model provided to save.")

def extract_feature_importance(model_pipeline, model_name='Random Forest Tuned'):
    """
    Extracts and prints feature importance for tree-based models within the pipeline.
    """
    try:
        classifier = model_pipeline.named_steps['classifier']
        preprocessor = model_pipeline.named_steps['preprocessor']
        
        if not hasattr(classifier, 'feature_importances_'):
            return

        ohe = preprocessor.named_transformers_['cat']
        cat_features = ohe.get_feature_names_out()
        num_features = preprocessor.named_transformers_['num'].get_feature_names_out()
        feature_names = np.r_[num_features, cat_features]
        
        importances = classifier.feature_importances_
        
        if len(feature_names) == len(importances):
            feature_imp_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
            
            print(f"\n--- Top 10 Features ({model_name}) ---")
            print(feature_imp_df.head(10))
    except Exception as e:
        print(f"Could not extract feature importance: {e}")