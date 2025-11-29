from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import GridSearchCV

def train_models(X_train, y_train, preprocessor):
    """
    Trains multiple models and tunes Random Forest.
    
    Returns:
        dict: Dictionary containing trained pipelines/models.
    """
    models = {}
    
    # Base Models definition
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    # Training Base Models
    for name, clf in classifiers.items():
        print(f"Training {name}...")
        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', clf)
        ])
        pipeline.fit(X_train, y_train)
        models[name] = pipeline
        print(f"{name} trained.")

    # Tuning Random Forest
    print("Tuning Random Forest...")
    rf_pipeline = models['Random Forest']
    
    param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [10, 20],
        'classifier__min_samples_leaf': [1, 2]
    }
    
    grid_search = GridSearchCV(
        rf_pipeline, 
        param_grid, 
        cv=3, 
        scoring='f1', 
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    print(f"Best Params: {grid_search.best_params_}")
    
    models['Random Forest Tuned'] = grid_search.best_estimator_
    
    return models