from data_loader import load_data
from preprocess import clean_data, treat_outliers, split_data, get_preprocessor
from feature_engineering import create_features
from train import train_models
from evaluate import evaluate_models, save_best_model, extract_feature_importance

def main():
    # 1. Load Data
    file_path = 'data/Hotel Reservations.csv' # Adjust path as needed
    df = load_data(file_path)
    
    if df is not None:
        # 2. Cleaning and Feature Engineering
        df = clean_data(df)
        df = create_features(df)
        df = treat_outliers(df)
        
        # 3. Preprocessing (Split and pipeline setup)
        X_train, X_test, y_train, y_test = split_data(df)
        preprocessor = get_preprocessor(X_train)
        
        # 4. Training
        models = train_models(X_train, y_train, preprocessor)
        
        # 5. Evaluation
        evaluate_models(models, X_test, y_test)
        
        # 6. Feature Importance and Save
        best_model_name = 'Random Forest Tuned'
        if best_model_name in models:
            extract_feature_importance(models[best_model_name], best_model_name)
            save_best_model(models[best_model_name])

if __name__ == "__main__":
    main()