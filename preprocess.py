import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def clean_data(df):
    """
    Performs initial data cleaning: dropping duplicates and imputing missing values.
    """
    # Drop duplicates
    initial_count = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {initial_count - len(df)} duplicate rows.")

    # Identify columns
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    # Impute missing values
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    if cat_cols:
        df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
    
    return df

def treat_outliers(df):
    """
    Caps outliers for Price and Lead Time at the 99th percentile.
    """
    q99_price = df['avg_price_per_room'].quantile(0.99)
    df.loc[df['avg_price_per_room'] > q99_price, 'avg_price_per_room'] = q99_price
            
    q99_lead = df['lead_time'].quantile(0.99)
    df.loc[df['lead_time'] > q99_lead, 'lead_time'] = q99_lead
    
    print(f"Capped outliers: Price > {q99_price:.2f}, Lead Time > {q99_lead:.2f}")
    return df

def get_preprocessor(X):
    """
    Creates the ColumnTransformer for preprocessing.
    """
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

    print(f"Categorical columns: {list(categorical_cols)}")
    print(f"Numerical columns count: {len(numerical_cols)}")

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])
    
    return preprocessor

def split_data(df, target_col='booking_status'):
    """
    Prepares X and y and splits them into train and test sets.
    """
    # Drop ID column if exists
    if 'Booking_ID' in df.columns:
        df = df.drop(columns=['Booking_ID'])

    X = df.drop(columns=[target_col])
    # Map target variable
    y = df[target_col].map({'Not_Canceled': 0, 'Canceled': 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test