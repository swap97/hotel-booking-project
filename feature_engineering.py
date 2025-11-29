import pandas as pd

def create_features(df):
    """
    Performs feature engineering on the dataframe.
    """
    # Date processing
    # Note: Based on the notebook logic. Ensure 'arrival_date' is compatible.
    if 'arrival_date' in df.columns:
        df['arrival_date'] = pd.to_datetime(df['arrival_date'], errors='coerce')
        df['arrival_month'] = df['arrival_date'].dt.month
        df['arrival_day'] = df['arrival_date'].dt.day

    # Interaction features
    df['total_stay_nights'] = df['no_of_weekend_nights'] + df['no_of_week_nights']
    df['total_guests'] = df['no_of_adults'] + df['no_of_children']
    
    # Binning Lead Time
    df['lead_time_category'] = pd.cut(
        df['lead_time'], 
        bins=[-1, 7, 30, 1000], 
        labels=['Short', 'Medium', 'Long']
    )

    # derived metrics
    df['price_per_person'] = df['avg_price_per_room'] / (df['total_guests'] + 1e-5)
    df['is_weekend_booking'] = (df['no_of_weekend_nights'] > 0).astype(int)

    return df