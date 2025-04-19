import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def load_preprocessed(filepath="./Downloads/preprocessed_crop_data.csv"):
    df = pd.read_csv(filepath)
    feature_cols = [
        'T2M', 'RH2M', 'NDVI', 'VCI_(%)', 'GDD',
        'Moisture_Stress', 'Thermal_Time', 'LATITUDE', 'LONGITUDE'
    ]
    yield_cols = [col for col in df.columns if 'YIELD' in col]
    X = df[feature_cols].fillna(df[feature_cols].median())
    Y = df[yield_cols].fillna(0)
    return X.values, Y.values

def split_and_scale(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, "saved_models/feature_scaler.joblib")
    return X_train_scaled, X_test_scaled, Y_train, Y_test
