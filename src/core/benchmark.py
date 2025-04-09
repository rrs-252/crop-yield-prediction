from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

def benchmark_models(X, y):
    """Compare model performance with temporal validation"""
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=200),
        'XGBoost': XGBRegressor(),
        'DeepFusionNN': load_pretrained_model()  # Existing model
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    results = {}
    
    for name, model in models.items():
        rmse_scores = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            rmse_scores.append(np.sqrt(mean_squared_error(y_test, preds)))
        
        results[name] = {
            'mean_rmse': np.mean(rmse_scores),
            'std_rmse': np.std(rmse_scores)
        }
    
    return results
