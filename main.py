import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

from data_utils import load_preprocessed, split_and_scale
from models import DeepFusionNN, CNNLSTM
from train_utils import train_pytorch_model
from metrics import calculate_regression_metrics, plot_regression_metrics

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
os.makedirs("saved_models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

def main():
    X, Y = load_preprocessed()
    X_train, X_test, Y_train, Y_test = split_and_scale(X, Y)
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled), 
        torch.FloatTensor(Y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_scaled), 
        torch.FloatTensor(Y_test)
    )
    
    batch_size = 64
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        pin_memory=True
    )

    # --------------------------
    # Model Initialization and Training
    # --------------------------
    input_dim = X_train_scaled.shape[1]
    output_dim = Y_train.shape[1]
    
    dl_models = {
        "DeepFusionNN": DeepFusionNN(input_dim, output_dim),
        "CNNLSTM": CNNLSTM(input_dim, output_dim)
    }
    
    # Train deep learning models
    dl_histories = {}
    for name, model in dl_models.items():
        print(f"\nTraining {name}")
        dl_histories[name] = train_pytorch_model(
            model, 
            train_loader, 
            test_loader,
            device
        )
        # Load best weights for evaluation
        model.load_state_dict(
            torch.load(f"saved_models/best_{name}.pth", map_location=device)
        )

    # --------------------------
    # Traditional Model Training
    # --------------------------
    print("\nTraining Traditional Models...")
    # Random Forest
    rf = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=500, 
            max_depth=12, 
            random_state=42,
            n_jobs=-1
        )
    )
    rf.fit(X_train_scaled, Y_train)
    joblib.dump(rf, "saved_models/RandomForest.joblib")

    # XGBoost
    xgb = MultiOutputRegressor(
        XGBRegressor(
            n_estimators=500, 
            learning_rate=0.05, 
            max_depth=7,
            tree_method='hist'
        )
    )
    xgb.fit(X_train_scaled, Y_train)
    joblib.dump(xgb, "saved_models/XGBoost.joblib")

    # --------------------------
    # Metrics Calculation and Plotting
    # --------------------------
    all_metrics = {}
    n_features = X_train_scaled.shape[1]

    # Evaluate DL models
    for name, model in dl_models.items():
        model.eval()
        test_preds = []
        with torch.no_grad():
            for xb, _ in test_loader:
                xb = xb.to(device)
                outputs = model(xb)
                test_preds.append(outputs.cpu().numpy())
        
        y_pred = np.concatenate(test_preds)
        all_metrics[name] = calculate_regression_metrics(
            Y_test, 
            y_pred, 
            n_features
        )

    # Evaluate traditional models
    traditional_models = {
        "RandomForest": rf,
        "XGBoost": xgb
    }
    
    for name, model in traditional_models.items():
        y_pred = model.predict(X_test_scaled)
        all_metrics[name] = calculate_regression_metrics(
            Y_test, 
            y_pred, 
            n_features
        )

    # Plot comparison metrics
    plot_regression_metrics(all_metrics)
    
    # Print final metrics
    print("\nFinal Model Performance Comparison:")
    for model_name, metrics in all_metrics.items():
        print(f"\n{model_name}:")
        print(f"  RMSE: {metrics['RMSE']:.2f}")
        print(f"  MAE: {metrics['MAE']:.2f}")
        print(f"  R²: {metrics['R2']:.2f}")
        print(f"  Adjusted R²: {metrics['Adjusted R2']:.2f}")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
