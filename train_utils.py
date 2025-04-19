import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error

def train_pytorch_model(model, train_loader, test_loader, epochs=300):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.HuberLoss()
    
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    best_rmse = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        # Store training loss
        history['train_loss'].append(total_loss/len(train_loader))
        
        # Validation
        model.eval()
        val_loss = 0
        preds = []
        truths = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                preds.append(outputs.cpu().numpy())
                truths.append(yb.cpu().numpy())
                val_loss += criterion(outputs, yb).item()
        
        # Store validation loss
        history['val_loss'].append(val_loss/len(test_loader))
        val_rmse = np.sqrt(mean_squared_error(np.concatenate(truths), np.concatenate(preds)))
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(model.state_dict(), f"saved_models/best_{model.__class__.__name__}.pth")
        
        print(f"Epoch {epoch+1}: Train Loss={history['train_loss'][-1]:.4f}, Val Loss={history['val_loss'][-1]:.4f}, Val RMSE={val_rmse:.2f}")
    
    return history

