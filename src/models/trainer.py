import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss
from tqdm import tqdm

def train_model(data_path="data/processed"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize dataset and model
    dataset = ClimateDataset(data_path)
    model = DeepFusionModel(num_districts=len(dataset.district_map)).to(device)
    
    # Data loading
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Training setup
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = MSELoss()
    
    best_loss = float('inf')
    for epoch in range(100):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs = {k: v.to(device) for k,v in batch.items() if k != 'yield'}
            targets = batch['yield'].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_model.pth')
