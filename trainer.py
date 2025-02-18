import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path
import json

class CropDataset(Dataset):
    def __init__(self, features: pd.DataFrame, yields: np.array):
        self.weather_features = torch.FloatTensor(
            features.filter(like='weather').values
        )
        self.soil_features = torch.FloatTensor(
            features.filter(like='soil').values
        )
        self.satellite_features = torch.FloatTensor(
            features.filter(like='satellite').values
        )
        self.yields = torch.FloatTensor(yields)
        
    def __len__(self):
        return len(self.yields)
        
    def __getitem__(self, idx):
        return {
            'weather': self.weather_features[idx],
            'soil': self.soil_features[idx],
            'satellite': self.satellite_features[idx],
            'yield': self.yields[idx]
        }

class ModelTrainer:
    def __init__(self, model: torch.nn.Module, config: Dict, save_dir: Path):
        self.model = model
        self.config = config
        self.save_dir = save_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config['learning_rate']
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            patience=5,
            factor=0.5
        )
        self.criterion = self._gaussian_nll_loss
    
    def _gaussian_nll_loss(self, prediction: torch.Tensor, 
                          uncertainty: torch.Tensor, 
                          target: torch.Tensor) -> torch.Tensor:
        """Gaussian Negative Log Likelihood loss for uncertainty estimation"""
        return (torch.log(uncertainty) + 
                (prediction - target)**2 / (2 * uncertainty)).mean()
    
    def train(self, train_loader: DataLoader, 
              valid_loader: Optional[DataLoader] = None,
              early_stopping_patience: int = 10) -> Dict:
        """Train the model with early stopping"""
        best_valid_loss = float('inf')
        patience_counter = 0
        training_history = {
            'train_loss': [],
            'valid_loss': [],
            'learning_rates': []
        }
        
        for epoch in range(self.config['epochs']):
            # Training phase
            train_loss = self._train_epoch(train_loader)
            training_history['train_loss'].append(train_loss)
            training_history['learning_rates'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Validation phase
            if valid_loader:
                valid_loss = self._validate(valid_loader)
                training_history['valid_loss'].append(valid_loss)
                
                # Learning rate scheduling
                self.scheduler.step(valid_loss)
                
                # Early stopping check
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    patience_counter = 0
                    self._save_checkpoint(epoch, valid_loss)
                else:
                    patience_counter += 1
                
                print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, "
                      f"Valid Loss = {valid_loss:.4f}, "
                      f"LR = {self.optimizer.param_groups[0]['lr']:.6f}")
                
                if patience_counter >= early_stopping_patience:
                    print("Early stopping triggered!")
                    break
            else:
                print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, "
                      f"LR = {self.optimizer.param_groups[0]['lr']:.6f}")
        
        self._save_training_history(training_history)
        return training_history
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in train_loader:
            weather_features = batch['weather'].to(self.device)
            soil_features = batch['soil'].to(self.device)
            satellite_features = batch['satellite'].to(self.device)
            targets = batch['yield'].to(self.device)
            
            self.optimizer.zero_grad()
            predictions, uncertainties, _ = self.model(
                weather_features,
                soil_features,
                satellite_features
            )
            
            loss = self.criterion(predictions, uncertainties, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def _validate(self, valid_loader: DataLoader) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in valid_loader:
                weather_features = batch['weather'].to(self.device)
                soil_features = batch['soil'].to(self.device)
                satellite_features = batch['satellite'].to(self.device)
                targets = batch['yield'].to(self.device)
                
                predictions, uncertainties, _ = self.model(
                    weather_features,
                    soil_features,
                    satellite_features
                )
                
                loss = self.criterion(predictions, uncertainties, targets)
                total_loss += loss.item()
        
        return total_loss / len(valid_loader)
    
    def _save_checkpoint(self, epoch: int, valid_loss: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'valid_loss': valid_loss,
            'config': self.config
        }
        
        checkpoint_path = self.save_dir / f'model_checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
    
    def _save_training_history(self, history: Dict):
        """Save training history"""
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(
                {k: [float(v) for v in vals] for k, vals in history.items()},
                f,
                indent=4
            )
