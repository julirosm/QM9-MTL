import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torch.optim.lr_scheduler import _LRScheduler
from torch_geometric.loader import DataLoader
from torch.nn.modules.loss import _Loss

from typing import Optional, Union, Dict, Any, Tuple, List
import multiprocessing
import psutil
import os
import sys
import numpy as np

def log_system_info(process):
    print(f"Visible CPUs: {multiprocessing.cpu_count()}")
    print(f"PyTorch: {torch.get_num_threads()} used threads")
    print(f"[Used RAM] {process.memory_info().rss / 1e9:.2f} GB")


def create_dataloaders(dataset, batch_size, n_train, n_valid, n_test, seed=None):
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # Total samples to consider
    total_samples = n_train + n_valid + n_test

    if total_samples > len(dataset):
        raise ValueError(f"Requested total samples ({total_samples}) exceeds dataset size ({len(dataset)})")

    # Generate random permutation of indices
    indices = np.random.permutation(len(dataset))[:total_samples]

    # Split indices
    train_indices = indices[:n_train]
    valid_indices = indices[n_train:n_train + n_valid]
    test_indices  = indices[n_train + n_valid:n_train + n_valid + n_test]

    # Create subsets
    train_subset = Subset(dataset, train_indices)
    valid_subset = Subset(dataset, valid_indices)
    test_subset  = Subset(dataset, test_indices)

    # Create dataloaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    print(f"Dataloaders created with {n_train} training samples, {n_valid} validation samples and {n_test} testing samples\n")

    return train_loader, valid_loader, test_loader


class Multitasking_Loss(_Loss):
    def __init__(self, loss_function):
        super().__init__()
        self.loss_function = loss_function
    
    def forward(self, prediction, target):
        loss = self.loss_function(prediction, target, reduction='none')
        loss = loss.mean(dim=0)
        return loss

def l1_loss(prediction, target):
        loss = F.l1_loss(prediction, target, reduction='none')
        loss = loss.mean(dim=0)
        return loss

def compute_weighted_loss(loss_tensor, weights):
    return (loss_tensor*weights).sum()

def train(model, data, optimizer, criterion, loss_normalization, weights, device):
    data = data.to(device)
    optimizer.zero_grad()
    y_pred = model(data)
    losses = criterion(y_pred.view(-1,model.out_channels), data.y.view(-1,model.out_channels))/loss_normalization
    weighted_loss = compute_weighted_loss(losses, weights)
    weighted_loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  
    optimizer.step()
    return losses

def evaluate(model, valid_loader, criterion, device, return_predictions = False):  
    model.eval()
    total_loss = torch.zeros(model.out_channels, device=device)
    total_mae = torch.zeros(model.out_channels, device=device)
    predictions = []
    real_values = []
    with torch.no_grad():
        for data in valid_loader:
            data = data.to(device)
            y_pred = model(data).view(-1,model.out_channels)
            predictions.append(y_pred)
            real_values.append(data.y.view(-1,model.out_channels))
            loss = criterion(y_pred, data.y.view(-1,model.out_channels))
            total_loss += loss.detach()
            total_mae += l1_loss(y_pred, data.y.view(-1,model.out_channels)).detach() 

    if return_predictions:
        predictions = torch.vstack(predictions)
        real_values = torch.vstack(real_values)
        return total_loss/len(valid_loader), total_mae/len(valid_loader), real_values, predictions
    else:
        return total_loss/len(valid_loader), total_mae/len(valid_loader)

def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    target: list[int] = list(range(3,15)),
    criterion: torch.nn.Module = nn.MSELoss(),
    epochs: int = 100,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    scheduler: Optional[Union[str, _LRScheduler]] = None,
    scheduler_kwargs: Optional[Dict] = None,
    device: str = 'cuda',
    loss_weights: Optional[Union[List[float],Tensor]] = None,
    early_stopping_patience: float = 20,
    checkpoint_path: Optional[str] = None
):
    
    # Handle device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    model.to(device)

    # Handle loss_weights
    if loss_weights is not None:
        if isinstance(loss_weights, list):
            loss_weights = torch.tensor(loss_weights, device=device)
        else:
            loss_weights = loss_weights.to(device)
    else:
        num_losses = model.out_channels
        loss_weights = torch.ones((num_losses,), dtype=torch.float32, device=device) / num_losses

    # Weights for normalizing multitasking loss (std's)
    loss_normalization = torch.tensor([1,1,1,1.530394,8.187793,0.602227,1.2771955,1.2930548,279.75717,
                                         0.9054288,10.377184,10.470608,10.5441885,9.548344,6.1265764],
                                         device=device)
    loss_normalization = loss_normalization[target]
    if criterion.loss_function is F.mse_loss:
        loss_normalization = loss_normalization**2
    elif criterion.loss_function is F.l1_loss:
        pass
    else:
        raise ValueError(f"Unsupported multitasking_loss: {criterion.multitasking_loss}")
    
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Handle scheduler creation
    scheduler_kwargs = scheduler_kwargs or {}  # <- make sure it's a dict
    if isinstance(scheduler, str):
        scheduler = scheduler.lower()
        scheduler_kwargs = scheduler_kwargs or {}  # Ensure dict exists
        
        if scheduler == "exponential":
            default_kwargs = {"gamma": 0.96}  # Default values
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, 
                **{**default_kwargs, **scheduler_kwargs}  # Merge defaults with user kwargs
            )
            
        elif scheduler == "cosine":
            default_kwargs = {"T_max": epochs, "eta_min": 0}  # Defaults
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                **{**default_kwargs, **scheduler_kwargs}
            )
            
        elif scheduler == "reduce_on_plateau":
            default_kwargs = {"mode": "min", "factor": 0.1, "patience": 10}
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                **{**default_kwargs, **scheduler_kwargs}
            )
        elif scheduler == "onecyclelr":
            default_kwargs = {
                "max_lr": 1e-2,
                "epochs": epochs,
                "steps_per_epoch": len(train_loader),
            }
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                **{**default_kwargs, **scheduler_kwargs}
            )

            
        else:
            raise ValueError(f"Unknown scheduler: {scheduler}")
    
    # Variables initialization
    training_losses = []
    validation_losses = []
    weighted_losses = []
    weighted_val_losses = []
    start_epoch = 1
    best_val_loss = float('inf')
    best_model_state_dict = None
    epochs_no_improve = 0  

    # Handle checkpoint
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model and optimizer states
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore training state variables
        start_epoch: int = checkpoint.get('epoch', 1) + 1
        weighted_losses: float = checkpoint['training_loss']
        weighted_val_losses: float = checkpoint['validation_loss']
        training_losses: list[float] = checkpoint['training_losses']
        validation_losses: list[float] = checkpoint['validation_losses']
        epochs_no_improve: int = checkpoint['epochs_no_improve']
        best_val_loss: float = checkpoint['best_val_loss']
        best_model_state_dict: dict = checkpoint['best_model_state_dict']
    else:
        with open("metrics.csv", "w") as f:
            f.write("epoch,property,train_loss,val_loss,val_mae\n")
        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints")

    
    # Feedback
    process = psutil.Process(os.getpid())
    log_system_info(process)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")
    print("Start training:")    

    # Training loop
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = torch.zeros(model.out_channels, device=device)
        for data in train_loader:
            losses = train(model, data, optimizer, criterion, loss_normalization, loss_weights, device)
            train_loss += losses.detach().view(-1)
            # Onecycle step
            if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()
        
        # Other schedulers step
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        # Compute losses
        train_loss = train_loss/len(train_loader)
        weighted_loss = compute_weighted_loss(train_loss, loss_weights)
        val_loss, val_mae = evaluate(model, valid_loader, criterion, device)
        val_loss = val_loss/loss_normalization
        weighted_val_loss = compute_weighted_loss(val_loss, loss_weights)
        
        # Track loss
        training_losses.append(train_loss)
        validation_losses.append(val_loss)
        weighted_losses.append(weighted_loss)
        weighted_val_losses.append(weighted_val_loss)
        
        # Track best model
        if weighted_val_loss < best_val_loss - 1e-8:  # Small delta to avoid noise
            best_val_loss = weighted_val_loss
            best_model_state_dict = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        # Loggings   
        if epoch % 10 == 0:
            log_system_info(process)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'training_loss': weighted_losses,
                'validation_loss': weighted_val_losses,
                'training_losses': training_losses,
                'validation_losses':validation_losses,
                'epochs_no_improve':epochs_no_improve,
                'best_val_loss':best_val_loss,
                'best_model_state_dict':best_model_state_dict 
            }, f'checkpoints/checkpoint_epoch_{epoch}.pth')
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch: {epoch:03d}, LR: {current_lr:.6f}, "
                  f"Train Loss: {weighted_loss:.7f}, Val Loss: {weighted_val_loss:.7f}")
            
            for i,(t_loss, v_loss, v_mae) in enumerate(zip(train_loss, val_loss, val_mae)):
                print(f"Property: {i}, "
                  f"Train Loss: {t_loss:.4f}, Val Loss: {v_loss:.4f}, Val MAE:{v_mae:.4f}")
                with open("metrics.csv", "a") as f:
                    f.write(f"{epoch},{i},{t_loss:.4f},{v_loss:.4f},{v_mae:.4f}\n")
                    f.flush()
            sys.stdout.flush()

        # Early stopping check
        if epochs_no_improve >= early_stopping_patience:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
    
    # Restore best model
    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)


    print("\n Training finished!")
    training_losses=torch.vstack(training_losses)
    validation_losses=torch.vstack(validation_losses)
    weighted_losses=torch.vstack(weighted_losses)
    weighted_val_losses = torch.vstack(weighted_val_losses)
    return training_losses, validation_losses, weighted_losses, weighted_val_losses