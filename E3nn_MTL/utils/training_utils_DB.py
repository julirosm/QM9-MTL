from typing import Optional, Union, Dict
import psutil
import os
import sys


import torch
from torch import Tensor
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler

from utils.training_utils import evaluate, log_system_info

def train(model, data, optimizer, criterion, device):
    data = data.to(device)
    optimizer.zero_grad()
    y_pred = model(data)
    losses = criterion(y_pred.view(-1,model.out_channels), data.y.view(-1,model.out_channels))
    model.backward(losses)
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  
    optimizer.step()
    return losses

def train_model_DB(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    reference_values: Optional[Tensor] = None,
    target: list[int] = list(range(3,15)),
    criterion: torch.nn.Module = nn.MSELoss(),
    epochs: int = 100,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    scheduler: Optional[Union[str, _LRScheduler]] = None,
    scheduler_kwargs: Optional[Dict] = None,
    device: str = 'cuda',
    early_stopping_patience: float = 20,
    checkpoint_path: Optional[str] = None
):
    
    # Handle device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    model.to(device)    
    
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
    start_epoch = 1
    best_delta = -float('inf')
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
        training_losses: list[float] = checkpoint['training_losses']
        validation_losses: list[float] = checkpoint['validation_losses']
        epochs_no_improve: int = checkpoint['epochs_no_improve']
        best_delta: float = checkpoint['best_delta']
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
            losses = train(model, data, optimizer, criterion, device)
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
        val_loss, val_mae = evaluate(model, valid_loader, criterion, device)
        delta = torch.mean((reference_values-val_mae)/reference_values)*100 

        
        # Track loss
        training_losses.append(train_loss)
        validation_losses.append(val_loss)
        
        # Track best model
        if delta > best_delta + 1e-8:  
            best_delta = delta
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
                'training_losses': training_losses,
                'validation_losses':validation_losses,
                'epochs_no_improve':epochs_no_improve,
                'best_delta':best_delta,
                'best_model_state_dict':best_model_state_dict 
            }, f'checkpoints/checkpoint_epoch_{epoch}.pth')
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch: {epoch:03d}, LR: {current_lr:.6f}, Delta: {delta:.4f}%")
            
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
    return training_losses, validation_losses