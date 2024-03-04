import torch
from src.utility import cauchy_stress_vectorized
import numpy as np


def cal_loss(model,val_loader,criterion):
    losses = []
    for batch_idx, (invariants, bases, target_stress) in enumerate(val_loader):
        base1,base2,base3,base4,base5 = bases
        psi = model(invariants)
        dPsidX = torch.autograd.grad(psi, invariants, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
        sigma = cauchy_stress_vectorized(invariants,dPsidX,base1,base2,base3,base4,base5)
        loss = criterion(sigma, target_stress)
        losses.append(loss.item())
    return np.sum(losses)


def train(model, train_loader, val_loader, optimizer, criterion, num_epochs, 
          print_every=100, 
          save_models=False, 
          patience=10, 
          tolerance=0.001,
          update_ui = None):
    """
    Trains a given model using the provided data loaders and optimization parameters.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): The data loader for the training set.
        val_loader (torch.utils.data.DataLoader): The data loader for the validation set.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        criterion (callable): The loss function used for training.
        num_epochs (int): The number of training epochs.
        print_every (int, optional): The frequency of printing training progress. Defaults to 100.
        save_models (bool, optional): Whether to save the best model during training. Defaults to False.
        patience (int, optional): The number of epochs to wait for improvement in validation loss before early stopping. Defaults to 10.
        tolerance (float, optional): The minimum improvement in validation loss required to be considered as improvement. Defaults to 0.001.

    Returns:
        tuple: A tuple containing the training losses, validation losses, and a flag indicating if early stopping was triggered.
    """
    model.train()
    losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(num_epochs):
        model.train()  # Ensure the model is in training mode
        running_train_loss = 0.0

        for batch_idx, (invariants, bases, target_stress) in enumerate(train_loader):
            base1, base2, base3, base4, base5 = bases
            optimizer.zero_grad()
            psi = model(invariants)
            dPsidX = torch.autograd.grad(psi, invariants, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
            sigma = cauchy_stress_vectorized(invariants, dPsidX, base1, base2, base3, base4, base5)
            train_loss = criterion(sigma, target_stress)
            train_loss.backward()
            optimizer.step()
            
            running_train_loss += train_loss.item()

        # Calculate validation loss
        model.eval()  # Set the model to evaluation mode
        running_val_loss = cal_loss(model, val_loader, criterion)
        
        val_losses.append(running_val_loss)

        # Check if validation loss improved within the tolerance
        improvement = best_val_loss - running_val_loss
        if improvement > tolerance:
            best_val_loss = running_val_loss
            epochs_no_improve = 0
            if save_models:
                torch.save(model.state_dict(), 'best_model_custom.pth')
        else:
            epochs_no_improve += 1

        if (epoch + 1) % print_every == 0 or epochs_no_improve == patience:
            message = f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_train_loss:.4f}, Validation Loss: {running_val_loss:.4f}'
            if update_ui is not None:
                update_ui(message)
            print(message)

        if epochs_no_improve >= patience:
            message = 'Early stopping triggered. Stopping training.'
            if update_ui is not None:
                update_ui(message)
            print(message)
            early_stop = True
            break  # Break out of the loop

        losses.append(running_train_loss)

    return losses, val_losses, early_stop
