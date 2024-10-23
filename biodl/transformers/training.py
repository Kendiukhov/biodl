import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional, Dict, Any
import time
import os
import copy

class Trainer:
    """
    Trainer class to manage the training and validation of models.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        scheduler: Optional[_LRScheduler] = None,
        device: Optional[torch.device] = None,
        save_dir: str = 'checkpoints',
        max_grad_norm: Optional[float] = None
    ):
        """
        Initializes the Trainer.

        Args:
            model: The model to train.
            optimizer: Optimizer for training.
            criterion: Loss function.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data (optional).
            scheduler: Learning rate scheduler (optional).
            device: Device to run training on (CPU or GPU).
            save_dir: Directory to save model checkpoints.
            max_grad_norm: Maximum gradient norm for gradient clipping (optional).
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = save_dir
        self.max_grad_norm = max_grad_norm

        self.model.to(self.device)

        os.makedirs(self.save_dir, exist_ok=True)

    def train(
        self,
        num_epochs: int,
        log_interval: int = 100,
        early_stopping_patience: Optional[int] = None
    ):
        """
        Runs the training loop.

        Args:
            num_epochs: Number of epochs to train.
            log_interval: Steps between logging training status.
            early_stopping_patience: Number of epochs with no improvement after which training will be stopped.
        """
        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()
            train_loss = self._train_epoch(epoch, log_interval)
            val_loss = None

            if self.val_loader:
                val_loss = self._evaluate()

                print('-' * 89)
                print(f'| end of epoch {epoch} | time: {(time.time() - epoch_start_time):.2f}s | '
                      f'train loss {train_loss:.4f} | val loss {val_loss:.4f}')
                print('-' * 89)

                # Check for improvement
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    # Save the best model
                    self._save_checkpoint(epoch, best=True)
                else:
                    epochs_no_improve += 1
                    if early_stopping_patience and epochs_no_improve >= early_stopping_patience:
                        print('Early stopping triggered.')
                        break
            else:
                print('-' * 89)
                print(f'| end of epoch {epoch} | time: {(time.time() - epoch_start_time):.2f}s | '
                      f'train loss {train_loss:.4f}')
                print('-' * 89)

            # Save the model at the end of each epoch
            self._save_checkpoint(epoch)

            if self.scheduler:
                self.scheduler.step()

    def _train_epoch(self, epoch: int, log_interval: int) -> float:
        """
        Trains the model for one epoch.

        Args:
            epoch: Current epoch number.
            log_interval: Steps between logging training status.

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        start_time = time.time()

        for batch_idx, (input_ids, target_ids) in enumerate(self.train_loader, 1):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(input_ids)

            # Flatten the outputs and targets for loss computation
            output_flat = output.view(-1, output.size(-1))
            target_flat = target_ids.view(-1)

            loss = self.criterion(output_flat, target_flat)
            loss.backward()

            if self.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % log_interval == 0:
                elapsed = time.time() - start_time
                current_loss = total_loss / log_interval
                print(f'| epoch {epoch} | {batch_idx}/{len(self.train_loader)} batches | '
                      f'lr {self.optimizer.param_groups[0]["lr"]:.6f} | '
                      f'ms/batch {(elapsed * 1000 / log_interval):.2f} | '
                      f'loss {current_loss:.4f}')
                total_loss = 0.0
                start_time = time.time()

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def _evaluate(self) -> float:
        """
        Evaluates the model on the validation set.

        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for input_ids, target_ids in self.val_loader:
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)

                output = self.model(input_ids)

                output_flat = output.view(-1, output.size(-1))
                target_flat = target_ids.view(-1)

                loss = self.criterion(output_flat, target_flat)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        return avg_loss

    def _save_checkpoint(self, epoch: int, best: bool = False):
        """
        Saves the model checkpoint.

        Args:
            epoch: Current epoch number.
            best: Whether this is the best model so far.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        filename = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, filename)
        if best:
            best_filename = os.path.join(self.save_dir, 'best_model.pt')
            copy.copyfile(filename, best_filename)
            print(f'Best model saved to {best_filename}.')

def get_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0
) -> Optimizer:
    """
    Sets up the optimizer.

    Args:
        model: The model to optimize.
        learning_rate: Learning rate for the optimizer.
        weight_decay: Weight decay (L2 regularization).

    Returns:
        optimizer: An instance of torch.optim.Optimizer
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    return optimizer

def get_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = 'StepLR',
    **kwargs
) -> _LRScheduler:
    """
    Sets up the learning rate scheduler.

    Args:
        optimizer: Optimizer for which to schedule the learning rate.
        scheduler_type: Type of scheduler ('StepLR', 'ExponentialLR', 'ReduceLROnPlateau', etc.).
        **kwargs: Additional arguments for the scheduler.

    Returns:
        scheduler: An instance of torch.optim.lr_scheduler._LRScheduler
    """
    if scheduler_type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif scheduler_type == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
    elif scheduler_type == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    else:
        raise ValueError(f'Unsupported scheduler type: {scheduler_type}')
    return scheduler

def get_criterion(
    criterion_type: str = 'CrossEntropyLoss',
    **kwargs
) -> nn.Module:
    """
    Sets up the loss function.

    Args:
        criterion_type: Type of loss function ('CrossEntropyLoss', 'NLLLoss', etc.).
        **kwargs: Additional arguments for the loss function.

    Returns:
        criterion: An instance of nn.Module representing the loss function.
    """
    if criterion_type == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss(**kwargs)
    elif criterion_type == 'NLLLoss':
        criterion = nn.NLLLoss(**kwargs)
    else:
        raise ValueError(f'Unsupported criterion type: {criterion_type}')
    return criterion
