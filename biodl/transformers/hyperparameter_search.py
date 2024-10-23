import os
import random
import logging
import torch
import numpy as np
from typing import Dict, Any, List
from models import TransformerModel
from data import (
    load_fasta, clean_sequence, build_tokenizer, get_dataloaders
)
from training import Trainer, get_optimizer, get_scheduler, get_criterion
from evaluation import evaluate_model
from utils import (
    set_random_seed, get_device, count_parameters, Timer
)
from copy import deepcopy

class HyperparameterSearch:
    """
    Class to conduct hyperparameter search for transformer models.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        hyperparameter_space: Dict[str, Any],
        num_trials: int = 20,
        search_method: str = 'random',
        seed: int = 42
    ):
        """
        Initializes the hyperparameter search.

        Args:
            config: Base configuration dictionary.
            hyperparameter_space: Dictionary defining the hyperparameter search space.
            num_trials: Number of hyperparameter configurations to evaluate.
            search_method: Method for hyperparameter search ('random', 'grid').
            seed: Random seed for reproducibility.
        """
        self.base_config = config
        self.hyperparameter_space = hyperparameter_space
        self.num_trials = num_trials
        self.search_method = search_method
        self.seed = seed
        self.results = []

        set_random_seed(self.seed)
        self.device = get_device(use_gpu=config['training']['use_gpu'])
        self.prepare_data()

    def prepare_data(self):
        """
        Prepares the data loaders.
        """
        config = self.base_config
        # Load and clean sequences (assuming data is already downloaded and extracted)
        sequences = load_fasta(os.path.join(config['data']['data_dir'], config['data']['extracted_file']))
        allowed_tokens = set(config['data']['allowed_tokens'])
        cleaned_sequences = [clean_sequence(seq, allowed_tokens) for seq in sequences]
        min_seq_length = config['data']['min_seq_length']
        cleaned_sequences = [seq for seq in cleaned_sequences if len(seq) >= min_seq_length]
        tokens = sorted(list(allowed_tokens))
        self.tokenizer = build_tokenizer(tokens)

        # Use full dataset for hyperparameter search
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            sequences=cleaned_sequences,
            tokenizer=self.tokenizer,
            batch_size=config['training']['batch_size'],
            max_seq_length=config['model']['max_seq_length'],
            val_split=config['data']['val_split'],
            test_split=config['data']['test_split'],
            shuffle=True,
            num_workers=config['data']['num_workers']
        )

    def sample_hyperparameters(self) -> Dict[str, Any]:
        """
        Samples a random hyperparameter configuration.

        Returns:
            A dictionary containing hyperparameter values.
        """
        hyperparams = {}
        for key, value in self.hyperparameter_space.items():
            if isinstance(value, list):
                hyperparams[key] = random.choice(value)
            elif isinstance(value, dict) and 'type' in value:
                if value['type'] == 'int':
                    hyperparams[key] = random.randint(value['min'], value['max'])
                elif value['type'] == 'float':
                    hyperparams[key] = random.uniform(value['min'], value['max'])
                elif value['type'] == 'categorical':
                    hyperparams[key] = random.choice(value['choices'])
            else:
                hyperparams[key] = value  # Fixed value
        return hyperparams

    def run_trial(self, trial_num: int, hyperparams: Dict[str, Any]):
        """
        Runs a single trial with the given hyperparameters.

        Args:
            trial_num: The trial number.
            hyperparams: Dictionary of hyperparameters for this trial.
        """
        logging.info(f"Starting trial {trial_num} with hyperparameters: {hyperparams}")

        # Update configuration with sampled hyperparameters
        config = deepcopy(self.base_config)
        config['model'].update({
            'num_layers': hyperparams['num_layers'],
            'embedding_dim': hyperparams['embedding_dim'],
            'num_heads': hyperparams['num_heads'],
            'hidden_dim': hyperparams['hidden_dim'],
            'dropout': hyperparams['dropout']
        })
        config['training'].update({
            'learning_rate': hyperparams['learning_rate'],
            'optimizer_type': hyperparams['optimizer_type'],
            'weight_decay': hyperparams['weight_decay'],
            'batch_size': hyperparams['batch_size'],
            'max_grad_norm': hyperparams.get('max_grad_norm', None)
        })
        config['training']['scheduler_type'] = hyperparams['scheduler_type']
        config['training']['scheduler_params'] = hyperparams['scheduler_params']
        config['training']['criterion_type'] = hyperparams['criterion_type']

        # Update data loaders with new batch size
        self.train_loader.batch_size = hyperparams['batch_size']
        self.val_loader.batch_size = hyperparams['batch_size']
        self.test_loader.batch_size = hyperparams['batch_size']

        # Initialize the model
        model = TransformerModel(
            vocab_size=len(self.tokenizer),
            num_layers=hyperparams['num_layers'],
            embedding_dim=hyperparams['embedding_dim'],
            num_heads=hyperparams['num_heads'],
            hidden_dim=hyperparams['hidden_dim'],
            max_seq_length=config['model']['max_seq_length'],
            dropout=hyperparams['dropout']
        ).to(self.device)

        # Count parameters
        n_params = count_parameters(model)

        # Set up optimizer, scheduler, and criterion
        optimizer = get_optimizer(
            model,
            learning_rate=hyperparams['learning_rate'],
            optimizer_type=hyperparams['optimizer_type'],
            weight_decay=hyperparams['weight_decay'],
            momentum=hyperparams.get('momentum', 0.0),
            betas=hyperparams.get('betas', (0.9, 0.999))
        )
        scheduler = get_scheduler(
            optimizer,
            scheduler_type=hyperparams['scheduler_type'],
            **hyperparams['scheduler_params']
        )
        criterion = get_criterion(criterion_type=hyperparams['criterion_type'])

        # Create Trainer instance
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            scheduler=scheduler,
            device=self.device,
            save_dir=self.base_config['training']['save_dir'],
            max_grad_norm=hyperparams.get('max_grad_norm', None)
        )

        # Training loop
        num_epochs = self.base_config['training']['num_epochs']
        log_interval = self.base_config['training']['log_interval']

        # Train the model
        with Timer(f'Trial {trial_num}'):
            trainer.train(
                num_epochs=num_epochs,
                log_interval=log_interval
            )

        # Evaluate on validation data
        val_loss, val_perplexity, val_accuracy = evaluate_model(
            model=model,
            data_loader=self.val_loader,
            criterion=criterion,
            device=self.device
        )
        logging.info(f'Trial {trial_num} results - Val Loss: {val_loss:.4f}, Perplexity: {val_perplexity:.4f}, Accuracy: {val_accuracy:.4f}')

        # Store the results
        trial_result = {
            'trial_num': trial_num,
            'n_params': n_params,
            'val_loss': val_loss,
            'val_perplexity': val_perplexity,
            'val_accuracy': val_accuracy,
            'hyperparameters': hyperparams
        }
        self.results.append(trial_result)

        # Save intermediate results
        self.save_results()

    def save_results(self):
        """
        Saves the results to a file.
        """
        results_file = os.path.join(self.base_config['training']['save_dir'], 'hyperparameter_search_results.npy')
        np.save(results_file, self.results)
        logging.info(f'Results saved to {results_file}')

    def run(self):
        """
        Runs the hyperparameter search.
        """
        for trial_num in range(1, self.num_trials + 1):
            hyperparams = self.sample_hyperparameters()
            self.run_trial(trial_num, hyperparams)

        # After all trials, save the final results
        self.save_results()
        logging.info('Hyperparameter search completed.')
