import os
import torch
import logging
import itertools
import numpy as np
from models import TransformerModel
from data import (
    download_dataset, extract_gzip, load_fasta, clean_sequence,
    build_tokenizer, get_dataloaders
)
from evaluation import evaluate_model, fit_scaling_law, plot_scaling_law
from utils import (
    setup_logging, load_config, set_random_seed, get_device,
    count_parameters, Timer
)

from biodl.transformers.training import Trainer, get_optimizer, get_scheduler, get_criterion


def train():
    # Load configuration
    config = load_config('config.yaml')

    # Set up logging
    setup_logging(log_file=config['logging']['log_file'], log_level=logging.INFO)

    # Set random seed
    set_random_seed(config['training']['seed'])

    # Get device
    device = get_device(use_gpu=config['training']['use_gpu'])

    # Data preparation
    # Create data directory if it doesn't exist
    os.makedirs(config['data']['data_dir'], exist_ok=True)

    # Define dataset URL and file paths
    dataset_url = config['data']['dataset_url']
    compressed_file = os.path.join(config['data']['data_dir'], config['data']['compressed_file'])
    extracted_file = os.path.join(config['data']['data_dir'], config['data']['extracted_file'])

    # Download and extract dataset
    download_dataset(dataset_url, compressed_file)
    extract_gzip(compressed_file, extracted_file)

    # Load sequences from the extracted FASTA file
    sequences = load_fasta(extracted_file)

    # Define allowed tokens (e.g., amino acids for protein sequences)
    allowed_tokens = set(config['data']['allowed_tokens'])

    # Clean sequences by removing invalid characters
    cleaned_sequences = [clean_sequence(seq, allowed_tokens) for seq in sequences]

    # Filter sequences by minimum length
    min_seq_length = config['data']['min_seq_length']
    cleaned_sequences = [seq for seq in cleaned_sequences if len(seq) >= min_seq_length]

    # Log the number of sequences after cleaning and filtering
    logging.info(f'Number of sequences after cleaning and filtering: {len(cleaned_sequences)}')

    # Build tokenizer
    tokens = sorted(list(allowed_tokens))
    tokenizer = build_tokenizer(tokens)

    # Define model sizes and dataset sizes
    model_sizes = config['scaling']['model_sizes']
    dataset_sizes = config['scaling']['dataset_sizes']

    # Initialize results storage
    results = []

    # Loop over model sizes and dataset sizes
    for model_config in model_sizes:
        # Calculate model parameter count
        temp_model = TransformerModel(
            vocab_size=len(tokenizer),
            **model_config
        )
        n_params = count_parameters(temp_model)

        for d_fraction in dataset_sizes:
            logging.info(f'Training model with {n_params} parameters on {d_fraction*100}% of the data')

            # Sample the dataset
            num_sequences = int(len(cleaned_sequences) * d_fraction)
            sampled_sequences = cleaned_sequences[:num_sequences]

            # Create data loaders
            train_loader, val_loader, test_loader = get_dataloaders(
                sequences=sampled_sequences,
                tokenizer=tokenizer,
                batch_size=batch_size,
                max_seq_length=max_seq_length,
                val_split=val_split,
                test_split=test_split,
                shuffle=True,
                num_workers=num_workers
            )

            # Initialize the model
            model = TransformerModel(
                vocab_size=len(tokenizer),
                **model_config
            ).to(device)

            # Set up optimizer, scheduler, and criterion
            optimizer = get_optimizer(
                model,
                learning_rate=config['training']['learning_rate'],
                weight_decay=config['training']['weight_decay']
            )
            scheduler = get_scheduler(
                optimizer,
                scheduler_type=config['training']['scheduler_type'],
                **config['training']['scheduler_params']
            )
            criterion = get_criterion(criterion_type=config['training']['criterion_type'])

            # Create Trainer instance
            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                train_loader=train_loader,
                val_loader=val_loader,
                scheduler=scheduler,
                device=device,
                save_dir=config['training']['save_dir'],
                max_grad_norm=config['training'].get('max_grad_norm', None)
            )

            # Training loop
            num_epochs = config['training']['num_epochs']
            log_interval = config['training']['log_interval']
            early_stopping_patience = config['training'].get('early_stopping_patience', None)

            # Train the model
            with Timer(f'Training model with {n_params} parameters on {num_sequences} sequences'):
                trainer.train(
                    num_epochs=num_epochs,
                    log_interval=log_interval,
                    early_stopping_patience=early_stopping_patience
                )

            # Evaluate on validation data
            val_loss, _, _ = evaluate_model(
                model=model,
                data_loader=val_loader,
                criterion=criterion,
                device=device
            )

            # Store the results
            results.append({
                'n_params': n_params,
                'd_fraction': d_fraction,
                'val_loss': val_loss
            })

            # Save intermediate results
            np.save(os.path.join(config['training']['save_dir'], 'scaling_results.npy'), results)

    # After all experiments, fit scaling laws
    analyze_scaling_laws(results, config)

def analyze_scaling_laws(results, config):
    import pandas as pd

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Fit scaling laws for varying model sizes (n) at fixed dataset size
    for d_fraction in df['d_fraction'].unique():
        subset = df[df['d_fraction'] == d_fraction]
        x = subset['n_params'].values
        y = subset['val_loss'].values

        # Fit scaling law
        fit_result = fit_scaling_law(
            x_data=x,
            y_data=y,
            law_type='power_law'
        )

        # Plot scaling law
        plot_filename = os.path.join(
            config['training']['save_dir'],
            f'scaling_n_d{int(d_fraction*100)}.png'
        )
        plot_scaling_law(
            x_data=x,
            y_data=y,
            fit_result=fit_result,
            x_label='Model Size (Number of Parameters)',
            y_label='Validation Loss',
            title=f'Scaling Law (n) at {int(d_fraction*100)}% Dataset Size',
            save_path=plot_filename
        )

    # Fit scaling laws for varying dataset sizes (d) at fixed model size
    for n_params in df['n_params'].unique():
        subset = df[df['n_params'] == n_params]
        x = subset['d_fraction'].values
        y = subset['val_loss'].values

        # Fit scaling law
        fit_result = fit_scaling_law(
            x_data=x,
            y_data=y,
            law_type='power_law'
        )

        # Plot scaling law
        plot_filename = os.path.join(
            config['training']['save_dir'],
            f'scaling_d_n{n_params}.png'
        )
        plot_scaling_law(
            x_data=x,
            y_data=y,
            fit_result=fit_result,
            x_label='Dataset Fraction',
            y_label='Validation Loss',
            title=f'Scaling Law (d) at Model Size {n_params}',
            save_path=plot_filename
        )

    # Save final results
    df.to_csv(os.path.join(config['training']['save_dir'], 'scaling_results.csv'), index=False)

if __name__ == '__main__':
    train()
