# **BIODL.Transformers**

**BIODL** is a library for deep learning in biology with an emphasis on systems biology. The Transformers subpackage provides tools and utilities for building, training, and evaluating transformer models specifically tailored for biological sequence data, such as DNA, RNA, and proteins.

## **Table of Contents**

## **Table of Contents**

* [Introduction](#introduction)
* [Features](#features) 
* [Installation](#installation)
* [Quick Start](#quick-start)
* [Modules](#modules)
 * [models.py](#modelspy)
 * [data.py](#datapy)
 * [training.py](#trainingpy)
 * [evaluation.py](#evaluationpy) 
 * [utils.py](#utilspy)
 * [train.py](#trainpy)
* [Scaling Law Research Pipeline](#scaling-law-research-pipeline)
* [Hyperparameter Search](#hyperparameter-search)
* [Contributing](#contributing)
* [License](#license)

## **Introduction**

The BIODL.Transformers subpackage is designed to facilitate research and development of transformer-based models in the context of biological sequences. It provides a comprehensive framework that includes data preprocessing, model implementation, training routines, evaluation metrics, and utilities for scaling law analysis.

## **Features**

* **Flexible Transformer Models**: Easily customizable transformer architectures suitable for biological sequence prediction tasks.
* **Data Handling**: Utilities for downloading, preprocessing, and loading biological sequence datasets.
* **Training Pipeline**: Modular training loop with support for various optimizers, schedulers, and loss functions.
* **Evaluation Metrics**: Functions to compute loss, perplexity, accuracy, and perform scaling law analysis.
* **Hyperparameter Search**: Tools to perform extensive hyperparameter optimization.
* **Scaling Law Analysis**: Capabilities to investigate how model performance scales with model and dataset sizes.
* **Utilities**: Helper functions for configuration management, logging, and reproducibility.

## **Installation**

To use the BIODL.Transformers subpackage, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/biodl.git
cd biodl
pip install -r requirements.txt
```

## **Quick Start**

Here is a minimal example to train a transformer model on biological sequence data:

```python
from models import TransformerModel
from data import (
    load_fasta, clean_sequence, build_tokenizer, get_dataloaders
)
from training import Trainer, get_optimizer, get_scheduler, get_criterion
from utils import (
    setup_logging, set_random_seed, get_device, count_parameters
)

# Set up
setup_logging(log_file='training.log')
set_random_seed(42)
device = get_device(use_gpu=True)

# Data preparation
sequences = load_fasta('data/sequences.fasta')
allowed_tokens = set('ACDEFGHIKLMNPQRSTVWY')
cleaned_sequences = [clean_sequence(seq, allowed_tokens) for seq in sequences]
tokenizer = build_tokenizer(sorted(list(allowed_tokens)))
train_loader, val_loader, test_loader = get_dataloaders(
    sequences=cleaned_sequences,
    tokenizer=tokenizer,
    batch_size=32,
    max_seq_length=1024,
    val_split=0.1,
    test_split=0.1,
    shuffle=True
)

# Model initialization
vocab_size = len(tokenizer)
model = TransformerModel(
    vocab_size=vocab_size,
    embedding_dim=256,
    num_layers=4,
    num_heads=8,
    hidden_dim=512,
    max_seq_length=1024,
    dropout=0.1
).to(device)

# Training setup
optimizer = get_optimizer(model, learning_rate=1e-4)
scheduler = get_scheduler(optimizer, scheduler_type='StepLR', step_size=10, gamma=0.1)
criterion = get_criterion(criterion_type='CrossEntropyLoss')

# Training
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    train_loader=train_loader,
    val_loader=val_loader,
    scheduler=scheduler,
    device=device
)
trainer.train(num_epochs=50)
```

## **Modules**

### **models.py**

Defines the transformer models used for biological sequences.

* **Classes**:
  * BaseModel: An abstract base class for all models.
  * PositionalEncoding: Implements positional encoding as described in "Attention is All You Need".
  * TransformerModel: A standard transformer model suitable for sequence prediction.

### **data.py**

Handles data downloading, preprocessing, and loading.

* **Functions**:
  * download_dataset: Downloads datasets from specified URLs.
  * extract_gzip: Extracts gzip-compressed files.
  * load_fasta: Loads sequences from FASTA files.
  * clean_sequence: Cleans sequences by removing invalid characters.
  * build_tokenizer: Creates a mapping from tokens to indices.
  * get_dataloaders: Creates DataLoader objects for training, validation, and testing.

### **training.py**

Manages the training process.

* **Classes**:
  * Trainer: Encapsulates the training and evaluation loops.
* **Functions**:
  * get_optimizer: Sets up the optimizer.
  * get_scheduler: Sets up the learning rate scheduler.
  * get_criterion: Defines the loss function.

### **evaluation.py**

Provides evaluation metrics and scaling law analysis tools.

* **Functions**:
  * evaluate_model: Evaluates the model on a dataset.
  * plot_training_curves: Plots training and validation loss curves.
  * fit_scaling_law: Fits scaling laws to performance data.
  * plot_scaling_law: Visualizes scaling law fits.

### **utils.py**

Contains utility functions for configuration, logging, and more.

* **Functions**:
  * load_config: Loads configuration from a YAML file.
  * save_config: Saves configuration to a YAML file.
  * setup_logging: Configures logging.
  * set_random_seed: Sets the random seed for reproducibility.
  * get_device: Determines whether to use CPU or GPU.
  * count_parameters: Counts the number of trainable parameters in a model.
  * save_model: Saves a model's state dictionary.
  * load_model: Loads a model's state dictionary.
* **Classes**:
  * Timer: A context manager for timing code execution.

### **train.py**

The main script to run training and evaluation.

* **Functions**:
  * train: Orchestrates the entire training pipeline, including data preparation, model training, evaluation, and scaling law analysis.

## **Scaling Law Research Pipeline**

The subpackage includes a pipeline to conduct scaling law research:

* **Objective**: Investigate how model performance scales with model size (number of parameters) and dataset size.
* **Implementation**:
  * Vary model hyperparameters to create models of different sizes.
  * Subsample datasets to create different dataset sizes.
  * Collect validation loss and other metrics for each model-dataset combination.
  * Fit scaling laws to the collected data using the fit_scaling_law function.
  * Visualize the results with plot_scaling_law.

## **Hyperparameter Search**

An additional module, hyperparameter_search.py, is provided to perform hyperparameter optimization.

* **Features**:
  * Define a comprehensive hyperparameter space, including model architecture, optimization settings, and regularization techniques.
  * Implement random search strategy for efficient exploration of the hyperparameter space.
  * Integrate seamlessly with the existing training pipeline.
  * Log and save results for analysis.
* **Usage**:
  * Configure the hyperparameter space and search parameters.
  * Run the hyperparameter search script to evaluate multiple configurations.
  * Analyze the results to identify the best hyperparameter settings.

## **Contributing**

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes with clear messages.
4. Submit a pull request detailing your changes.

Please ensure that your code adheres to the project's coding standards and includes appropriate tests.

## **License**

This project is licensed under the MIT License. See the LICENSE file for details.

## **Contact**

For questions or suggestions, please open an issue on the GitHub repository or contact the maintainer directly at kendiukhov@gmail.com.
