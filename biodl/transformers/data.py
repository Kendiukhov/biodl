import os
import requests
import gzip
import shutil
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
import random

class SequenceDataset(Dataset):
    """
    PyTorch Dataset for biological sequences.
    """
    def __init__(
        self,
        sequences: List[str],
        tokenizer: Dict[str, int],
        max_seq_length: int = 1024
    ):
        """
        Initializes the dataset with sequences and tokenizer.

        Args:
            sequences: List of biological sequences (DNA/RNA/proteins).
            tokenizer: Dictionary mapping tokens to indices.
            max_seq_length: Maximum sequence length. Sequences longer than this are truncated.
        """
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        token_ids = [self.tokenizer.get(token, self.tokenizer['<UNK>']) for token in seq]
        # Truncate or pad sequences
        if len(token_ids) > self.max_seq_length:
            token_ids = token_ids[:self.max_seq_length]
        else:
            token_ids += [self.tokenizer['<PAD>']] * (self.max_seq_length - len(token_ids))
        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(token_ids[1:], dtype=torch.long)
        return input_ids, target_ids

def download_dataset(url: str, save_path: str):
    """
    Downloads a dataset from a URL.

    Args:
        url: URL to download the dataset from.
        save_path: Local path to save the downloaded dataset.
    """
    if not os.path.exists(save_path):
        print(f"Downloading dataset from {url}...")
        response = requests.get(url, stream=True)
        with open(save_path, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        print(f"Dataset downloaded and saved to {save_path}.")
    else:
        print(f"Dataset already exists at {save_path}.")

def extract_gzip(file_path: str, extract_to: str):
    """
    Extracts a gzip-compressed file.

    Args:
        file_path: Path to the gzip file.
        extract_to: Path to save the extracted file.
    """
    if not os.path.exists(extract_to):
        print(f"Extracting {file_path}...")
        with gzip.open(file_path, 'rb') as f_in:
            with open(extract_to, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"File extracted to {extract_to}.")
    else:
        print(f"Extracted file already exists at {extract_to}.")

def load_fasta(file_path: str) -> List[str]:
    """
    Loads sequences from a FASTA file.

    Args:
        file_path: Path to the FASTA file.

    Returns:
        List of sequences as strings.
    """
    sequences = []
    with open(file_path, 'r') as f:
        seq = ''
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if seq:
                    sequences.append(seq)
                    seq = ''
            else:
                seq += line.upper()
        if seq:
            sequences.append(seq)
    return sequences

def clean_sequence(seq: str, allowed_tokens: set) -> str:
    """
    Cleans a sequence by removing invalid characters.

    Args:
        seq: The sequence string.
        allowed_tokens: Set of valid token characters.

    Returns:
        Cleaned sequence string.
    """
    return ''.join([token for token in seq if token in allowed_tokens])

def build_tokenizer(tokens: List[str]) -> Dict[str, int]:
    """
    Builds a tokenizer mapping tokens to indices.

    Args:
        tokens: List of unique tokens.

    Returns:
        A dictionary mapping tokens to indices.
    """
    tokenizer = {token: idx + 2 for idx, token in enumerate(tokens)}  # Reserve 0 and 1 for PAD and UNK
    tokenizer['<PAD>'] = 0
    tokenizer['<UNK>'] = 1
    return tokenizer

def get_dataloaders(
    sequences: List[str],
    tokenizer: Dict[str, int],
    batch_size: int = 32,
    max_seq_length: int = 1024,
    val_split: float = 0.1,
    test_split: float = 0.1,
    shuffle: bool = True,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates DataLoader objects for training, validation, and testing.

    Args:
        sequences: List of sequences.
        tokenizer: Tokenizer dictionary.
        batch_size: Batch size.
        max_seq_length: Maximum sequence length.
        val_split: Fraction of data to use for validation.
        test_split: Fraction of data to use for testing.
        shuffle: Whether to shuffle the data.
        num_workers: Number of subprocesses for data loading.

    Returns:
        Tuple of DataLoaders: (train_loader, val_loader, test_loader)
    """
    total_size = len(sequences)
    indices = list(range(total_size))
    if shuffle:
        random.shuffle(indices)

    test_size = int(test_split * total_size)
    val_size = int(val_split * total_size)
    train_size = total_size - test_size - val_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_sequences = [sequences[i] for i in train_indices]
    val_sequences = [sequences[i] for i in val_indices]
    test_sequences = [sequences[i] for i in test_indices]

    train_dataset = SequenceDataset(train_sequences, tokenizer, max_seq_length)
    val_dataset = SequenceDataset(val_sequences, tokenizer, max_seq_length)
    test_dataset = SequenceDataset(test_sequences, tokenizer, max_seq_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
