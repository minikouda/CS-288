"""
Experiment with different MLP configurations: activation functions, optimizers, learning rates.

Usage examples:
    # Test different activations
    python experiment_mlp.py --activation relu --optimizer adam --lr 0.001
    python experiment_mlp.py --activation tanh --optimizer adam --lr 0.001
    python experiment_mlp.py --activation sigmoid --optimizer adam --lr 0.001
    
    # Test different optimizers
    python experiment_mlp.py --activation relu --optimizer sgd --lr 0.01
    python experiment_mlp.py --activation relu --optimizer adagrad --lr 0.01
    python experiment_mlp.py --activation relu --optimizer adam --lr 0.001
    
    # Grid search
    python experiment_mlp.py --grid_search
"""

import argparse
import csv
import os
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from multilayer_perceptron_sst import (
    BOWDataset,
    Tokenizer,
    get_label_mappings,
)
from utils import DataPoint, DataType, accuracy, load_data, save_results


class ConfigurableMLP(nn.Module):
    """MLP with configurable activation functions."""
    
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        padding_index: int,
        embed_dim: int = 256,
        hidden_dims: List[int] = [256, 128, 64],
        activation: str = "relu",
        dropout: float = 0.2,
    ):
        super().__init__()
        self.padding_index = padding_index
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_index)
        
        # Build layers
        layers = []
        prev_dim = embed_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Add activation
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "sigmoid":
                layers.append(nn.Sigmoid())
            elif activation == "mixed":
                # Mix activations across layers
                if i % 3 == 0:
                    layers.append(nn.Tanh())
                elif i % 3 == 1:
                    layers.append(nn.ReLU())
                else:
                    layers.append(nn.Sigmoid())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer (no activation - CrossEntropyLoss applies softmax internally)
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, input_features_b_l: torch.Tensor, input_length_b: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_features_b_l)
        
        # Better pooling: average + max
        avg_pooled = embedded.sum(dim=1) / (input_length_b.unsqueeze(1).float() + 1e-9)
        max_pooled, _ = embedded.max(dim=1)
        pooled = (avg_pooled + max_pooled) / 2
        
        output = self.network(pooled)
        return output


class Trainer:
    def __init__(self, model: nn.Module, device: torch.device):
        self.device = device
        self.model = model.to(device)
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> float:
        """Train one epoch and return average loss."""
        self.model.train()
        total_loss = 0
        
        for inputs_b_l, lengths_b, labels_b in dataloader:
            inputs_b_l = inputs_b_l.to(self.device)
            lengths_b = lengths_b.to(self.device)
            labels_b = labels_b.to(self.device)
            
            logits = self.model(inputs_b_l, lengths_b)
            loss = loss_fn(logits, labels_b)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate and return accuracy."""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs_b_l, lengths_b, labels_b in dataloader:
                inputs_b_l = inputs_b_l.to(self.device)
                lengths_b = lengths_b.to(self.device)
                
                logits = self.model(inputs_b_l, lengths_b)
                preds = torch.argmax(logits, dim=1)
                
                all_predictions.extend(preds.cpu().tolist())
                all_targets.extend(labels_b.tolist())
        
        return accuracy(all_predictions, all_targets)


def run_experiment(
    dataset_name: str,
    activation: str,
    optimizer_name: str,
    lr: float,
    epochs: int,
    batch_size: int,
    embed_dim: int,
    hidden_dims: List[int],
    dropout: float,
    device: torch.device,
) -> Dict:
    """Run a single experiment with given configuration."""
    
    print(f"\n{'='*80}")
    print(f"Experiment: {activation.upper()} + {optimizer_name.upper()} (lr={lr})")
    print(f"{'='*80}")
    
    # Load data
    data_type = DataType(dataset_name)
    train_data, val_data, dev_data, test_data = load_data(data_type)
    
    tokenizer = Tokenizer(train_data, max_vocab_size=20000)
    label2id, id2label = get_label_mappings(train_data)
    
    # Create datasets
    train_ds = BOWDataset(train_data, tokenizer, label2id, max_length=100)
    val_ds = BOWDataset(val_data, tokenizer, label2id, max_length=100)
    dev_ds = BOWDataset(dev_data, tokenizer, label2id, max_length=100)
    
    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = ConfigurableMLP(
        vocab_size=len(tokenizer.token2id),
        num_classes=len(label2id),
        padding_index=tokenizer.TOK_PADDING_INDEX,
        embed_dim=embed_dim,
        hidden_dims=hidden_dims,
        activation=activation,
        dropout=dropout,
    )
    
    trainer = Trainer(model, device)
    
    # Create optimizer
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    elif optimizer_name == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    loss_fn = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_acc = 0
    val_accs = []
    
    for epoch in range(epochs):
        train_loss = trainer.train_epoch(train_loader, optimizer, loss_fn)
        val_acc = trainer.evaluate(val_loader)
        val_accs.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Val Acc: {val_acc*100:.2f}%")
    
    # Final evaluation on dev set
    dev_acc = trainer.evaluate(dev_loader)
    print(f"\nFinal Dev Accuracy: {dev_acc*100:.2f}%")
    
    return {
        "dataset": dataset_name,
        "activation": activation,
        "optimizer": optimizer_name,
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "embed_dim": embed_dim,
        "hidden_dims": "-".join(map(str, hidden_dims)),
        "dropout": dropout,
        "best_val_acc": best_val_acc,
        "final_dev_acc": dev_acc,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def save_experiment_result(result: Dict, filename: str = "experiment_results.csv"):
    """Save experiment result to CSV."""
    file_exists = os.path.exists(filename)
    
    with open(filename, "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(result)
    
    print(f"\n✓ Results saved to {filename}")


def grid_search(dataset_name: str, device: torch.device):
    """Run grid search over key hyperparameters."""
    
    print("\n" + "="*80)
    print("GRID SEARCH: Testing activation functions and optimizers")
    print("="*80)
    
    # Grid search parameters
    activations = ["relu", "tanh", "sigmoid"]
    optimizers = ["sgd", "adam", "adagrad"]
    learning_rates = {
        "sgd": [0.1, 0.01],
        "adam": [0.001, 0.0001],
        "adagrad": [0.01, 0.001],
    }
    
    results = []
    total_experiments = sum(len(learning_rates[opt]) for opt in optimizers) * len(activations)
    current = 0
    
    for activation in activations:
        for optimizer_name in optimizers:
            for lr in learning_rates[optimizer_name]:
                current += 1
                print(f"\n[{current}/{total_experiments}] Running experiment...")
                
                result = run_experiment(
                    dataset_name=dataset_name,
                    activation=activation,
                    optimizer_name=optimizer_name,
                    lr=lr,
                    epochs=10,
                    batch_size=64,
                    embed_dim=256,
                    hidden_dims=[256, 128, 64],
                    dropout=0.2,
                    device=device,
                )
                
                results.append(result)
                save_experiment_result(result)
    
    # Print summary
    print("\n" + "="*80)
    print("GRID SEARCH SUMMARY")
    print("="*80)
    
    results.sort(key=lambda x: x["final_dev_acc"], reverse=True)
    
    print(f"\n{'Rank':<5} {'Activation':<10} {'Optimizer':<10} {'LR':<8} {'Dev Acc':<10}")
    print("-"*80)
    
    for i, result in enumerate(results[:10], 1):
        print(f"{i:<5} {result['activation']:<10} {result['optimizer']:<10} "
              f"{result['learning_rate']:<8.4f} {result['final_dev_acc']*100:<10.2f}%")
    
    print(f"\nBest configuration:")
    best = results[0]
    print(f"  Activation: {best['activation']}")
    print(f"  Optimizer: {best['optimizer']}")
    print(f"  Learning Rate: {best['learning_rate']}")
    print(f"  Dev Accuracy: {best['final_dev_acc']*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLP activation/optimizer experiments")
    
    parser.add_argument("--dataset", type=str, default="sst2", choices=["sst2", "newsgroups"])
    parser.add_argument("--activation", type=str, default="relu", 
                       choices=["relu", "tanh", "sigmoid", "mixed"])
    parser.add_argument("--optimizer", type=str, default="adam",
                       choices=["sgd", "adam", "adamw", "adagrad"])
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 128, 64])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--grid_search", action="store_true",
                       help="Run grid search over activations and optimizers")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    if args.grid_search:
        grid_search(args.dataset, device)
    else:
        result = run_experiment(
            dataset_name=args.dataset,
            activation=args.activation,
            optimizer_name=args.optimizer,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            embed_dim=args.embed_dim,
            hidden_dims=args.hidden_dims,
            dropout=args.dropout,
            device=device,
        )
        save_experiment_result(result)
