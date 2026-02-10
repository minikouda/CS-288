#!/usr/bin/env python3
"""
Helper script to log experiment results.

Usage:
    python log_experiment.py --model mlp --dataset sst2 --epochs 30 --lr 0.001 --accuracy 69.23 --notes "baseline model"
"""

import argparse
import csv
from datetime import datetime

def log_result(model, dataset, features, epochs, lr, batch_size, dropout, embed_dim, hidden_dims, activation, accuracy, notes=""):
    """Log experiment result to CSV"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    row = [
        model,
        dataset,
        features,
        epochs,
        lr,
        batch_size,
        dropout,
        embed_dim,
        hidden_dims,
        activation,
        f"{accuracy:.2f}",
        notes,
        timestamp
    ]
    
    with open("experiment_results.csv", "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)
    
    print(f"✓ Logged: {model} on {dataset} - Dev Acc: {accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Log experiment results")
    parser.add_argument("--model", type=str, required=True, help="Model type (mlp, perceptron)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset (sst2, newsgroups)")
    parser.add_argument("--features", type=str, default="N/A", help="Features used (for perceptron)")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--hidden_dims", type=str, default="256-128-64", help="Hidden layer dimensions")
    parser.add_argument("--activation", type=str, default="tanh-relu-relu", help="Activation functions")
    parser.add_argument("--accuracy", type=float, required=True, help="Dev accuracy (%)")
    parser.add_argument("--notes", type=str, default="", help="Additional notes")
    
    args = parser.parse_args()
    
    log_result(
        args.model,
        args.dataset,
        args.features,
        args.epochs,
        args.lr,
        args.batch_size,
        args.dropout,
        args.embed_dim,
        args.hidden_dims,
        args.activation,
        args.accuracy,
        args.notes
    )
