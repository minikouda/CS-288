"""
Benchmark GPU training and inference speed with different batch sizes.

Measures wall-time seconds to process 1,000 examples with and without batching.
Reports average time and standard deviation across multiple runs.
"""

import argparse
import time
from typing import List, Tuple
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from multilayer_perceptron_sst import (
    MultilayerPerceptronModel,
    BOWDataset,
    Tokenizer,
    get_label_mappings,
)
from utils import DataType, load_data


def benchmark_inference(
    model: nn.Module,
    dataset: BOWDataset,
    batch_size: int,
    device: torch.device,
    num_examples: int = 1000,
    num_runs: int = 5,
) -> Tuple[float, float]:
    """
    Benchmark inference speed.
    
    Returns:
        (mean_time, std_time) in seconds to process num_examples
    """
    model.eval()
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    times = []
    
    for run in range(num_runs):
        processed = 0
        start_time = time.time()
        
        with torch.no_grad():
            for inputs_b_l, lengths_b, labels_b in dataloader:
                if processed >= num_examples:
                    break
                
                inputs_b_l = inputs_b_l.to(device)
                lengths_b = lengths_b.to(device)
                
                # Forward pass
                _ = model(inputs_b_l, lengths_b)
                
                processed += inputs_b_l.size(0)
        
        elapsed = time.time() - start_time
        times.append(elapsed)
    
    return np.mean(times), np.std(times)


def benchmark_training(
    model: nn.Module,
    dataset: BOWDataset,
    batch_size: int,
    device: torch.device,
    num_examples: int = 1000,
    num_runs: int = 5,
) -> Tuple[float, float]:
    """
    Benchmark training speed (forward + backward + optimizer step).
    
    Returns:
        (mean_time, std_time) in seconds to process num_examples
    """
    times = []
    
    for run in range(num_runs):
        # Reset model for each run
        model_copy = MultilayerPerceptronModel(
            vocab_size=model.embedding.num_embeddings,
            num_classes=model.l4.out_features,
            padding_index=model.padding_index,
        ).to(device)
        model_copy.train()
        
        optimizer = torch.optim.Adam(model_copy.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        processed = 0
        start_time = time.time()
        
        for inputs_b_l, lengths_b, labels_b in dataloader:
            if processed >= num_examples:
                break
            
            inputs_b_l = inputs_b_l.to(device)
            lengths_b = lengths_b.to(device)
            labels_b = labels_b.to(device)
            
            # Forward pass
            logits = model_copy(inputs_b_l, lengths_b)
            loss = loss_fn(logits, labels_b)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            processed += inputs_b_l.size(0)
        
        elapsed = time.time() - start_time
        times.append(elapsed)
    
    return np.mean(times), np.std(times)


def run_benchmark(
    dataset_name: str = "sst2",
    num_examples: int = 1000,
    num_runs: int = 5,
    batch_sizes: List[int] = None,
):
    """Run comprehensive batching benchmark."""
    
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    
    print("="*80)
    print("GPU BATCHING BENCHMARK")
    print("="*80)
    print(f"Dataset: {dataset_name}")
    print(f"Examples per test: {num_examples}")
    print(f"Runs per batch size: {num_runs}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    data_type = DataType(dataset_name)
    train_data, val_data, dev_data, test_data = load_data(data_type)
    
    # Create tokenizer and dataset
    tokenizer = Tokenizer(train_data, max_vocab_size=20000)
    label2id, id2label = get_label_mappings(train_data)
    
    # Use dev data for benchmarking
    dataset = BOWDataset(dev_data, tokenizer, label2id, max_length=100)
    print(f"Dataset size: {len(dataset)} examples")
    
    # Create model
    model = MultilayerPerceptronModel(
        vocab_size=len(tokenizer.token2id),
        num_classes=len(label2id),
        padding_index=tokenizer.TOK_PADDING_INDEX,
    ).to(device)
    
    print("\n" + "="*80)
    print("INFERENCE BENCHMARK")
    print("="*80)
    print(f"{'Batch Size':<12} {'Mean Time (s)':<15} {'Std Dev (s)':<15} {'Speedup':<10}")
    print("-"*80)
    
    inference_results = []
    baseline_inference_time = None
    
    for batch_size in batch_sizes:
        mean_time, std_time = benchmark_inference(
            model, dataset, batch_size, device, num_examples, num_runs
        )
        inference_results.append((batch_size, mean_time, std_time))
        
        if batch_size == 1:
            baseline_inference_time = mean_time
            speedup = 1.0
        else:
            speedup = baseline_inference_time / mean_time
        
        print(f"{batch_size:<12} {mean_time:<15.4f} {std_time:<15.4f} {speedup:<10.2f}x")
    
    print("\n" + "="*80)
    print("TRAINING BENCHMARK")
    print("="*80)
    print("Note: Batch size 1 skipped for training (BatchNorm requires batch_size > 1)")
    print(f"{'Batch Size':<12} {'Mean Time (s)':<15} {'Std Dev (s)':<15} {'Speedup':<10}")
    print("-"*80)
    
    training_results = []
    baseline_training_time = None
    training_batch_sizes = [bs for bs in batch_sizes if bs > 1]
    
    for batch_size in training_batch_sizes:
        mean_time, std_time = benchmark_training(
            model, dataset, batch_size, device, num_examples, num_runs
        )
        training_results.append((batch_size, mean_time, std_time))
        
        if baseline_training_time is None:
            baseline_training_time = mean_time
            speedup = 1.0
        else:
            speedup = baseline_training_time / mean_time
        
        print(f"{batch_size:<12} {mean_time:<15.4f} {std_time:<15.4f} {speedup:<10.2f}x")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    best_inference_idx = np.argmin([r[1] for r in inference_results])
    best_training_idx = np.argmin([r[1] for r in training_results])
    
    print(f"Best inference batch size: {inference_results[best_inference_idx][0]}")
    print(f"  Time: {inference_results[best_inference_idx][1]:.4f} ± {inference_results[best_inference_idx][2]:.4f}s")
    print(f"  Speedup vs batch_size=1: {baseline_inference_time / inference_results[best_inference_idx][1]:.2f}x")
    
    print(f"\nBest training batch size: {training_results[best_training_idx][0]}")
    print(f"  Time: {training_results[best_training_idx][1]:.4f} ± {training_results[best_training_idx][2]:.4f}s")
    baseline_bs = training_results[0][0]
    print(f"  Speedup vs batch_size={baseline_bs}: {training_results[0][1] / training_results[best_training_idx][1]:.2f}x")
    
    print("\n" + "="*80)
    
    # Save results
    output_file = f"benchmark_results_{dataset_name}.txt"
    with open(output_file, 'w') as f:
        f.write("GPU BATCHING BENCHMARK RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Device: {device}\n")
        if device.type == 'cuda':
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
        f.write(f"Examples per test: {num_examples}\n")
        f.write(f"Runs per batch size: {num_runs}\n\n")
        
        f.write("INFERENCE RESULTS\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Batch Size':<12} {'Mean Time (s)':<15} {'Std Dev (s)':<15} {'Speedup':<10}\n")
        f.write("-"*80 + "\n")
        for i, (bs, mean_t, std_t) in enumerate(inference_results):
            speedup = baseline_inference_time / mean_t
            f.write(f"{bs:<12} {mean_t:<15.4f} {std_t:<15.4f} {speedup:<10.2f}x\n")
        
        f.write("\nTRAINING RESULTS\n")
        f.write("-"*80 + "\n")
        f.write("Note: Batch size 1 skipped (BatchNorm requires batch_size > 1)\n")
        f.write(f"{'Batch Size':<12} {'Mean Time (s)':<15} {'Std Dev (s)':<15} {'Speedup':<10}\n")
        f.write("-"*80 + "\n")
        baseline_bs = training_results[0][0]
        for i, (bs, mean_t, std_t) in enumerate(training_results):
            if i == 0:
                speedup = 1.0
            else:
                speedup = training_results[0][1] / mean_t
            f.write(f"{bs:<12} {mean_t:<15.4f} {std_t:<15.4f} {speedup:<10.2f}x\n")
    
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark batching speed on GPU")
    parser.add_argument(
        "--dataset",
        type=str,
        default="sst2",
        choices=["sst2", "newsgroups"],
        help="Dataset to use for benchmarking",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=1000,
        help="Number of examples to process per test",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=5,
        help="Number of runs to average over",
    )
    parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128],
        help="List of batch sizes to test",
    )
    
    args = parser.parse_args()
    
    run_benchmark(
        dataset_name=args.dataset,
        num_examples=args.num_examples,
        num_runs=args.num_runs,
        batch_sizes=args.batch_sizes,
    )
