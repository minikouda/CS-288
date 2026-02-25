#!/usr/bin/env python3
"""
Part 4 Training Script (Colab-optimized)

Pipeline:
1. Download TinyStories + SQuAD (or use cached)
2. Train BPE tokenizer on TinyStories
3. Pretrain TransformerLM on TinyStories
4. Fine-tune on SQuAD (multiple-choice) → finetuned_predictions.json
5. Few-shot prompting on fine-tuned backbone → prompting_predictions.json

Usage (Colab):
    !python part4/train.py --config medium   # ~30 min on T4
    !python part4/train.py --config large    # ~60 min on A100
    !python part4/train.py --config quick    # ~5 min, smoke test
"""

import argparse
import json
import os
import sys
import random
import time
from pathlib import Path

import torch
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from part1.train_bpe import train_bpe
from part1.tokenizer import get_tokenizer
from part2.model import TransformerLM

from part4.datasets import create_pretraining_dataloader, create_qa_dataloader
from part4.sampling import generate_text
from part4.qa_model import TransformerForMultipleChoice, evaluate_qa_model
from part4.trainer import Trainer, TrainingConfig, create_qa_loss_fn

FIXTURES = Path(__file__).parent / "fixtures"
OUTPUTS = Path(__file__).parent / "outputs"
OUTPUTS.mkdir(exist_ok=True)
FIXTURES.mkdir(exist_ok=True)

SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>"]

# =============================================================================
# Configs
# =============================================================================

CONFIGS = {
    "quick": {
        "pretrain_data": ROOT / "part1/fixtures/tinystories_sample_5M.txt",
        "qa_train": FIXTURES / "squad_train.json",
        "qa_dev":   FIXTURES / "squad_dev.json",
        "qa_test":  FIXTURES / "squad_test.json",
        "vocab_size": 1024,
        "d_model": 256,
        "num_layers": 4,
        "num_heads": 4,
        "d_ff": 1024,
        "context_length": 256,
        "pretrain_epochs": 2,
        "finetune_epochs": 3,
        "batch_size": 32,
        "lr": 1e-3,
        "few_shot_k": 2,
    },
    "medium": {
        # ~50M params, ~30 min on T4
        "pretrain_data": FIXTURES / "tinystories_100k.txt",
        "qa_train": FIXTURES / "squad_train.json",
        "qa_dev":   FIXTURES / "squad_dev.json",
        "qa_test":  FIXTURES / "squad_test.json",
        "vocab_size": 8192,
        "d_model": 512,
        "num_layers": 8,
        "num_heads": 8,
        "d_ff": 2048,
        "context_length": 512,
        "pretrain_epochs": 3,
        "finetune_epochs": 10,
        "batch_size": 16,
        "lr": 3e-4,
        "few_shot_k": 3,
    },
    "large": {
        # ~120M params — fits T4 16GB; ~60-90 min
        "pretrain_data": FIXTURES / "tinystories_100k.txt",
        "qa_train": FIXTURES / "squad_train.json",
        "qa_dev":   FIXTURES / "squad_dev.json",
        "qa_test":  FIXTURES / "squad_test.json",
        "vocab_size": 8192,
        "d_model": 768,
        "num_layers": 12,
        "num_heads": 12,
        "d_ff": 3072,
        "context_length": 512,
        "pretrain_epochs": 3,
        "finetune_epochs": 10,
        "batch_size": 8,   # safe for T4 16GB
        "lr": 1e-4,
        "few_shot_k": 3,
    },
    "h100": {
        # ~300M params optimised for H100 (96GB VRAM) — ~45 min
        # BPE trains on 100k subset (fast); LM pretrains on full 2.1M stories
        "bpe_data": FIXTURES / "tinystories_100k.txt",   # BPE only — fast
        "pretrain_data": FIXTURES / "tinystories_full.txt",  # full corpus for LM
        "qa_train": FIXTURES / "squad_train.json",
        "qa_dev":   FIXTURES / "squad_dev.json",
        "qa_test":  FIXTURES / "squad_test.json",
        "vocab_size": 16384,
        "d_model": 1024,
        "num_layers": 16,
        "num_heads": 16,
        "d_ff": 4096,
        "context_length": 512,
        "pretrain_epochs": 3,
        "finetune_epochs": 15,
        "batch_size": 64,
        "lr": 2e-4,
        "few_shot_k": 4,
    },
}


# =============================================================================
# Step 0: Download datasets
# =============================================================================

def download_datasets(config: dict):
    """Download TinyStories and SQuAD if not already present."""
    need_tinystories = not config["pretrain_data"].exists()
    need_squad = not config["qa_train"].exists()

    if not need_tinystories and not need_squad:
        print("Datasets already present, skipping download.")
        return

    # part4/datasets.py shadows HuggingFace 'datasets' — use importlib to load by file path
    import importlib.util, importlib
    _part4_dir = str(Path(__file__).parent)

    # Remove stale cached module if it points to our local datasets.py
    if "datasets" in sys.modules:
        cached = sys.modules["datasets"]
        cached_file = getattr(cached, "__file__", "") or ""
        if "part4" in cached_file or not hasattr(cached, "load_dataset"):
            del sys.modules["datasets"]

    # Temporarily pull part4 out of path so HuggingFace datasets is found
    _had_part4 = _part4_dir in sys.path
    if _had_part4:
        sys.path.remove(_part4_dir)
    try:
        import datasets as hf_datasets
        if not hasattr(hf_datasets, "load_dataset"):
            raise ImportError
    except ImportError:
        os.system("pip install -q datasets")
        if "datasets" in sys.modules:
            del sys.modules["datasets"]
        import datasets as hf_datasets
    finally:
        if _had_part4 and _part4_dir not in sys.path:
            sys.path.insert(0, _part4_dir)

    if need_tinystories:
        print("Downloading TinyStories...")
        ds = hf_datasets.load_dataset("roneneldan/TinyStories", split="train")
        full_path = FIXTURES / "tinystories_full.txt"
        subset_path = FIXTURES / "tinystories_100k.txt"
        with open(full_path, "w", encoding="utf-8") as ff, \
             open(subset_path, "w", encoding="utf-8") as sf:
            for i, ex in enumerate(ds):
                story = ex["text"].strip() + "\n<|endoftext|>\n"
                ff.write(story)
                if i < 100_000:
                    sf.write(story)
                if (i + 1) % 100_000 == 0:
                    print(f"  {i+1:,} stories written...")
        print(f"TinyStories saved ({full_path})")

    if need_squad:
        print("Downloading SQuAD...")
        import random as _rnd
        _rnd.seed(42)

        def squad_to_mc(split_ds, n=None):
            ctx_answers = {}
            for ex in split_ds:
                key = ex["context"][:80]
                ctx_answers.setdefault(key, []).append(ex["answers"]["text"][0])
            all_answers = list({ex["answers"]["text"][0] for ex in split_ds if ex["answers"]["text"]})
            out = []
            for i, ex in enumerate(split_ds):
                if n and i >= n:
                    break
                if not ex["answers"]["text"]:
                    continue
                correct = ex["answers"]["text"][0]
                key = ex["context"][:80]
                same_ctx = [a for a in ctx_answers.get(key, []) if a != correct]
                other = [a for a in all_answers if a != correct and a not in same_ctx]
                distractors = []
                if same_ctx:
                    distractors += _rnd.sample(same_ctx, min(1, len(same_ctx)))
                need = 3 - len(distractors)
                if need and other:
                    distractors += _rnd.sample(other, min(need, len(other)))
                generics = ["Unknown", "Not mentioned", "Cannot determine"]
                while len(distractors) < 3:
                    distractors.append(_rnd.choice(generics))
                choices = [correct] + distractors[:3]
                idx_perm = list(range(4))
                _rnd.shuffle(idx_perm)
                choices = [choices[j] for j in idx_perm]
                answer_idx = idx_perm.index(0)
                out.append({
                    "context": ex["context"],
                    "question": ex["question"],
                    "choices": choices,
                    "answer": answer_idx,
                    "id": ex["id"],
                })
            return out

        train_ds = hf_datasets.load_dataset("squad", split="train")
        val_ds   = hf_datasets.load_dataset("squad", split="validation")
        train_mc = squad_to_mc(train_ds, n=10_000)
        dev_mc   = squad_to_mc(val_ds,   n=2_000)
        test_mc  = squad_to_mc(val_ds,   n=1_000)
        for path, data in [
            (FIXTURES / "squad_train.json", train_mc),
            (FIXTURES / "squad_dev.json",   dev_mc),
            (FIXTURES / "squad_test.json",  test_mc),
        ]:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        print(f"SQuAD saved (train={len(train_mc)}, dev={len(dev_mc)}, test={len(test_mc)})")


# =============================================================================
# Step 1: Train BPE tokenizer
# =============================================================================

def train_tokenizer(pretrain_data: Path, vocab_size: int):
    print("\n" + "=" * 60)
    print("STEP 1: Training BPE Tokenizer")
    print("=" * 60)
    vocab, merges = train_bpe(
        input_path=pretrain_data,
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer = get_tokenizer(vocab, merges, SPECIAL_TOKENS)
    test = "Once upon a time, there was a little girl."
    print(f"Vocab size: {len(vocab)}  |  Merges: {len(merges)}")
    print(f"Test: '{test}' → {len(tokenizer.encode(test))} tokens")
    return tokenizer, vocab, merges


# =============================================================================
# Step 2: Pretrain LM
# =============================================================================

def pretrain_lm(tokenizer, config: dict, device: str, checkpoint_path: Path = None) -> TransformerLM:
    print("\n" + "=" * 60)
    print("STEP 2: Pretraining Language Model")
    print("=" * 60)

    model = TransformerLM(
        vocab_size=len(tokenizer.vocab),
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Resume from checkpoint if available
    if checkpoint_path and checkpoint_path.exists():
        print(f"Loading pretrain checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        return model

    dataloader = create_pretraining_dataloader(
        file_path=config["pretrain_data"],
        tokenizer=tokenizer,
        batch_size=config["batch_size"],
        max_length=config["context_length"],
        stride=config["context_length"] // 2,
        shuffle=True,
    )
    print(f"Data: {config['pretrain_data']} | sequences: {len(dataloader.dataset)} | batches/epoch: {len(dataloader)}")

    train_config = TrainingConfig(
        num_epochs=config["pretrain_epochs"],
        learning_rate=config["lr"],
        weight_decay=0.01,
        warmup_steps=min(200, len(dataloader) // 5),
        max_grad_norm=1.0,
        device=device,
        log_interval=max(1, len(dataloader) // 10),
    )
    trainer = Trainer(model=model, config=train_config, train_dataloader=dataloader)

    t0 = time.time()
    results = trainer.train()
    print(f"Pretrain done in {(time.time()-t0)/60:.1f} min | final loss: {results['train_losses'][-1]:.4f}")

    # Sample generation
    for prompt in ["Once upon a time", "The little dog"]:
        text = generate_text(model, tokenizer, prompt, max_new_tokens=40, method="greedy")
        print(f"  '{prompt}' → '{text[:100]}'")

    if checkpoint_path:
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    return model


# =============================================================================
# Step 3: Fine-tune for QA
# =============================================================================

def finetune_qa(
    pretrained_model: TransformerLM,
    tokenizer,
    config: dict,
    device: str,
    checkpoint_path: Path = None,
) -> TransformerForMultipleChoice:
    print("\n" + "=" * 60)
    print("STEP 3: Fine-tuning for Multiple-Choice QA")
    print("=" * 60)

    qa_model = TransformerForMultipleChoice(
        transformer_lm=pretrained_model,
        hidden_size=pretrained_model.d_model,
        num_choices=4,
        pooling="last",
        freeze_backbone=False,
    ).to(device)

    if checkpoint_path and checkpoint_path.exists():
        print(f"Loading finetune checkpoint: {checkpoint_path}")
        qa_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        return qa_model

    with open(config["qa_train"]) as f:
        train_data = json.load(f)

    train_loader = create_qa_dataloader(
        data=train_data,
        tokenizer=tokenizer,
        batch_size=config["batch_size"],
        max_length=config["context_length"],
        num_choices=4,
        shuffle=True,
    )

    print(f"Train examples: {len(train_data)} | batches/epoch: {len(train_loader)}")

    train_config = TrainingConfig(
        num_epochs=config["finetune_epochs"],
        learning_rate=config["lr"] / 3,
        weight_decay=0.01,
        warmup_steps=min(100, len(train_loader) // 5),
        max_grad_norm=1.0,
        device=device,
        log_interval=max(1, len(train_loader) // 5),
    )
    trainer = Trainer(
        model=qa_model,
        config=train_config,
        train_dataloader=train_loader,
        compute_loss_fn=create_qa_loss_fn(device),
    )

    t0 = time.time()
    results = trainer.train()
    print(f"Finetune done in {(time.time()-t0)/60:.1f} min | final loss: {results['train_losses'][-1]:.4f}")

    if checkpoint_path:
        torch.save(qa_model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    return qa_model


# =============================================================================
# Step 4: Generate finetuned predictions
# =============================================================================

def get_finetuned_predictions(qa_model, tokenizer, config: dict, device: str) -> dict:
    print("\n" + "=" * 60)
    print("STEP 4: Evaluating Fine-tuned Model")
    print("=" * 60)

    with open(config["qa_dev"]) as f:
        dev_data = json.load(f)
    with open(config["qa_test"]) as f:
        test_data = json.load(f)

    def eval_split(data, label):
        loader = create_qa_dataloader(
            data=data, tokenizer=tokenizer,
            batch_size=config["batch_size"],
            max_length=config["context_length"],
            num_choices=4, shuffle=False,
        )
        results = evaluate_qa_model(qa_model, loader, device)
        print(f"  {label} accuracy: {results['accuracy']:.2%}")
        return results

    dev_results  = eval_split(dev_data,  "Dev")
    test_results = eval_split(test_data, "Test")

    return {
        "dev": dev_results,
        "test": test_results,
    }


# =============================================================================
# Step 5: Few-shot prompting
# =============================================================================

class FewShotPromptTemplate:
    """
    Few-shot prompt that prepends k gold examples before the test question.
    The model scores by comparing next-token logits for 'A','B','C','D'.
    """

    TEMPLATE = (
        "The following are multiple choice questions with answers.\n\n"
        "{few_shot_examples}"
        "Context: {context}\n"
        "Question: {question}\n"
        "{choices_formatted}\n"
        "Answer:"
    )

    EXAMPLE_TEMPLATE = (
        "Context: {context}\n"
        "Question: {question}\n"
        "{choices_formatted}\n"
        "Answer: {answer_letter}\n\n"
    )

    def __init__(self, examples: list, k: int = 3):
        # Randomly sample k examples to use as few-shot demonstrations
        self.k = k
        self.examples = examples

    def _fmt_choices(self, choices):
        labels = ["A", "B", "C", "D"]
        return "\n".join(f"{l}. {c}" for l, c in zip(labels, choices))

    def format(self, context: str, question: str, choices: list, exclude_ids: set = None) -> str:
        pool = [e for e in self.examples if e.get("id") not in (exclude_ids or set())]
        shots = random.sample(pool, min(self.k, len(pool)))
        few_shot_str = ""
        for s in shots:
            lbl = chr(ord("A") + s["answer"])
            few_shot_str += self.EXAMPLE_TEMPLATE.format(
                context=s["context"][:300],  # truncate long contexts
                question=s["question"],
                choices_formatted=self._fmt_choices(s["choices"]),
                answer_letter=lbl,
            )
        prompt = self.TEMPLATE.format(
            few_shot_examples=few_shot_str,
            context=context[:300],
            question=question,
            choices_formatted=self._fmt_choices(choices),
        )
        return prompt


def get_choice_token_ids(tokenizer):
    """Map A/B/C/D to a single token ID each."""
    choice_tokens = {}
    for label in ["A", "B", "C", "D"]:
        for prefix in [" ", ""]:
            ids = tokenizer.encode(prefix + label)
            if ids:
                choice_tokens[label] = ids[-1]
                break
    return choice_tokens


@torch.no_grad()
def predict_with_prompting(
    model: TransformerLM,
    tokenizer,
    examples: list,
    template: FewShotPromptTemplate,
    choice_tokens: dict,
    device: str,
    context_length: int,
) -> list:
    model.eval()
    predictions = []
    labels_order = ["A", "B", "C", "D"]

    for ex in examples:
        exclude = {ex.get("id")}
        prompt = template.format(
            context=ex["context"],
            question=ex["question"],
            choices=ex["choices"],
            exclude_ids=exclude,
        )
        token_ids = tokenizer.encode(prompt)
        # Truncate to fit context window (keep the end = the actual question)
        if len(token_ids) > context_length:
            token_ids = token_ids[-context_length:]
        input_ids = torch.tensor([token_ids], device=device)
        logits = model(input_ids)[:, -1, :]  # (1, vocab_size)
        choice_logits = torch.tensor(
            [logits[0, choice_tokens.get(lbl, 0)].item() for lbl in labels_order],
            device=device,
        )
        pred = choice_logits.argmax().item()
        predictions.append(pred)

    return predictions


def evaluate_prompting_pipeline(
    model: TransformerLM,
    tokenizer,
    train_data: list,
    eval_data: list,
    config: dict,
    device: str,
) -> dict:
    print("\n" + "=" * 60)
    print("STEP 5: Few-shot Prompting Evaluation")
    print("=" * 60)

    random.seed(42)
    template = FewShotPromptTemplate(examples=train_data, k=config["few_shot_k"])
    choice_tokens = get_choice_token_ids(tokenizer)
    print(f"Choice token IDs: {choice_tokens}")
    print(f"Few-shot k={config['few_shot_k']}, eval examples={len(eval_data)}")

    predictions = predict_with_prompting(
        model, tokenizer, eval_data, template, choice_tokens, device,
        config["context_length"],
    )
    correct = sum(
        1 for p, ex in zip(predictions, eval_data)
        if ex.get("answer", -1) >= 0 and p == ex["answer"]
    )
    total = sum(1 for ex in eval_data if ex.get("answer", -1) >= 0)
    accuracy = correct / total if total > 0 else 0.0
    print(f"Few-shot prompting accuracy: {accuracy:.2%}")
    return {"accuracy": accuracy, "predictions": predictions}


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["quick", "medium", "large", "h100"], default="large")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-pretrain", action="store_true", help="Load pretrain checkpoint")
    parser.add_argument("--skip-finetune", action="store_true", help="Load finetune checkpoint")
    parser.add_argument("--no-download", action="store_true", help="Skip dataset download")
    args = parser.parse_args()

    config = CONFIGS[args.config]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(f"CS288 Part 4 Training | config={args.config} | device={device}")
    print("=" * 60)

    # ── 0. Download datasets ──────────────────────────────────────────────────
    if not args.no_download:
        download_datasets(config)

    assert config["pretrain_data"].exists(), f"Missing: {config['pretrain_data']}"
    assert config["qa_train"].exists(),      f"Missing: {config['qa_train']}"

    # ── 1. Tokenizer ──────────────────────────────────────────────────────────
    # Use bpe_data if specified (smaller subset = much faster BPE training)
    bpe_data = config.get("bpe_data", config["pretrain_data"])
    tokenizer, _, _ = train_tokenizer(bpe_data, config["vocab_size"])

    # ── 2. Pretrain ───────────────────────────────────────────────────────────
    pretrain_ckpt = OUTPUTS / f"pretrain_{args.config}.pt"
    pretrained_model = pretrain_lm(
        tokenizer, config, device,
        checkpoint_path=pretrain_ckpt if args.skip_pretrain else None,
    )
    # Always save after training
    torch.save(pretrained_model.state_dict(), pretrain_ckpt)

    # ── 3. Fine-tune ──────────────────────────────────────────────────────────
    finetune_ckpt = OUTPUTS / f"finetune_{args.config}.pt"
    qa_model = finetune_qa(
        pretrained_model, tokenizer, config, device,
        checkpoint_path=finetune_ckpt if args.skip_finetune else None,
    )
    torch.save(qa_model.state_dict(), finetune_ckpt)

    # ── 4. Finetuned predictions ──────────────────────────────────────────────
    ft_results = get_finetuned_predictions(qa_model, tokenizer, config, device)

    finetuned_out = {
        "predictions": ft_results["test"]["predictions"],
        "accuracy":    ft_results["test"]["accuracy"],
        "dev_accuracy": ft_results["dev"]["accuracy"],
        "config": args.config,
    }
    ft_path = OUTPUTS / "finetuned_predictions.json"
    with open(ft_path, "w") as f:
        json.dump(finetuned_out, f, indent=2)
    print(f"\nSaved: {ft_path}")

    # ── 5. Few-shot prompting ─────────────────────────────────────────────────
    with open(config["qa_train"]) as f:
        train_data = json.load(f)
    with open(config["qa_test"]) as f:
        test_data = json.load(f)
    with open(config["qa_dev"]) as f:
        dev_data = json.load(f)

    # Use the fine-tuned backbone (without classification head) for prompting
    backbone = qa_model.transformer

    # Evaluate on dev first
    print("\nFew-shot on dev set:")
    dev_prompt_results = evaluate_prompting_pipeline(
        backbone, tokenizer, train_data, dev_data, config, device
    )
    # Evaluate on test
    print("\nFew-shot on test set:")
    test_prompt_results = evaluate_prompting_pipeline(
        backbone, tokenizer, train_data, test_data, config, device
    )

    prompting_out = {
        "predictions": test_prompt_results["predictions"],
        "accuracy":    test_prompt_results["accuracy"],
        "dev_accuracy": dev_prompt_results["accuracy"],
        "config": args.config,
        "few_shot_k": config["few_shot_k"],
    }
    pt_path = OUTPUTS / "prompting_predictions.json"
    with open(pt_path, "w") as f:
        json.dump(prompting_out, f, indent=2)
    print(f"Saved: {pt_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    ft_acc  = ft_results["test"]["accuracy"]
    pt_acc  = test_prompt_results["accuracy"]
    boost   = pt_acc - ft_acc

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Config:              {args.config}")
    print(f"Fine-tuned accuracy: {ft_acc:.2%}  (30%=0pts → 50%=12pts)")
    print(f"Prompting accuracy:  {pt_acc:.2%}  (need +2% over fine-tuned)")
    print(f"Prompting boost:     {boost:+.2%}  (need +4% for full prompting score)")
    print(f"Random baseline:     25.00%")

    ft_score = max(0.0, min(1.0, (ft_acc - 0.30) / 0.20))
    pt_score = max(0.0, min(1.0, boost / 0.04)) if boost > 0 else 0.0
    print(f"\nEstimated scores → Fine-tune: {ft_score:.0%} | Prompting: {pt_score:.0%}")
    print(f"\nPrediction files saved to {OUTPUTS}/")


if __name__ == "__main__":
    main()
