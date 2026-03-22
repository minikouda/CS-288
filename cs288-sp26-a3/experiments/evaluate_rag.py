import json
import os
import sys
import re
import string
import csv
from datetime import datetime
import numpy as np

# Ensure project root is in path
sys.path.append(os.getcwd())

CSV_LOG = "experiments/eval_results.csv"

from src.retrieval.retrieve import Retriever
from src.generation.generator import Generator
from tqdm import tqdm
from collections import Counter

def normalize_answer(s):
    """Lowercases, removes punctuation and articles."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def run_evaluation(reference_path, index_dir="models/retrieval", k=10):
    retriever = Retriever(index_dir)
    generator = Generator(model="meta-llama/llama-3.1-8b-instruct")

    with open(reference_path, 'r', encoding='utf-8') as f:
        references = [json.loads(line) for line in f]

    em_scores = []
    f1_scores = []
    retrieval_recalls = []

    results = []

    print(f"Evaluating {len(references)} samples...")
    for ref in tqdm(references):
        query = ref['question']
        ground_truths = [gt.strip() for gt in ref['answer'].split('|')]
        gt_url = ref.get('url', None)

        # Retrieve
        context_chunks = retriever.retrieve(query, k=k)

        # Calculate Retrieval Recall
        # 1. Check if GT URL is in any retrieved chunk metadata
        # 2. Check if any GT answer string is inside any retrieved chunk content
        found_in_retrieval = False
        gt_url_chunk_file = None
        content_match_chunk_files = []
        for chunk in context_chunks:
            chunk_file = chunk['metadata'].get('file', None)
            # URL check
            if gt_url and gt_url.strip() == chunk['metadata'].get('url', '').strip():
                gt_url_chunk_file = chunk_file
            # Content check
            for gt in ground_truths:
                if normalize_answer(gt) in normalize_answer(chunk['content']):
                    if chunk_file is not None and chunk_file not in content_match_chunk_files:
                        content_match_chunk_files.append(chunk_file)
                    break

        found_in_retrieval = (gt_url_chunk_file is not None) or (len(content_match_chunk_files) > 0)

        retrieval_recalls.append(1.0 if found_in_retrieval else 0.0)

        # Generate
        try:
            prediction = generator.generate(query, context_chunks)
        except Exception as e:
            print(f"Error for '{query}': {e}")
            prediction = "I don't know"

        # Calculate Metrics
        em = metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        f1 = metric_max_over_ground_truths(f1_score, prediction, ground_truths)

        em_scores.append(em)
        f1_scores.append(f1)

        results.append({
            "question": query,
            "ground_truths": ground_truths,
            "prediction": prediction,
            "em": em,
            "f1": f1,
            "retrieval_recall": found_in_retrieval,
            "gt_url_chunk_file": gt_url_chunk_file,
            "content_match_chunk_files": content_match_chunk_files,
            "retrieved_files": [c['metadata']['file'] for c in context_chunks]
        })

    em = float(np.mean(em_scores))
    f1 = float(np.mean(f1_scores))
    rr = float(np.mean(retrieval_recalls))

    print("\n" + "="*30)
    print(f"RAG Performance (k={k}):")
    print(f"Exact Match:      {em*100:.2f}%")
    print(f"F1 Score:         {f1*100:.2f}%")
    print(f"Retrieval Recall: {rr*100:.2f}%")
    print("="*30)

    # Save detailed results
    with open("experiments/last_eval_results.json", 'w', encoding='utf-8') as f:
        json.dump({"overall_em": em, "overall_f1": f1, "details": results}, f, indent=2)

    return {"reference": reference_path, "index": index_dir, "k": k,
            "n_samples": len(references), "em": em, "f1": f1, "retrieval_recall": rr}

def append_to_csv(rows):
    """Append one or more result dicts to the CSV log."""
    fieldnames = ["timestamp", "reference", "index", "k", "n_samples", "em", "f1", "retrieval_recall"]
    file_exists = os.path.isfile(CSV_LOG)
    with open(CSV_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for row in rows:
            writer.writerow({
                "timestamp": ts,
                "reference": row["reference"],
                "index": row["index"],
                "k": row["k"],
                "n_samples": row["n_samples"],
                "em": f"{row['em']*100:.2f}",
                "f1": f"{row['f1']*100:.2f}",
                "retrieval_recall": f"{row['retrieval_recall']*100:.2f}",
            })
    print(f"\nResults appended to {CSV_LOG}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="data/hidden_dev.jsonl", help="Path to first reference JSONL file")
    parser.add_argument("--reference2", default=None, help="Path to second reference JSONL file (optional)")
    parser.add_argument("--index", default="models/retrieval", help="Path to retrieval index directory")
    parser.add_argument("--k", type=int, default=10, help="Number of chunks to retrieve")
    args = parser.parse_args()

    rows = []

    if args.reference2:
        # Run both and report combined totals
        with open(args.reference) as f:
            refs1 = [json.loads(l) for l in f]
        with open(args.reference2) as f:
            refs2 = [json.loads(l) for l in f]

        retriever = Retriever(args.index)
        generator = Generator(model="meta-llama/llama-3.1-8b-instruct")

        all_em, all_f1, all_rr = [], [], []

        for label, refs in [(args.reference, refs1), (args.reference2, refs2)]:
            em_scores, f1_scores, retrieval_recalls, results = [], [], [], []
            print(f"\nEvaluating {len(refs)} samples from {label}...")
            for ref in tqdm(refs):
                query = ref['question']
                ground_truths = [gt.strip() for gt in ref['answer'].split('|')]
                gt_url = ref.get('url', None)
                context_chunks = retriever.retrieve(query, k=args.k)
                found_in_retrieval = False
                gt_url_chunk_file = None
                content_match_chunk_files = []
                for chunk in context_chunks:
                    chunk_file = chunk['metadata'].get('file', None)
                    if gt_url and gt_url.strip() == chunk['metadata'].get('url', '').strip():
                        gt_url_chunk_file = chunk_file
                    for gt in ground_truths:
                        if normalize_answer(gt) in normalize_answer(chunk['content']):
                            if chunk_file is not None and chunk_file not in content_match_chunk_files:
                                content_match_chunk_files.append(chunk_file)
                            break
                found_in_retrieval = (gt_url_chunk_file is not None) or (len(content_match_chunk_files) > 0)
                retrieval_recalls.append(1.0 if found_in_retrieval else 0.0)
                try:
                    prediction = generator.generate(query, context_chunks)
                except Exception as e:
                    print(f"Error for '{query}': {e}")
                    prediction = "I don't know"
                em = metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
                f1 = metric_max_over_ground_truths(f1_score, prediction, ground_truths)
                em_scores.append(em)
                f1_scores.append(f1)
                results.append({
                    "question": query, "ground_truths": ground_truths, "prediction": prediction,
                    "em": em, "f1": f1, "retrieval_recall": found_in_retrieval,
                    "gt_url_chunk_file": gt_url_chunk_file, "content_match_chunk_files": content_match_chunk_files,
                    "retrieved_files": [c['metadata']['file'] for c in context_chunks]
                })
            em_val = float(np.mean(em_scores))
            f1_val = float(np.mean(f1_scores))
            rr_val = float(np.mean(retrieval_recalls))
            print("\n" + "="*30)
            print(f"RAG Performance on {label} (k={args.k}):")
            print(f"Exact Match:      {em_val*100:.2f}%")
            print(f"F1 Score:         {f1_val*100:.2f}%")
            print(f"Retrieval Recall: {rr_val*100:.2f}%")
            print("="*30)
            rows.append({"reference": label, "index": args.index, "k": args.k,
                         "n_samples": len(refs), "em": em_val, "f1": f1_val, "retrieval_recall": rr_val})
            all_em.extend(em_scores)
            all_f1.extend(f1_scores)
            all_rr.extend(retrieval_recalls)

        combined_em = float(np.mean(all_em))
        combined_f1 = float(np.mean(all_f1))
        combined_rr = float(np.mean(all_rr))
        print("\n" + "="*30)
        print(f"COMBINED ({len(all_em)} samples):")
        print(f"Exact Match:      {combined_em*100:.2f}%")
        print(f"F1 Score:         {combined_f1*100:.2f}%")
        print(f"Retrieval Recall: {combined_rr*100:.2f}%")
        print("="*30)
        rows.append({"reference": "COMBINED", "index": args.index, "k": args.k,
                     "n_samples": len(all_em), "em": combined_em, "f1": combined_f1, "retrieval_recall": combined_rr})
    else:
        result = run_evaluation(args.reference, index_dir=args.index, k=args.k)
        rows.append(result)

    append_to_csv(rows)
