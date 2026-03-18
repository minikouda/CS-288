import json
import os
import sys
import re
import string
import numpy as np

# Ensure project root is in path
sys.path.append(os.getcwd())

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

def run_evaluation(reference_path, k=10):
    retriever = Retriever("models/retrieval")
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

    print("\n" + "="*30)
    print(f"RAG Performance (k={k}):")
    print(f"Exact Match:      {np.mean(em_scores)*100:.2f}%")
    print(f"F1 Score:         {np.mean(f1_scores)*100:.2f}%")
    print(f"Retrieval Recall: {np.mean(retrieval_recalls)*100:.2f}%")
    print("="*30)

    
    # Save detailed results
    with open("experiments/last_eval_results.json", 'w', encoding='utf-8') as f:
        json.dump({
            "overall_em": float(np.mean(em_scores)),
            "overall_f1": float(np.mean(f1_scores)),
            "details": results
        }, f, indent=2)

if __name__ == "__main__":
    run_evaluation("data/reference.jsonl", k=3)
