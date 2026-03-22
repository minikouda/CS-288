import json
import os
import sys
import csv
import re
import string
import time
import argparse
import numpy as np
from collections import Counter
from tqdm import tqdm

# Ensure project root is in path
sys.path.append(os.getcwd())

from src.retrieval.retrieve import Retriever
from src.generation.generator import Generator

def normalize_answer(s):
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
    if num_same == 0: return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def evaluate_config(retriever, generator_model, questions, k):
    generator = Generator(model=generator_model)
    em_scores = []
    f1_scores = []
    times = []
    recalls = []
    details = []
    
    for ref in questions:
        query = ref['question']
        ground_truths = [gt.strip() for gt in ref['answer'].split('|')]
        gt_url = ref.get('url', None)
        
        start_time = time.time()
        context_chunks = retriever.retrieve(query, k=k)
        try:
            prediction = generator.generate(query, context_chunks)
        except:
            prediction = "I don't know"
        end_time = time.time()
        times.append(end_time - start_time)
        
        # Recall check
        found_in_retrieval = False
        for chunk in context_chunks:
            if gt_url and gt_url.strip() == chunk['metadata'].get('url', '').strip():
                found_in_retrieval = True
                break
            for gt in ground_truths:
                if normalize_answer(gt) in normalize_answer(chunk['content']):
                    found_in_retrieval = True
                    break
            if found_in_retrieval: break
        
        recalls.append(1.0 if found_in_retrieval else 0.0)
        em = max([exact_match_score(prediction, gt) for gt in ground_truths])
        f1 = max([f1_score(prediction, gt) for gt in ground_truths])
        
        em_scores.append(em)
        f1_scores.append(f1)
        details.append({
            "question": query,
            "ground_truths": ground_truths,
            "prediction": prediction,
            "em": bool(em),
            "f1": float(f1),
            "recall": found_in_retrieval
        })
        
    return np.mean(em_scores), np.mean(f1_scores), np.mean(recalls), np.sum(times), np.mean(times), details

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record_details", action="store_true", help="Save detailed QA results to JSON")
    args = parser.parse_args()

    reference_path = "data/reference.jsonl"
    with open(reference_path, 'r', encoding='utf-8') as f:
        questions = [json.loads(line) for line in f]
    
    retriever = Retriever("models/retrieval")
    
    # Models to test
    models = [
        "meta-llama/llama-3.1-8b-instruct",
        # "qwen/qwen-2.5-7b-instruct"
    ]
    
    # K values to test
    k_values = list(range(8,13))
    
    output_file = "experiments/model_selection_results.csv"
    all_details = {}

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['model', 'k', 'em', 'f1', 'recall', 'total_time', 'avg_time_per_q']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for model_name in models:
            for k in k_values:
                print(f"Testing Model: {model_name}, K: {k}")
                em, f1, recall, total_time, avg_time, run_details = evaluate_config(retriever, model_name, questions, k)
                print(f"Result - EM: {em:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}, Time: {total_time:.2f}s")
                
                writer.writerow({
                    'model': model_name, 
                    'k': k, 
                    'em': em, 
                    'f1': f1,
                    'recall': recall,
                    'total_time': total_time,
                    'avg_time_per_q': avg_time
                })
                csvfile.flush()
                
                if args.record_details:
                    run_id = f"{model_name.replace('/', '_')}_k{k}"
                    all_details[run_id] = run_details

    if args.record_details:
        with open("experiments/model_selection_details.json", 'w', encoding='utf-8') as f:
            json.dump(all_details, f, indent=2)
        print("Detailed results saved to experiments/model_selection_details.json")

if __name__ == "__main__":
    main()
