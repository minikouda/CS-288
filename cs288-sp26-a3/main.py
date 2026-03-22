import sys
import os
from src.retrieval.retrieve import Retriever
from src.generation.generator import Generator
from tqdm import tqdm

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 main.py <questions_txt_path> <predictions_out_path>")
        sys.exit(1)
    
    questions_path = sys.argv[1]
    predictions_path = sys.argv[2]
    
    # Initialize RAG components
    retriever = Retriever("models/retrieval")
    generator = Generator(model="meta-llama/llama-3.1-8b-instruct")
    
    # Read questions
    with open(questions_path, 'r', encoding='utf-8') as f:
        questions = [line.strip() for line in f]
    
    predictions = []
    
    print(f"Processing {len(questions)} questions...")
    for query in tqdm(questions):
        if not query:
            predictions.append("I don't know")
            continue
            
        # Retrieve
        try:
            context_chunks = retriever.retrieve(query, k=20)
            # Generate
            prediction = generator.generate(query, context_chunks)
        except Exception as e:
            print(f"Error processing query '{query}': {e}")
            prediction = "I don't know" # Fallback
            
        # Ensure no newlines in prediction
        prediction = prediction.replace("\n", " ").strip()
        predictions.append(prediction)
    
    # Write predictions
    with open(predictions_path, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    
    print(f"Predictions saved to {predictions_path}")

if __name__ == "__main__":
    main()
