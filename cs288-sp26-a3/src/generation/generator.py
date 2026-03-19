from src.llm import call_llm

class Generator:
    def __init__(self, model="meta-llama/llama-3.1-8b-instruct"):
        self.model = model

    def generate(self, query, context_chunks):
        context_text = "\n\n".join([f"Source {i+1}:\n{chunk['content']}" for i, chunk in enumerate(context_chunks)])

        system_prompt = (
            "You are an extractive Berkeley EECS QA assistant. "
            "Use only the provided context and do not use outside knowledge. "
            "Output only the final answer text, with no explanation. "
            "Prefer an exact span copied from context when possible. "
            "Keep the answer very short (ideally 1-6 words, never more than 10). "
            "For dates/times, preserve the exact format shown in context. "
            "For true/false questions, answer exactly 'Yes' or 'No'. "
            "If multiple candidates appear, choose the single most specific answer. "
            "If the answer is not explicitly supported by context, output exactly 'I don't know'."
        )


        prompt = (
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
        
        response = call_llm(
            query=prompt,
            system_prompt=system_prompt,
            model=self.model,
            max_tokens=16,
            temperature=0.0
        )
        return response
