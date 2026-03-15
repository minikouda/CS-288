from src.llm import call_llm

class Generator:
    def __init__(self, model="meta-llama/llama-3.1-8b-instruct"):
        self.model = model

    def generate(self, query, context_chunks):
        context_text = "\n\n".join([f"Source {i+1}:\n{chunk['content']}" for i, chunk in enumerate(context_chunks)])

        system_prompt = (
            "You are a Berkeley EECS assistant. Answer based ONLY on context. "
            "Keep answers extremely short (1-5 words). "
            "If not in context, say 'I don't know'."
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
            max_tokens=128,
            temperature=0.0
        )
        return response
