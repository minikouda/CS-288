from src.llm import call_llm

class Generator:
    def __init__(self, model="meta-llama/llama-3.1-8b-instruct"):
        self.model = model

    def generate(self, query, context_chunks):
        context_text = "\n\n".join([f"Source {i+1}:\n{chunk['content']}" for i, chunk in enumerate(context_chunks)])

        system_prompt = (
            "You are an extractive Berkeley EECS QA assistant. "
            "Use only the provided context. Do not use outside knowledge. "
            "Output only the final answer — no explanation, no trailing punctuation, no filler words. "
            "Copy the shortest exact span from context that answers the question. "
            "Never prepend qualifiers like 'At least', 'up to', 'or better', or units like 'per week' unless they are part of the core answer span. "
            "For names, output the name only — no titles such as 'Professor' or 'Dr.'. "
            "For room numbers or locations, copy the full string as it appears in context. "
            "For counting questions (how many), count the relevant items in context and output only the number. "
            "For dates and times, copy the exact format from context. "
            "For yes/no questions, output exactly 'Yes' or 'No'. "
            "When multiple candidates are present, pick the single one most directly stated as the answer in context. "
            "If the answer is not supported by context, output exactly 'I don't know'."
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
