from src.llm import call_llm

class Generator:
    def __init__(self, model="meta-llama/llama-3.1-8b-instruct"):
        self.model = model

    def generate(self, query, context_chunks):
        context_text = "\n\n".join([f"Source {i+1}:\n{chunk['content']}" for i, chunk in enumerate(context_chunks)])

        system_prompt = (
            "You are an extractive Berkeley EECS QA assistant. "
            "Use only the provided context. Do not use outside knowledge. "
            "Output ONLY the answer — absolutely no explanation, no sentence fragments, no filler words, no trailing punctuation. "
            "BREVITY IS CRITICAL: your answer must be the shortest possible span that answers the question. "
            "If the answer is a number, output only the number. "
            "If the answer is a name, output only the name — no titles such as 'Professor' or 'Dr.'. "
            "If the answer is yes or no, output exactly 'Yes' or 'No'. "
            "If the answer is a date or year, copy the exact format from context. "
            "If the answer is an email, copy only the email address itself. "
            "If the answer is a phrase, copy the minimal span — never include surrounding words like 'your very own', 'at least', 'up to', or any prepended qualifiers. "
            "For counting questions (how many), count and output only the integer. "
            "For room numbers or locations, copy the full string as it appears in context. "
            "When multiple candidates are present, pick the single one most directly stated as the answer. "
            "If the answer cannot be found in the context, try your best to infer it from context clues before giving up. "
            "Only output exactly 'I don't know' if there is truly no relevant information in the context."
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
            max_tokens=64,
            temperature=0
        )
        return response
