import re
from src.llm import call_llm

_COUNTING_RE = re.compile(r'\bhow (many|much)\b', re.IGNORECASE)

class Generator:
    def __init__(self, model="meta-llama/llama-3.1-8b-instruct"):
        self.model = model

    def generate(self, query, context_chunks):
        context_text = "\n\n".join([f"Source {i+1}:\n{chunk['content']}" for i, chunk in enumerate(context_chunks)])

        if _COUNTING_RE.search(query):
            return self._generate_counting(query, context_text)
        return self._generate_default(query, context_text)

    def _generate_default(self, query, context_text):
        system_prompt = (
            "You are an extractive Berkeley EECS QA assistant. "
            "Use only the provided context. Do not use outside knowledge. "
            "Output ONLY the answer — absolutely no explanation, no sentence fragments, no filler words, no trailing punctuation. "
            "BREVITY IS CRITICAL: your answer must be the shortest possible span that answers the question. "
            "If the answer is a number, output only the number (digits, not words). "
            "If the answer is a name, output only the name — no titles such as 'Professor' or 'Dr.'. "
            "If the answer is yes or no, output exactly 'Yes' or 'No'. "
            "If the answer is a date or year, copy the exact format from context. "
            "If the answer is an email, copy it EXACTLY as it appears in context — do not add or remove any domain parts. "
            "If the answer is a phrase, copy the minimal span — never include surrounding words like 'your very own', 'at least', 'up to', or any prepended qualifiers. "
            "For room/office numbers or locations, copy the full string as it appears in context. "
            "When asked about time durations, output in the same unit as the question (e.g., years not semesters). "
            "When multiple candidates are present, pick the single one most directly stated as the answer. "
            "IMPORTANT: You MUST attempt to answer from context. Search carefully — the answer may be embedded in a list, table, or sentence. "
            "If you find any relevant information, extract the answer even if it requires combining two facts. "
            "Only output exactly 'I don't know' if the context contains absolutely no information related to the question."
        )
        prompt = (
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
        return call_llm(
            query=prompt,
            system_prompt=system_prompt,
            model=self.model,
            max_tokens=64,
            temperature=0,
        )

    def _generate_counting(self, query, context_text):
        """Two-step chain-of-thought for 'how many/much' questions.

        Step 1: Ask the model to enumerate all matching items across all sources.
        Step 2: Extract just the final number from that enumeration.
        """
        # Step 1 — enumerate
        enum_system = (
            "You are a precise Berkeley EECS QA assistant. "
            "Use only the provided context. Do not use outside knowledge. "
            "Your job is to find and LIST every distinct item relevant to the question, "
            "one per line, using ALL sources. "
            "After the list write a line: COUNT: <integer> "
            "If the context states a total directly (e.g. '6+ units'), write that value after COUNT:. "
            "If no relevant information exists, write: COUNT: unknown"
        )
        enum_prompt = (
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n\n"
            "List each relevant item, then write COUNT: <number>"
        )
        enum_response = call_llm(
            query=enum_prompt,
            system_prompt=enum_system,
            model=self.model,
            max_tokens=256,
            temperature=0,
        )

        # Step 2 — extract the COUNT line
        count_match = re.search(r'COUNT:\s*(.+)', enum_response, re.IGNORECASE)
        if count_match:
            raw = count_match.group(1).strip()
            if raw.lower() == "unknown":
                return "I don't know"
            return raw

        # Fallback: extract last standalone number/value from response
        numbers = re.findall(r'\b[\d,]+\+?\b', enum_response)
        if numbers:
            return numbers[-1]

        return "I don't know"
