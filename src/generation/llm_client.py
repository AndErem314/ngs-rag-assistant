import ollama
from typing import List, Dict

class OllamaGenerator:
    """Client for generating answers using a local Ollama LLM."""

    def __init__(self, model: str = "llama3.1:8b", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host
        # The ollama client uses environment variable or default; we keep host for reference.

    def generate(self, prompt: str) -> str:
        """Generate a completion for the given prompt."""
        try:
            response = ollama.generate(model=self.model, prompt=prompt)
            return response["response"]
        except Exception as e:
            print(f"Error generating: {e}")
            return ""

    def answer_question(self, question: str, context: str, metadata_list: List[Dict]) -> str:
        """Generate an answer with citations using the provided context."""
        # Build a simple citation list
        citations = []
        for meta in metadata_list:
            citations.append(f"{meta['source']}, page {meta['page']}")
        citation_text = "\n".join(set(citations))  # unique citations

        prompt = f"""You are an expert assistant for NGS sample preparation. Use only the following context to answer the question. If the answer is not in the context, say "I cannot find that information in the manuals." Cite the source file and page number when referencing specific details.

Context:
{context}

Question: {question}

Answer:
"""
        return self.generate(prompt)