import ollama
from typing import List, Dict


class OllamaGenerator:
    """
    Client for generating answers using a local Ollama LLM.

    Attributes:
        model (str): Name of the generation model (default "llama3.1:8b").
        client (ollama.Client): Ollama client bound to the specified host.
    """

    def __init__(self, model: str = "llama3.1:8b", host: str = "http://localhost:11434"):
        """
        Initialize the generator.

        Args:
            model: Name of the Ollama generation model (assumed to be already pulled).
            host: Ollama server URL. Passed directly to ollama.Client so it
                  actually takes effect (unlike relying on the OLLAMA_HOST env var).
        """
        self.model = model
        self.client = ollama.Client(host=host)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate a completion using a system + user message pair.

        Using ollama.chat() with explicit roles gives chat-style models
        (like llama3.1) much better instruction-following than a single
        plain-text prompt string.

        Args:
            system_prompt: Instructions / persona for the model.
            user_prompt: The user's question or request.

        Returns:
            The model's response as a string, or an empty string on error.
        """
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
            )
            return response["message"]["content"]
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""

    def answer_question(self, question: str, context: str, metadata_list: List[Dict]) -> str:
        """
        Generate a cited answer using the provided RAG context.

        Args:
            question:      The user's natural-language question.
            context:       Concatenated retrieved chunks from the vector store.
            metadata_list: List of dicts with "source" and "page" keys for
                           each retrieved chunk.

        Returns:
            The model's answer as a string (may include inline citations).
        """
        # Build a deduplicated citation list and include it in the prompt
        # so the model can reference specific sources in its answer.
        citations = sorted({
            f"{m['source']}, page {m['page']}"
            for m in metadata_list
        })
        citation_text = "\n".join(f"- {c}" for c in citations)

        system_prompt = (
            "You are an expert assistant for NGS sample preparation. "
            "Answer questions using ONLY the provided context. "
            "If the answer is not in the context, respond with: "
            "'I cannot find that information in the manuals.' "
            "When referencing specific details, cite the source file and page number."
        )

        user_prompt = (
            f"Context:\n{context}\n\n"
            f"Available sources:\n{citation_text}\n\n"
            f"Question: {question}"
        )

        return self.generate(system_prompt, user_prompt)


# ----------------------------------------------------------------------
# Example usage (commented out):
# if __name__ == "__main__":
#     generator = OllamaGenerator(host="http://localhost:11434")
#
#     context = "The minimum DNA input is 100 ng. See TruSight_500_Manual.pdf, page 12."
#     metadata = [{"source": "TruSight_500_Manual.pdf", "page": 12}]
#     answer = generator.answer_question(
#         question="What is the minimum DNA input?",
#         context=context,
#         metadata_list=metadata,
#     )
#     print(answer)