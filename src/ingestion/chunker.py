from typing import List, Tuple, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_document(
    pages: List[Tuple[int, str]],
    source_filename: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> List[Dict]:
    """
    Split a document's pages into overlapping chunks, attaching metadata.

    Args:
        pages: List of (page_num, text) for each page. Page numbers are 1-indexed.
        source_filename: The name of the source PDF file.
        chunk_size: Approximate number of tokens per chunk (character‑based).
        overlap: Number of overlapping tokens between consecutive chunks.

    Returns:
        List of dicts, each with keys:
            - "text": the chunk text
            - "metadata": {"source": source_filename, "page": page_num}
        The page number is the page where the chunk starts (approximated).
    """
    if not pages:
        return []

    # Build a single string with page markers and keep track of where each page starts
    # We'll use a special separator to later find page boundaries
    page_separator = "\n\n--- PAGE BREAK ---\n\n"
    full_text = ""
    page_boundaries = []  # list of (start_char_index, page_num)
    current_pos = 0
    for page_num, text in pages:
        # Add the page content (without separator for the first page)
        if full_text:
            full_text += page_separator
            current_pos += len(page_separator)
        full_text += text
        page_boundaries.append((current_pos, page_num))
        current_pos += len(text)

    # Create the text splitter (using character count as proxy for tokens)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(full_text)

    # For each chunk, determine which page it starts on
    # We can find the character index of the start of the chunk in full_text
    # and then find the page boundary with the greatest start index <= chunk_start.
    # A simpler approach: since we know the cumulative lengths, we can scan.
    # We'll use a helper that returns the page number for a given character position.
    def get_page_for_pos(pos: int) -> int:
        # page_boundaries are sorted by start index
        for i in range(len(page_boundaries) - 1, -1, -1):
            if pos >= page_boundaries[i][0]:
                return page_boundaries[i][1]
        return 1  # fallback

    # Now we need to find the starting position of each chunk in full_text.
    # Since split_text doesn't return offsets, we can reconstruct by scanning.
    # We'll iterate over chunks and find their start index in full_text.
    # A simple way: start at 0 and use .find() incrementally.
    last_pos = 0
    results = []
    for chunk in chunks:
        # Find the chunk in full_text starting from last_pos
        pos = full_text.find(chunk, last_pos)
        if pos == -1:
            # Fallback: if not found (rare), just use last_pos
            pos = last_pos
        page_num = get_page_for_pos(pos)
        results.append({
            "text": chunk,
            "metadata": {"source": source_filename, "page": page_num}
        })
        last_pos = pos + len(chunk)

    return results

# ----------------------------------------------------------------------
# Example usage (commented out):
# if __name__ == "__main__":
#     from pdf_parser import extract_pages   # assuming Module 1
#     pages = extract_pages("test.pdf")
#     chunks = chunk_document(pages, "test.pdf", chunk_size=500, overlap=50)
#     for chunk in chunks[:3]:
#         print(f"Page {chunk['metadata']['page']}: {chunk['text'][:100]}...")