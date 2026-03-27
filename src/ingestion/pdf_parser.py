import pdfplumber
from typing import Union, IO, List, Tuple


def table_to_markdown(table: List[List[str]]) -> str:
    """
    Convert a list of rows (list of strings) into a Markdown table.

    Args:
        table: List of rows, each row is a list of strings (cells).

    Returns:
        A string containing the Markdown representation of the table.
    """
    if not table or len(table) < 1:
        return ""

    # Determine number of columns (use max row length)
    num_cols = max(len(row) for row in table)
    # Normalise rows: ensure all have same number of columns
    norm_rows = []
    for row in table:
        if len(row) < num_cols:
            row = row + [""] * (num_cols - len(row))
        norm_rows.append(row)

    # Build header and separator
    header = "| " + " | ".join(norm_rows[0]) + " |"
    separator = "|" + "|".join([" --- " for _ in range(num_cols)]) + "|"
    lines = [header, separator]

    # Add data rows
    for row in norm_rows[1:]:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def extract_tables_from_page(page) -> List[str]:
    """
    Extract all tables from a pdfplumber page and return them as Markdown strings.

    Args:
        page: A pdfplumber page object.

    Returns:
        List of Markdown table strings.
    """
    tables = page.extract_tables()
    markdown_tables = []
    for table in tables:
        if table:  # non-empty
            md = table_to_markdown(table)
            if md:
                markdown_tables.append(md)
    return markdown_tables


def extract_pages(pdf_source: Union[str, IO]) -> List[Tuple[int, str]]:
    """
    Extract text and tables from a PDF.

    Args:
        pdf_source: Either a file path (str) or a file-like object (e.g., BytesIO).
    Returns:
        List of (page_number, processed_text).
    """
    result = []
    try:
        with (pdfplumber.open(pdf_source) as pdf):
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text is None:
                    text = ""
                tables_md = extract_tables_from_page(page)
                if tables_md:
                    combined = text + "\n\n" + "\n\n".join(tables_md)
                else:
                    combined = text
                result.append((i, combined))
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return []

    return result


# ----------------------------------------------------------------------
# Example usage (commented out):
# if __name__ == "__main__":
#     pages = extract_pages("sample.pdf")
#     for page_num, content in pages:
#         print(f"Page {page_num}: {len(content)} characters")
#         print(content[:500])  # first 500 chars