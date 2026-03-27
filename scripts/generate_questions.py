#!/usr/bin/env python3
"""
Generate a set of test questions from a PDF manual for validation/demonstration.

Usage:
    python scripts/generate_questions.py path/to/manual.pdf
    python scripts/generate_questions.py --model gemini path/to/manual.pdf
    python scripts/generate_questions.py --model openai path/to/manual1.pdf path/to/manual2.pdf

Output:
    JSON files will be saved in validation/questions/<filename>_questions.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pdfplumber
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# GPT-4o-mini has a 128k token context window. 120k chars is a safe character
# limit that stays well within that budget after tokenisation overhead.
PDF_TEXT_CHAR_LIMIT = 120_000


def extract_full_text(pdf_path: str) -> tuple[str, int]:
    """
    Extract all text from a PDF and return (text, total_pages).
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
            return text, len(pdf.pages)
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}", file=sys.stderr)
        return "", 0


def generate_questions_openai(pdf_text: str, total_pages: int) -> list[dict]:
    """
    Call OpenAI API to generate questions from the PDF text.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment. Please set it in .env file.")

    client = OpenAI(api_key=api_key)

    truncated_text = pdf_text[:PDF_TEXT_CHAR_LIMIT]

    prompt = f"""
You are an expert in NGS protocols. Read the following user manual and generate a list of 20-25 realistic questions that a lab technician might ask. Include questions about:
- Input amounts (DNA/RNA)
- Incubation times and temperatures
- Reagent storage conditions
- Equipment settings (e.g., Covaris shearing)
- Steps in the protocol
- Important notes and limitations
- Any other practical details

For each question, provide:
- The question text
- The expected answer (a short phrase or sentence, derived from the manual)
- The approximate page number where the answer can be found (based on the text; if unknown, put 0)

Output the result as a JSON list of objects, each with keys: "question", "expected_answer", "source_page".

Manual text:
{truncated_text}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates realistic questions from technical documentation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=4000,
        )
        content = response.choices[0].message.content
        # Extract JSON from the response (might be wrapped in markdown code fences)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        questions = json.loads(content)
        return questions
    except Exception as e:
        print(f"Error calling OpenAI API: {e}", file=sys.stderr)
        return []


def generate_questions_gemini(pdf_text: str, total_pages: int) -> list[dict]:
    """
    Call Google Gemini API to generate questions from the PDF text.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment. Please set it in .env file.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')  # or 'gemini-1.5-pro' if needed

    truncated_text = pdf_text[:PDF_TEXT_CHAR_LIMIT]

    prompt = f"""
You are an expert in NGS protocols. Read the following user manual and generate a list of 20-25 realistic questions that a lab technician might ask. Include questions about:
- Input amounts (DNA/RNA)
- Incubation times and temperatures
- Reagent storage conditions
- Equipment settings (e.g., Covaris shearing)
- Steps in the protocol
- Important notes and limitations
- Any other practical details

For each question, provide:
- The question text
- The expected answer (a short phrase or sentence, derived from the manual)
- The approximate page number where the answer can be found (based on the text; if unknown, put 0)

Output the result as a JSON list of objects, each with keys: "question", "expected_answer", "source_page".

Manual text:
{truncated_text}
"""

    try:
        response = model.generate_content(prompt)
        content = response.text
        # Extract JSON from the response (might be wrapped in markdown code fences)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        questions = json.loads(content)
        return questions
    except Exception as e:
        print(f"Error calling Gemini API: {e}", file=sys.stderr)
        return []


def main():
    parser = argparse.ArgumentParser(description="Generate question sets from PDF manuals.")
    parser.add_argument("pdf_paths", nargs="+", help="Paths to PDF files")
    parser.add_argument("--model", choices=["openai", "gemini"], default="openai",
                        help="Which LLM to use for question generation (default: openai)")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path("validation/questions")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Choose the generator function based on model
    if args.model == "openai":
        generator = generate_questions_openai
    else:
        generator = generate_questions_gemini

    for pdf_path in args.pdf_paths:
        print(f"Processing {pdf_path} with {args.model}...")
        text, total_pages = extract_full_text(pdf_path)
        if not text:
            print(f"  Skipping {pdf_path}: no text extracted.")
            continue

        questions = generator(text, total_pages)
        if not questions:
            print(f"  Failed to generate questions for {pdf_path}.")
            continue

        # Save to JSON
        base_name = Path(pdf_path).stem
        output_file = output_dir / f"{base_name}_questions.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(questions, f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(questions)} questions to {output_file}")


if __name__ == "__main__":
    main()