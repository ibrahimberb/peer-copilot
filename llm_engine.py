"""
LLM Engine for processing documents.
This module handles the backend processing of uploaded documents.
"""

from pathlib import Path
from typing import Optional
import mimetypes


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text content from PDF files.

    Args:
        file_path: Path to the PDF file

    Returns:
        Extracted text content
    """
    try:
        try:
            import PyPDF2

            pdf_module = PyPDF2
        except ImportError:
            import pypdf as PyPDF2

            pdf_module = PyPDF2

        text = ""
        with open(file_path, "rb") as f:
            pdf_reader = pdf_module.PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    except ImportError:
        return "[PDF processing requires PyPDF2 or pypdf. Install with: pip install pypdf2 or pip install pypdf]"
    except Exception as e:
        return f"[Error processing PDF: {str(e)}]"


def extract_text_from_file(file_path: str) -> str:
    """
    Extract text content from various file types.

    Args:
        file_path: Path to the file

    Returns:
        Extracted text content
    """
    file_path_obj = Path(file_path)
    file_extension = file_path_obj.suffix.lower()

    try:
        if file_extension == ".txt" or file_extension == ".md":
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

        elif file_extension == ".pdf":
            return extract_text_from_pdf(file_path)

        elif file_extension == ".docx":
            # Word document processing
            try:
                from docx import Document

                doc = Document(file_path)
                text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                return text
            except ImportError:
                return "[Word document processing requires python-docx. Install with: pip install python-docx]"

        else:
            # Try to read as text for code files, etc.
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            except UnicodeDecodeError:
                return f"[Binary file detected: {file_path_obj.name}. Text extraction not supported for this file type.]"

    except Exception as e:
        return f"[Error reading file: {str(e)}]"


def process_peer_review(
    manuscript_path: str, checklist_path: str, manuscript_name: str, checklist_name: str
) -> str:
    """
    Process peer review with manuscript and checklist files.

    Args:
        manuscript_path: Path to the manuscript PDF file
        checklist_path: Path to the checklist PDF file
        manuscript_name: Name of the manuscript file
        checklist_name: Name of the checklist file

    Returns:
        Processed result as formatted string
    """
    # Extract text from both PDFs
    manuscript_text = extract_text_from_pdf(manuscript_path)
    checklist_text = extract_text_from_pdf(checklist_path)

    # Print to console for debugging
    print(f"\n{'='*80}")
    print(f"Processing Peer Review")
    print(f"{'='*80}")
    print(f"\nManuscript: {manuscript_name}")
    print(f"Content Length: {len(manuscript_text)} characters")
    print(f"\nChecklist: {checklist_name}")
    print(f"Content Length: {len(checklist_text)} characters")
    print(f"{'='*80}")
    print("\nManuscript Content:")
    print("-" * 80)
    print(
        manuscript_text[:1000] + "..."
        if len(manuscript_text) > 1000
        else manuscript_text
    )
    print("-" * 80)
    print("\nChecklist Content:")
    print("-" * 80)
    print(checklist_text)
    print("-" * 80)
    print(f"{'='*80}\n")

    # Basic processing (placeholder for actual LLM integration)
    # TODO: Integrate with your preferred LLM API for peer review

    # For now, return a formatted summary
    result = f"""
### Peer Review Processing

**Manuscript:** {manuscript_name}  
**Checklist:** {checklist_name}  
**Manuscript Length:** {len(manuscript_text)} characters  
**Checklist Length:** {len(checklist_text)} characters

---

### Manuscript Preview

{manuscript_text[:500]}{"..." if len(manuscript_text) > 500 else ""}

---

### Checklist Preview

{checklist_text[:500]}{"..." if len(checklist_text) > 500 else ""}

---

### Next Steps

Peer review analysis will be performed here based on the checklist criteria.
"""

    return result


def analyze_document_with_llm(text_content: str, analysis_type: str = "summary") -> str:
    """
    Placeholder for LLM-based document analysis.

    Args:
        text_content: Extracted text from document
        analysis_type: Type of analysis to perform

    Returns:
        Analysis result
    """
    # This is where you would integrate with your LLM of choice
    # Examples: OpenAI, Anthropic Claude, local models via Ollama, etc.

    return f"[LLM Analysis - {analysis_type}] Placeholder for actual LLM processing."
