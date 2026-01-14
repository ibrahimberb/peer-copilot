"""
LLM Engine for processing documents.
This module handles the backend processing of uploaded documents.
"""

from pathlib import Path
from typing import Optional, List, Dict
import mimetypes
import requests
import re
import json
import llm_config


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


def call_llm_chat_completion(
    messages: List[Dict[str, str]], system_prompt: Optional[str] = None
) -> str:
    try:
        api_messages = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        api_messages.extend(messages)

        url = f"{llm_config.LLM_BASE_URL}{llm_config.CHAT_COMPLETIONS_ENDPOINT}"
        headers = {"Content-Type": "application/json"}
        if llm_config.LLM_API_KEY:
            headers["Authorization"] = f"Bearer {llm_config.LLM_API_KEY}"

        payload = {
            "model": llm_config.LLM_MODEL,
            "messages": api_messages,
            "temperature": llm_config.TEMPERATURE,
            "max_tokens": llm_config.MAX_TOKENS,
            "stream": llm_config.STREAM,
        }
        
        if llm_config.MAX_TOKENS > 0:
            payload["top_p"] = llm_config.TOP_P

        response = requests.post(
            url, json=payload, headers=headers, timeout=llm_config.REQUEST_TIMEOUT
        )
        
        if response.status_code != 200:
            try:
                error_detail = response.json()
                error_msg = error_detail.get("error", {}).get("message", response.text)
                return f"[Error: LLM API returned error (Status {response.status_code}): {error_msg}]"
            except:
                return f"[Error: LLM API returned error (Status {response.status_code}): {response.text}]"
        
        response_data = response.json()
        if "choices" in response_data and len(response_data["choices"]) > 0:
            return response_data["choices"][0]["message"]["content"]
        else:
            return "[Error: No response from LLM]"

    except requests.exceptions.ConnectionError:
        return f"[Error: Could not connect to LLM. Is LM Studio running at {llm_config.LLM_BASE_URL}?]"
    except requests.exceptions.Timeout:
        return "[Error: LLM request timed out]"
    except requests.exceptions.RequestException as e:
        return f"[Error: Request failed: {str(e)}]"
    except Exception as e:
        return f"[Error calling LLM: {str(e)}]"


def parse_checklist_to_structured_data(checklist_markdown: str) -> List[Dict[str, any]]:
    """
    Parse the LLM-generated markdown checklist into a structured list of items.
    
    Args:
        checklist_markdown: Markdown formatted checklist from LLM
        
    Returns:
        List of dictionaries containing checklist items with metadata
    """
    structured_items = []
    current_section = "General"
    
    lines = checklist_markdown.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Detect section headers (lines ending with just heading markers or numbered sections)
        if re.match(r'^#{1,3}\s+(.+)$', line):
            # Markdown heading
            match = re.match(r'^#{1,3}\s+(.+)$', line)
            current_section = match.group(1).strip()
            continue
        elif re.match(r'^\d+\.\s+(.+)$', line) and not line.endswith('?'):
            # Numbered section heading (but not a question)
            match = re.match(r'^\d+\.\s+(.+)$', line)
            potential_section = match.group(1).strip()
            # If it's short and doesn't look like a checklist item, treat as section
            if len(potential_section) < 100 and not any(keyword in potential_section.lower() for keyword in ['does', 'is', 'are', 'should', 'verify', 'check']):
                current_section = potential_section
                continue
        
        # Extract checklist items (bullets or sub-bullets)
        if re.match(r'^[-*]\s+(.+)$', line):
            # Main bullet point
            item_text = re.match(r'^[-*]\s+(.+)$', line).group(1).strip()
            structured_items.append({
                'section': current_section,
                'item': item_text,
                'type': 'main',
                'is_question': line.endswith('?'),
                'checked': False
            })
        elif re.match(r'^\s+[-*]\s+(.+)$', line):
            # Sub-bullet point
            item_text = re.match(r'^\s+[-*]\s+(.+)$', line).group(1).strip()
            structured_items.append({
                'section': current_section,
                'item': item_text,
                'type': 'sub',
                'is_question': line.endswith('?'),
                'checked': False
            })
    
    return structured_items


def process_checklist_with_llm(checklist_text: str) -> str:
    max_chars = 6000
    
    if len(checklist_text) > max_chars:
        checklist_text = checklist_text[:max_chars] + "\n\n[... document truncated due to length ...]"
    
    system_prompt = """You are an expert at extracting and structuring peer review checklists from documents. 
Your task is to analyze the provided checklist document and extract all checklist items, criteria, and evaluation points.

Format your response as clear, actionable bullet points organized by categories if present.
Each item should be a specific, actionable instruction or criterion that can be used for peer review.

If the document contains categories or sections, organize the checklist items under those categories.
Use markdown formatting with bullet points (-) and sub-bullets if needed."""

    user_prompt = f"""Please extract and structure the checklist items from the following peer review checklist document:

{checklist_text}

Provide a well-organized, structured checklist with clear bullet points that can be used for evaluating manuscripts."""

    messages = [{"role": "user", "content": user_prompt}]
    structured_checklist = call_llm_chat_completion(messages, system_prompt=system_prompt)
    
    print(f"\nStructured Checklist from LLM:\n{structured_checklist}\n")
    
    return structured_checklist


def evaluate_manuscript_item(
    checklist_item: Dict[str, any],
    manuscript_text: str,
    item_number: int,
    total_items: int
) -> str:
    """
    Evaluate a single checklist item against the manuscript using LLM.
    
    Args:
        checklist_item: Dictionary containing checklist item details
        manuscript_text: Full manuscript text
        item_number: Current item number for context
        total_items: Total number of items
        
    Returns:
        LLM evaluation response
    """
    system_prompt = """You are an expert peer reviewer for academic manuscripts. Your task is to evaluate the manuscript based on specific checklist criteria.

Provide a thorough, constructive, and evidence-based assessment. Your response should:
1. Directly answer the question or address the criterion
2. Cite specific examples from the manuscript when relevant
3. Be objective and professional
4. Provide actionable feedback if issues are found
5. Keep your response concise (2-4 sentences for most items)

If the manuscript does not contain enough information to evaluate a criterion, state that clearly."""

    item_text = checklist_item['item']
    section = checklist_item['section']
    is_question = checklist_item['is_question']
    
    # Truncate manuscript if too long (keep first 8000 chars for context)
    manuscript_excerpt = manuscript_text[:8000] + "\n\n[...manuscript continues...]" if len(manuscript_text) > 8000 else manuscript_text
    
    if is_question:
        user_prompt = f"""Please evaluate the following aspect of the manuscript:

**Section:** {section}
**Question:** {item_text}

**Manuscript to evaluate:**
{manuscript_excerpt}

Provide your assessment:"""
    else:
        user_prompt = f"""Please verify the following criterion for the manuscript:

**Section:** {section}
**Criterion:** {item_text}

**Manuscript to evaluate:**
{manuscript_excerpt}

Provide your assessment:"""
    
    messages = [{"role": "user", "content": user_prompt}]
    response = call_llm_chat_completion(messages, system_prompt=system_prompt)
    
    return response


def evaluate_manuscript_with_checklist(
    checklist_items: List[Dict[str, any]],
    manuscript_text: str,
    output_path: str,
    progress_callback=None
) -> List[Dict[str, any]]:
    """
    Evaluate manuscript against all checklist items using LLM.
    
    Args:
        checklist_items: List of checklist item dictionaries
        manuscript_text: Full manuscript text
        output_path: Path to save review results JSON
        progress_callback: Optional callback function(current, total, item_name)
        
    Returns:
        List of evaluation results
    """
    review_results = []
    total_items = len(checklist_items)
    
    print(f"\n🔍 Starting manuscript evaluation with {total_items} checklist items...\n")
    
    for idx, item in enumerate(checklist_items, 1):
        item_name = item['item'][:60] + "..." if len(item['item']) > 60 else item['item']
        
        # Progress callback for UI
        if progress_callback:
            progress_callback(idx, total_items, item_name)
        
        print(f"[{idx}/{total_items}] Evaluating: {item_name}")
        
        # Get LLM evaluation
        llm_response = evaluate_manuscript_item(item, manuscript_text, idx, total_items)
        
        # Build result entry
        result = {
            'item_number': idx,
            'section': item['section'],
            'checklist_item': item['item'],
            'type': item['type'],
            'is_question': item['is_question'],
            'llm_evaluation': llm_response,
            'status': 'completed' if not llm_response.startswith('[Error') else 'error'
        }
        
        review_results.append(result)
    
    # Save results to JSON
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(review_results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Review results saved to: {output_path}")
    except Exception as e:
        print(f"⚠️ Warning: Could not save review results: {e}")
    
    return review_results


def process_peer_review(
    manuscript_path: str, checklist_path: str, manuscript_name: str, checklist_name: str
) -> tuple[str, List[Dict[str, any]]]:
    """
    Process peer review with manuscript and checklist files.

    Args:
        manuscript_path: Path to the manuscript PDF file
        checklist_path: Path to the checklist PDF file
        manuscript_name: Name of the manuscript file
        checklist_name: Name of the checklist file

    Returns:
        Tuple of (formatted result string, structured checklist data)
    """
    manuscript_text = extract_text_from_pdf(manuscript_path)
    checklist_text = extract_text_from_pdf(checklist_path)

    print(f"\nProcessing: {manuscript_name} ({len(manuscript_text)} chars) + {checklist_name} ({len(checklist_text)} chars)")

    structured_checklist = process_checklist_with_llm(checklist_text)

    if structured_checklist.startswith("[Error"):
        structured_checklist = f"**Raw Checklist Text:**\n\n{checklist_text[:1000]}{'...' if len(checklist_text) > 1000 else ''}"
    
    # Parse the checklist into structured data
    checklist_items = parse_checklist_to_structured_data(structured_checklist)
    
    # Save structured checklist to JSON file for persistence
    try:
        checklist_json_path = Path(checklist_path).parent / "parsed_checklist.json"
        with open(checklist_json_path, "w", encoding="utf-8") as f:
            json.dump(checklist_items, f, indent=2, ensure_ascii=False)
        print(f"\nSaved structured checklist to: {checklist_json_path}")
        print(f"Total checklist items extracted: {len(checklist_items)}")
    except Exception as e:
        print(f"Warning: Could not save checklist JSON: {e}")

    # For now, return a formatted summary with structured checklist
    result = f"""
### Peer Review Processing

**Manuscript:** {manuscript_name}  
**Checklist:** {checklist_name}  
**Manuscript Length:** {len(manuscript_text)} characters  
**Checklist Length:** {len(checklist_text)} characters
**Checklist Items Extracted:** {len(checklist_items)}

---

### Structured Checklist (from LLM)

{structured_checklist}

---

### Manuscript Preview

{manuscript_text[:500]}{"..." if len(manuscript_text) > 500 else ""}

---

### Next Steps

The structured checklist above will be used to evaluate the manuscript in the next step.
"""

    return result, checklist_items


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
