# Peer Copilot ✨

A minimalist, elegant Streamlit application for document processing powered by LLM backend.

## Features

- 🎨 **Minimalist UI** - Clean, modern, and elegant interface
- 📄 **Document Upload** - Support for multiple file types (PDF, TXT, MD, DOCX, code files)
- 🤖 **LLM Processing** - Backend engine ready for LLM integration
- 🚀 **Easy to Extend** - Well-structured codebase for customization

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

1. **Clone or navigate to the project directory:**
   ```bash
   cd peer-copilot
   ```

2. **Initialize the project with uv:**
   ```bash
   uv sync
   ```

   This will automatically install all dependencies using the existing uv configuration files.

## Usage

1. **Run the Streamlit app:**
   ```bash
   streamlit run main.py
   ```

2. **Open your browser** to the URL shown in the terminal (typically `http://localhost:8501`)

3. **Upload a document:**
   - Select a document type from the dropdown
   - Upload your file using the file uploader
   - Click "Process Document" to analyze

## LLM Integration

The app is structured to easily integrate with your preferred LLM provider. Edit `llm_engine.py` to add:

- **OpenAI**: GPT-4, GPT-3.5
- **Anthropic**: Claude
- **Local Models**: Ollama, LM Studio
- **Other APIs**: Hugging Face, Cohere, etc.

### Example Integration

See the `process_document` function in `llm_engine.py` for integration examples and placeholders.

## Project Structure

```
peer-copilot/
├── main.py              # Streamlit app entry point
├── llm_engine.py        # LLM processing backend
├── pyproject.toml       # Project dependencies
└── README.md           # This file
```

## Customization

### UI Styling

The app uses custom CSS embedded in `main.py`. Modify the `<style>` block to customize:
- Colors and gradients
- Fonts and typography
- Spacing and layout
- Animations and transitions

### Document Processing

Extend `llm_engine.py` to:
- Add support for new file types
- Implement custom processing logic
- Integrate with your LLM provider
- Add analysis types (summarization, Q&A, etc.)

## Development

### Code Quality with Ruff

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting.

**Install ruff:**
```bash
pip install -e ".[dev]"
```

**Check code:**
```bash
ruff check .
```

**Format code:**
```bash
ruff format .
```

**Auto-fix issues:**
```bash
ruff check --fix .
```

## Requirements

- Python 3.10+
- Streamlit 1.51+
- pypdf2 (for PDF processing)
- docx (for Word document processing)

## License

MIT

---

Built with ❤️ using Streamlit

