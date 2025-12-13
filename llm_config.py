"""
LLM Configuration Settings
Configure your LLM provider settings here.
"""

# LLM Provider Type
LLM_TYPE = "lm_studio"  # Options: "lm_studio", "openai", "anthropic", etc.

# LM Studio Configuration (Local)
LLM_BASE_URL = "http://127.0.0.1:1234"
LLM_API_KEY = None  # Not needed for local LM Studio

# Model Configuration
LLM_MODEL = "openai/gpt-oss-20b"  # Model name in LM Studio (use full model name as shown in LM Studio)

# API Endpoints
CHAT_COMPLETIONS_ENDPOINT = "/v1/chat/completions"
MODELS_ENDPOINT = "/v1/models"
COMPLETIONS_ENDPOINT = "/v1/completions"
EMBEDDINGS_ENDPOINT = "/v1/embeddings"

# Generation Parameters
MAX_TOKENS = -1  # -1 for unlimited tokens (LM Studio default), or set positive number
TEMPERATURE = 0.7
TOP_P = 1.0
STREAM = False  # Set to True for streaming responses

# Timeout settings
REQUEST_TIMEOUT = 60  # seconds

