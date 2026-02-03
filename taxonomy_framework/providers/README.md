# LLM Providers

This module provides a unified interface for multiple LLM providers, enabling seamless switching between cloud and local models.

## Supported Providers

| Provider | Module | Tool Calling | JSON Mode | Use Case |
|----------|--------|--------------|-----------|----------|
| OpenAI | `openai_provider` | ✅ | ✅ | Production, high quality |
| Anthropic | `anthropic_provider` | ✅ | ✅ | Production, long context |
| Google Gemini | `google_provider` | ✅ | ✅ | Production, multimodal |
| Cohere | `cohere_provider` | ✅ | ✅ | Production, RAG-focused |
| Ollama | `ollama_provider` | ✅* | ✅ | Local development |
| vLLM | `vllm_provider` | ✅* | ✅ | Self-hosted production |
| HuggingFace TGI | `huggingface_tgi` | ❌ | ✅ | Self-hosted production |

*Depends on model capabilities

## Quick Start

### Using ProviderFactory (Recommended)

```python
from taxonomy_framework.providers import ProviderFactory

# List available providers
print(ProviderFactory.list_providers())
# ['openai', 'anthropic', 'google', 'cohere', 'ollama', 'vllm', 'huggingface_tgi']

# Create a provider
provider = ProviderFactory.create(
    provider_type="openai",
    model="gpt-4o-mini",
    api_key="sk-..."
)

# Use the provider
response = provider.generate_json(
    system_prompt="You are a classifier.",
    user_prompt="Classify this: iPhone 15 Pro",
)
```

### Direct Provider Instantiation

```python
from taxonomy_framework.providers.openai_provider import OpenAIProvider

provider = OpenAIProvider(
    model="gpt-4o-mini",
    api_key="sk-...",
    api_base=None  # Optional custom endpoint
)
```

## Provider Configuration

### OpenAI

```python
provider = ProviderFactory.create(
    "openai",
    model="gpt-4o-mini",      # or "gpt-4o", "gpt-4-turbo", etc.
    api_key="sk-...",
    api_base=None,            # For Azure OpenAI, set endpoint URL
    organization=None,        # Optional org ID
    timeout=60               # Request timeout
)
```

**Environment Variables:**
- `OPENAI_API_KEY`
- `OPENAI_API_BASE` (optional)
- `OPENAI_ORGANIZATION` (optional)

### Anthropic

```python
provider = ProviderFactory.create(
    "anthropic",
    model="claude-3-haiku-20240307",  # or claude-3-sonnet, claude-3-opus
    api_key="sk-ant-...",
    max_tokens=4096,
    timeout=60
)
```

**Environment Variables:**
- `ANTHROPIC_API_KEY`

### Google Gemini

```python
provider = ProviderFactory.create(
    "google",
    model="gemini-1.5-flash",  # or gemini-1.5-pro
    api_key="...",
    safety_settings=None  # Optional custom safety settings
)
```

**Environment Variables:**
- `GOOGLE_API_KEY`

### Cohere

```python
provider = ProviderFactory.create(
    "cohere",
    model="command-r",  # or command-r-plus
    api_key="..."
)
```

**Environment Variables:**
- `COHERE_API_KEY`

### Ollama (Local)

```python
provider = ProviderFactory.create(
    "ollama",
    model="llama3",  # or mistral, codellama, etc.
    base_url="http://localhost:11434/v1"  # Default Ollama endpoint
)
```

**No API key required** - Ollama runs locally.

**Starting Ollama:**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3

# Ollama runs automatically, or start manually:
ollama serve
```

### vLLM (Self-Hosted)

```python
provider = ProviderFactory.create(
    "vllm",
    model="meta-llama/Llama-2-7b-chat-hf",
    base_url="http://localhost:8000/v1"
)
```

**Starting vLLM Server:**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --port 8000
```

### HuggingFace TGI (Self-Hosted)

```python
provider = ProviderFactory.create(
    "huggingface_tgi",
    model="mistralai/Mistral-7B-Instruct-v0.2",
    base_url="http://localhost:8080",
    api_key=None  # Optional HF token
)
```

**Starting TGI Server:**
```bash
docker run --gpus all -p 8080:80 \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id mistralai/Mistral-7B-Instruct-v0.2
```

## Provider Interface

All providers implement `BaseLLMProvider`:

```python
from abc import ABC, abstractmethod
from taxonomy_framework.providers.base import ProviderCapabilities, ToolCallResult

class BaseLLMProvider(ABC):
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model identifier."""
        ...
    
    @property
    @abstractmethod
    def supports_tool_calling(self) -> bool:
        """Whether this provider supports function/tool calling."""
        ...
    
    @property
    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """Get provider capabilities."""
        ...
    
    @abstractmethod
    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """Generate a JSON response."""
        ...
    
    @abstractmethod
    def call_with_tools(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: List[Dict[str, Any]],
        tool_choice: Optional[str] = None
    ) -> ToolCallResult:
        """Call with function/tool definitions."""
        ...
```

### ProviderCapabilities

```python
@dataclass
class ProviderCapabilities:
    supports_json_mode: bool
    supports_tools: bool
    supports_streaming: bool
    max_context_length: int
    max_output_tokens: int
```

### ToolCallResult

```python
@dataclass
class ToolCallResult:
    tool_name: str
    tool_args: Dict[str, Any]
    raw_response: Optional[str] = None
```

## Using with HybridClassifier

```python
from taxonomy_framework.pipeline import HybridClassifier
from taxonomy_framework.embeddings import SentenceTransformerBackend, EnsembleEmbedder
from taxonomy_framework.providers import ProviderFactory

# Create provider
llm = ProviderFactory.create("openai", model="gpt-4o-mini", api_key="sk-...")

# Create classifier
classifier = HybridClassifier(
    taxonomy=taxonomy,
    embedder=EnsembleEmbedder([SentenceTransformerBackend()]),
    llm=llm
)

# Classify
result = classifier.classify("iPhone 15 Pro")
```

## JSON Generation

```python
response = provider.generate_json(
    system_prompt="Extract product info as JSON.",
    user_prompt="iPhone 15 Pro, 256GB, Blue",
    max_retries=3
)
# Returns: {"product": "iPhone 15 Pro", "storage": "256GB", "color": "Blue"}
```

## Tool Calling

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "select_category",
            "description": "Select the best matching category",
            "parameters": {
                "type": "object",
                "properties": {
                    "category_id": {
                        "type": "integer",
                        "description": "ID of the selected category"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence score 0-1"
                    }
                },
                "required": ["category_id"]
            }
        }
    }
]

result = provider.call_with_tools(
    system_prompt="Select the best category for the product.",
    user_prompt="iPhone 15 Pro",
    tools=tools,
    tool_choice="select_category"  # Force specific tool
)

print(result.tool_name)  # "select_category"
print(result.tool_args)  # {"category_id": 1, "confidence": 0.95}
```

## Error Handling

```python
from taxonomy_framework.providers import ProviderFactory

try:
    provider = ProviderFactory.create("openai", model="gpt-4o-mini")
    response = provider.generate_json(system_prompt, user_prompt)
except ImportError as e:
    print(f"Provider SDK not installed: {e}")
except ValueError as e:
    print(f"Invalid configuration: {e}")
except Exception as e:
    print(f"API error: {e}")
```

## Adding Custom Providers

Implement `BaseLLMProvider` and register with the factory:

```python
from taxonomy_framework.providers.base import BaseLLMProvider, ProviderCapabilities

class MyCustomProvider(BaseLLMProvider):
    def __init__(self, model: str, api_key: str):
        self._model = model
        self._api_key = api_key
    
    @property
    def model_name(self) -> str:
        return self._model
    
    @property
    def supports_tool_calling(self) -> bool:
        return True
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_json_mode=True,
            supports_tools=True,
            supports_streaming=False,
            max_context_length=8192,
            max_output_tokens=2048
        )
    
    def generate_json(self, system_prompt, user_prompt, max_retries=3):
        # Your implementation
        ...
    
    def call_with_tools(self, system_prompt, user_prompt, tools, tool_choice=None):
        # Your implementation
        ...

# Register with factory
from taxonomy_framework.providers.factory import ProviderFactory
ProviderFactory._registry["my_custom"] = MyCustomProvider
```

## Best Practices

1. **API Keys**: Always use environment variables, never hardcode
2. **Error Handling**: Wrap provider calls in try/except
3. **Timeouts**: Set appropriate timeouts for your use case
4. **Retries**: Use `max_retries` parameter for resilience
5. **Local First**: Use Ollama for development to avoid API costs
6. **Model Selection**: Start with smaller models, scale up as needed

## Performance Tips

- **Ollama**: Best for development and testing
- **vLLM**: Best throughput for batch processing
- **OpenAI/Anthropic**: Best quality for production
- **TGI**: Good balance of quality and self-hosted control
