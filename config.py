"""
Simple healthcare SQL agent configuration
"""

import os
from pydantic import BaseModel

# Which model to use by default
DEFAULT_MODEL = "ollama"  # Change to "openai" if you prefer

class ModelConfig(BaseModel):
    """Base model configuration"""
    type: str
    temperature: float = 0.0

class OllamaConfig(ModelConfig):
    """Ollama local model settings"""
    type: str = "ollama"
    model: str = "llama3.2:latest"
    base_url: str = "http://localhost:11434"

class OpenAIConfig(ModelConfig):
    """OpenAI API model settings"""
    type: str = "openai"
    model_name: str = "gpt-3.5-turbo"
    api_key_env: str = "OPENAI_API_KEY"

# Available models
MODELS = {
    "ollama": OllamaConfig(),
    "openai": OpenAIConfig()
}

def get_model_config(model_name: str = None):
    """Get config for a model"""
    if model_name is None:
        model_name = DEFAULT_MODEL
    
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Choose: {list(MODELS.keys())}")
    
    return MODELS[model_name]

def check_model_ready(model_name: str = None) -> tuple[bool, str]:
    """Check if model is ready to use"""
    config = get_model_config(model_name)
    
    if config.type == "openai":
        if not os.getenv(config.api_key_env):
            return False, f"Set {config.api_key_env} in .env file"
    
    return True, "Ready"

def list_models():
    """Show available models"""
    print("\n🤖 Available Models:")
    for name, config in MODELS.items():
        ready, status = check_model_ready(name)
        marker = "✅" if ready else "❌"
        cost = "Free" if name == "ollama" else "Paid"
        print(f"{marker} {name} ({cost}) - {status}")

if __name__ == "__main__":
    list_models()
    print(f"\nDefault: {DEFAULT_MODEL}")
