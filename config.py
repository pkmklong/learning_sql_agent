"""
Simple healthcare SQL agent configuration
"""
import os
from pydantic import BaseModel

# Default model
DEFAULT_MODEL = "ollama"

class ModelConfig(BaseModel):
    type: str
    model: str = ""
    model_name: str = ""
    base_url: str = ""
    api_key_env: str = ""

# Available models
MODELS = {
    "ollama": ModelConfig(
        type="ollama",
        model="llama3.2",
        base_url="http://localhost:11434"
    ),
    "openai": ModelConfig(
        type="openai", 
        model_name="gpt-3.5-turbo",
        api_key_env="OPENAI_API_KEY"
    )
}

def get_model_config(model_name: str = None):
    """Get config for a model"""
    if model_name is None:
        model_name = DEFAULT_MODEL
    
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    
    return MODELS[model_name]

def check_model_ready(model_name: str = None):
    """Check if model is ready to use"""
    config = get_model_config(model_name)
    
    if config.type == "openai":
        if not os.getenv(config.api_key_env):
            return False, f"Set {config.api_key_env} in .env file"
    
    return True, "Ready"

def list_models():
    """Show available models"""
    for name in MODELS:
        ready, status = check_model_ready(name)
        print(f"{'✅' if ready else '❌'} {name} - {status}")
