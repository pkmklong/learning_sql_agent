"""
Configuration file for Healthcare SQL Agent
Manage all LLM settings and model configurations here
"""

import os
from typing import Dict, Any

# Default LLM to use (change this to switch models)
DEFAULT_LLM = "ollama"  # Options: "openai", "ollama", "llamacpp", "huggingface"

# LLM Configurations
LLM_CONFIGS = {
    "openai": {
        "type": "openai",
        "model_name": "gpt-3.5-turbo",
        "temperature": 0,
        "max_tokens": 2000,
        "requires_api_key": True,
        "api_key_env": "OPENAI_API_KEY"
    },
    
    "ollama": {
        "type": "ollama", 
        "model": "llama3.2",  # or "codellama", "llama3.1:70b", etc.
        "base_url": "http://localhost:11434",
        "temperature": 0,
        "requires_api_key": False,
        "setup_instructions": [
            "1. Install: curl -fsSL https://ollama.ai/install.sh | sh",
            "2. Pull model: ollama pull llama3.1", 
            "3. Start server: ollama serve"
        ]
    },
    
    "llamacpp": {
        "type": "llamacpp",
        "model_path": "./models/llama-2-7b-chat.q4_0.bin",  # Update this path
        "temperature": 0,
        "max_tokens": 2000,
        "n_ctx": 4096,
        "verbose": False,
        "requires_api_key": False,
        "setup_instructions": [
            "1. Install: pip install llama-cpp-python",
            "2. Download model: wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.q4_0.bin",
            "3. Update model_path in config.py"
        ]
    },
    
    "huggingface": {
        "type": "huggingface",
        "model_name": "microsoft/DialoGPT-medium",  # or "codellama/CodeLlama-7b-hf"
        "temperature": 0.1,
        "max_tokens": 512,
        "requires_api_key": False,
        "setup_instructions": [
            "1. Install: pip install transformers torch",
            "2. Model will download automatically on first use"
        ]
    }
}

# Quick model aliases for easy switching
MODEL_ALIASES = {
    "fast": "openai",           # Fast but costs money
    "free": "ollama",           # Free and good balance
    "offline": "llamacpp",      # Completely offline
    "light": "huggingface"      # Lightweight option
}

def get_llm_config(model_name: str = None) -> Dict[str, Any]:
    """
    Get configuration for specified model or default
    
    Args:
        model_name: Name of the model or alias
        
    Returns:
        Dictionary with model configuration
    """
    if model_name is None:
        model_name = DEFAULT_LLM
    
    # Check if it's an alias
    if model_name in MODEL_ALIASES:
        model_name = MODEL_ALIASES[model_name]
    
    if model_name not in LLM_CONFIGS:
        available = list(LLM_CONFIGS.keys()) + list(MODEL_ALIASES.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
    
    return LLM_CONFIGS[model_name].copy()

def validate_llm_setup(config: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validate if the LLM is properly set up
    
    Returns:
        (is_valid, error_message)
    """
    llm_type = config["type"]
    
    if config.get("requires_api_key", False):
        api_key_env = config.get("api_key_env")
        if api_key_env and not os.getenv(api_key_env):
            return False, f"Missing {api_key_env} environment variable"
    
    if llm_type == "llamacpp":
        model_path = config.get("model_path")
        if model_path and not os.path.exists(model_path):
            return False, f"Model file not found: {model_path}"
    
    return True, ""

def print_setup_instructions(model_name: str = None):
    """Print setup instructions for the specified model"""
    config = get_llm_config(model_name)
    
    print(f"\n🔧 Setup Instructions for {config['type'].upper()}:")
    print("=" * 50)
    
    if "setup_instructions" in config:
        for instruction in config["setup_instructions"]:
            print(instruction)
    else:
        print("No special setup required!")

def list_available_models():
    """List all available models and their status"""
    print("\n🤖 Available Models:")
    print("=" * 50)
    
    for name, config in LLM_CONFIGS.items():
        is_valid, error = validate_llm_setup(config)
        status = "✅ Ready" if is_valid else f"❌ {error}"
        print(f"{name:12} - {config['type']:12} - {status}")
    
    print("\n🔗 Quick Aliases:")
    for alias, target in MODEL_ALIASES.items():
        print(f"{alias:12} -> {target}")

# Hackathon-specific settings
HACKATHON_SETTINGS = {
    "database_path": "healthcare_hackathon.db",
    "verbose_agent": True,
    "max_query_time": 30,  # seconds
    "enable_safety_checks": True
}

# Sample model switching examples
USAGE_EXAMPLES = """
# Quick Examples:

# Use default model (set in DEFAULT_LLM)
agent = HackathonSQLAgent("healthcare_hackathon.db")

# Use specific model
agent = HackathonSQLAgent("healthcare_hackathon.db", llm_config="ollama")

# Use alias
agent = HackathonSQLAgent("healthcare_hackathon.db", llm_config="free")

# Custom config
custom_config = get_llm_config("ollama")
custom_config["model"] = "codellama"  # Use CodeLlama instead
agent = HackathonSQLAgent("healthcare_hackathon.db", llm_config=custom_config)
"""

if __name__ == "__main__":
    # Quick config test
    print("🏥 Healthcare SQL Agent - Model Configuration")
    list_available_models()
    print(USAGE_EXAMPLES)
