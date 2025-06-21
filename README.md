# 🏥 Healthcare SQL Agent Hackathon

Query healthcare data with natural language using LangChain! Now supports multiple LLM options including **free local models**.

## 🚀 Quick Setup

```bash
git clone <your-repo-url>
cd healthcare-sql-agent

# Create virtual environment (recommended)
python -m venv healthcare-ai-env

# Activate virtual environment
# On macOS/Linux:
source healthcare-ai-env/bin/activate
# On Windows:
healthcare-ai-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment (optional - only needed for OpenAI)
cp .env.example .env
# Add your OpenAI API key to .env if using OpenAI

# Run the agent
python healthcare_agent.py
```

## 🤖 Supported Models

| Model | Type | Cost | Setup Difficulty | Best For |
|-------|------|------|------------------|----------|
| **Ollama** | Local | Free | ⭐⭐ | **Recommended starter** |
| OpenAI GPT | API | $$ | ⭐ | Fast & accurate |
| LlamaCpp | Local | Free | ⭐⭐⭐ | Offline use |
| HuggingFace | Local | Free | ⭐⭐ | Lightweight |

## ⚡ Quick Start Examples

```python
# Use default model (set in config.py)
agent = HackathonSQLAgent("healthcare_hackathon.db")

# Switch models easily
agent = HackathonSQLAgent("healthcare_hackathon.db", llm_config="ollama")

# Use aliases for quick switching
agent = HackathonSQLAgent("healthcare_hackathon.db", llm_config="free")    # -> Ollama
agent = HackathonSQLAgent("healthcare_hackathon.db", llm_config="fast")   # -> OpenAI

# Check available models
python config.py
```

## 🆓 Free Local Setup (Recommended)

### Option 1: Ollama (Easiest)
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3.2

# Start server
ollama serve

# Update config.py
DEFAULT_LLM = "ollama"
```

### Option 2: LlamaCpp (Offline)
```bash
# Install
pip install llama-cpp-python

# Download model
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.q4_0.bin

# Update model path in config.py
```

## 📊 What You Get

- 🔒 **Secure SQL agent** with read-only protections
- 🏥 **Healthcare datasets**: Prescriptions, diagnoses, providers  
- 📋 **Real medical codes**: ICD-10, NDC, CPT codes
- 🚀 **Multiple LLM options**: From free local to cloud APIs
- ⚙️ **Easy configuration**: Switch models in config.py

## 💬 Example Queries

```python
agent.query("How many diabetes patients received metformin?")
agent.query("Which specialty has the highest claim volumes?") 
agent.query("Show prescription patterns for chronic conditions")
agent.query("Find patients with both cardiac diagnosis and medications")
agent.query("What's the average cost per specialty?")
```

## 🏆 Challenge Ideas

| Level | Challenge | Description |
|-------|-----------|-------------|
| 🥉 **Basic** | Pattern Analysis | Find relationships between diagnoses and prescriptions |
| 🥈 **Intermediate** | Provider Dashboard | Build analytics for provider performance |
| 🥇 **Advanced** | Adherence Tracker | Create medication adherence monitoring |
| 🏆 **Expert** | Multi-Agent System | Build coordinated healthcare analytics agents |

## 📁 Project Structure

```
healthcare-sql-agent/
├── README.md                 # This file
├── requirements.txt          # Python dependencies  
├── .env.example             # Environment template
├── config.py                # 🎯 Model configurations
├── healthcare_agent.py      # Main agent code
└── healthcare_hackathon.db  # Auto-generated database
```

## 🛠️ Configuration Management

All model settings are managed in `config.py`:

```python
# Switch default model
DEFAULT_LLM = "ollama"  # Options: "openai", "ollama", "llamacpp", "huggingface"

# Quick aliases
MODEL_ALIASES = {
    "fast": "openai",      # Fast but costs money
    "free": "ollama",      # Free and good balance  
    "offline": "llamacpp", # Completely offline
    "light": "huggingface" # Lightweight option
}
```

## 🔧 Model-Specific Setup

<details>
<summary><b>🦙 Ollama Setup (Recommended)</b></summary>

```bash
# 1. Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Pull models
ollama pull llama3.1      # General purpose (~4GB)
ollama pull codellama     # Better for SQL (~4GB) 

# 3. Start server
ollama serve

# 4. Test
python -c "
agent = HackathonSQLAgent('healthcare_hackathon.db', 'ollama')
print(agent.query('How many claims are there?'))
"
```

**Benefits:**
- ✅ 100% free and local
- ✅ No API keys needed
- ✅ Privacy-focused
- ✅ Works offline
</details>

<details>
<summary><b>💰 OpenAI Setup</b></summary>

```bash
# 1. Get API key from https://platform.openai.com/api-keys

# 2. Add to .env file
echo "OPENAI_API_KEY=sk-your-key-here" >> .env

# 3. Set in config.py
DEFAULT_LLM = "openai"
```

**Benefits:**
- ✅ Fastest and most accurate
- ✅ Easiest setup
- ❌ Costs money per query
</details>

<details>
<summary><b>🔧 LlamaCpp Setup (Advanced)</b></summary>

```bash
# 1. Install
pip install llama-cpp-python

# 2. Download model
mkdir models
cd models
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.q4_0.bin

# 3. Update config.py
"model_path": "./models/llama-2-7b-chat.q4_0.bin"
```

**Benefits:**
- ✅ Completely offline
- ✅ No internet required after download
- ✅ High performance
- ❌ Larger setup
</details>

## 🛡️ Security Features

- ✅ **SQL injection prevention** - Multi-layer protection
- ✅ **Read-only operations** - No UPDATE/DELETE/DROP allowed
- ✅ **HIPAA-conscious design** - Privacy-focused patterns
- ✅ **Local data processing** - Healthcare data stays on your machine

## 📋 Database Schema

### DX Claims Table
```sql
dx_claims (
    claim_id, patient_id, diagnosis_code, service_date,
    provider_id, provider_specialty, cpt_code, claim_amount
)
```

### RX Prescriptions Table  
```sql
rx_prescriptions (
    prescription_id, patient_id, ndc_code, drug_name, generic_name,
    service_date, provider_id, provider_specialty, quantity, days_supply, copay
)
```

### Providers Table
```sql
providers (
    provider_id, provider_name, specialty, practice_location
)
```

## 🚨 Troubleshooting

### Common Issues

**"Model not found"**
```bash
# Check available models
python config.py

# Verify Ollama is running
ollama list
```

**"API key missing"**
```bash
# Check environment
echo $OPENAI_API_KEY

# Verify .env file
cat .env
```

**"Import errors"**
```bash
# Install missing packages
pip install -r requirements.txt

# For specific models
pip install langchain-ollama      # Ollama
pip install llama-cpp-python     # LlamaCpp
pip install transformers torch   # HuggingFace
```

## 💡 Pro Tips

1. **Start with Ollama** - Best balance of free + performance
2. **Use aliases** - `llm_config="free"` is easier than full config
3. **Check model status** - Run `python config.py` to see what's working
4. **Switch easily** - Change `DEFAULT_LLM` in config.py to try different models
5. **Local = Privacy** - Healthcare data never leaves your machine with local models

## 🤝 Contributing

Got ideas for new models or features? Open an issue or submit a PR!

## 📚 Learning Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Healthcare Data Analytics](https://www.healthcatalyst.com/insights/healthcare-analytics-101)
- [ICD-10 Codes](https://www.icd10data.com/)
- [NDC Drug Codes](https://www.fda.gov/drugs/drug-approvals-and-databases/national-drug-code-directory)

---

**Happy Hacking!** 🚀

*Build something amazing with healthcare AI!*
