# 🏥 Healthcare SQL Agent Hackathon

Query healthcare data with natural language using LangChain! Now supports multiple LLM options including **free local models**.

## 🎯 Training Objectives

This tutorial provides a lightweight orientation to **LLM/Agent development** with practical database querying as the core example. Many hackathon projects will require database interactions, making this a valuable foundational skill.

### 💡 What You'll Learn:
1. **LLM-to-SQL Translation** - How to convert natural language to database queries
2. **Database Security** - Safe query practices with LLMs
3. **Multi-Model Support** - Switching between local and cloud LLMs
4. **Healthcare Data Patterns** - Real-world domain modeling
5. **Configuration Management** - Clean, maintainable setups
6. **Simple vs Complex** - When to choose reliability over sophistication

## 🚀 Quick Setup

```bash
git clone <your-repo-url>
cd sql-agent

# Create virtual environment (recommended)
python -m venv sql-agent-env

# Activate virtual environment
# On macOS/Linux:
source sql-agent-env/bin/activate
# On Windows:
sql-agent-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment (optional - only needed for OpenAI)
cp .env.example .env
# Add your OpenAI API key to .env if using OpenAI

# Create database
python setup_database.py

# Run the agent
python sql_agent.py
```

## 🤖 Supported Models

| Model | Type | Cost | Setup Difficulty | Best For |
|-------|------|------|------------------|----------|
| **Ollama** | Local | Free | ⭐⭐ | **Recommended starter** |
| OpenAI GPT | API | $ | ⭐ | Fast & accurate |

## ⚡ Quick Start Examples
