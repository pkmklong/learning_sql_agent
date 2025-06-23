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

# Create database
python setup_database.py

# Run the agent
python healthcare_agent.py
```

## 🤖 Supported Models

| Model | Type | Cost | Setup Difficulty | Best For |
|-------|------|------|------------------|----------|
| **Ollama** | Local | Free | ⭐⭐ | **Recommended starter** |
| OpenAI GPT | API | $ | ⭐ | Fast & accurate |

## ⚡ Quick Start Examples

```python
# Use default model (set in config.py)
agent = HackathonSQLAgent("healthcare_hackathon.db")

# Switch models easily
agent = HackathonSQLAgent("healthcare_hackathon.db", "ollama")
agent = HackathonSQLAgent("healthcare_hackathon.db", "openai")

# Check available models and their status
python config.py
```

## 🆓 Free Local Setup (Recommended)

### Ollama Setup
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model (matches config.py default)
ollama pull llama3.2

# Start server
ollama serve

# Update config.py if needed
DEFAULT_MODEL = "ollama"
```

## 🧠 How It Works: Simple and Reliable

We've built this to be **simple and reliable** rather than complex. Here's how it works:

### **🔄 The Simple Process**

When you ask: *"How many diabetes patients are there?"*

1. **🤖 LLM Prompt**: Convert your question to SQL using the database schema
2. **⚡ Safety Check**: Ensure only SELECT queries (no dangerous operations)
3. **🔍 Execute**: Run the SQL directly against the database
4. **📊 Format**: Return clean, readable results

### **🛡️ Safety First**

- **Read-only queries** - Only SELECT statements allowed
- **No dangerous operations** - Blocks DROP, DELETE, UPDATE, etc.
- **Direct execution** - No complex agent loops that can get confused
- **Visible SQL** - You can see exactly what queries are generated

### **🎯 Why This Approach Works**

- **Reliability over complexity** - Simple systems are more predictable
- **Fast development** - No debugging complex agent workflows  
- **Easy to understand** - Clear path from question → SQL → result
- **Production-ready** - This pattern works in real applications

This demonstrates a key principle: **sometimes the simplest solution is the best solution**, especially when you need something that reliably works under pressure! 🎯

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
├── setup_database.py        # Database creation script
├── healthcare_agent.py      # Main agent code
└── healthcare_hackathon.db  # Auto-generated database
```

## 🏗️ Architecture Choices (Learning-Focused)

This codebase prioritizes **LEARNING** and **RAPID PROTOTYPING** over production patterns:

### ✅ Chosen for Learning:
- **Single file agent class** (easy to understand)
- **Simple configuration** (config.py with Pydantic) 
- **Direct imports** (clear dependencies)
- **Minimal abstraction** (see exactly what's happening)
- **SQLite database** (no external dependencies)

### ❌ Trade-offs Made:
- No dependency injection (harder to test)
- Basic error handling (production needs more robust)
- No logging framework (just print statements)
- No async support (production agents often async)
- Limited observability (no tracing/monitoring)

## 🚀 Production Evolution Path

To evolve this codebase toward production best practices:

<details>
<summary><b>1. Dependency Injection & Testing</b></summary>

```python
class SQLAgent:
    def __init__(self, llm: BaseLLM, db: BaseDatabase, safety: SafetyChecker):
        # Inject dependencies for easier testing

class TestSQLAgent:
    def test_query_safety(self):
        mock_llm = MockLLM()
        mock_db = MockDatabase() 
        agent = SQLAgent(mock_llm, mock_db, safety_checker)
```
</details>

<details>
<summary><b>2. Structured Logging & Observability</b></summary>

```python
import structlog

logger = structlog.get_logger()

def query(self, question: str):
    logger.info("query_started", question=question, model=self.model_type)
    # ... query logic
    logger.info("query_completed", duration=elapsed, tokens_used=tokens)
```
</details>

<details>
<summary><b>3. Async Support & Concurrency</b></summary>

```python
async def query_async(self, question: str) -> str:
    async with self.llm_session() as session:
        result = await session.arun(question)
    return result
```
</details>

<details>
<summary><b>4. Error Handling & Retries</b></summary>

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential())
async def robust_query(self, question: str):
    # Automatic retries with exponential backoff
```
</details>

<details>
<summary><b>5. Metrics & Monitoring</b></summary>

```python
from prometheus_client import Counter, Histogram

query_counter = Counter('agent_queries_total', ['status', 'model'])
query_duration = Histogram('agent_query_duration_seconds')
```
</details>

## 🔄 Alternative Frameworks & Approaches

### 1. LangGraph (Current Best Practice 2025)
```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Define workflow state
class QueryState(TypedDict):
    question: str
    sql_query: str
    results: str
    
# Create workflow graph
workflow = StateGraph(QueryState)
workflow.add_node("analyze_question", analyze_question_node)
workflow.add_node("generate_sql", generate_sql_node)
workflow.add_node("validate_sql", validate_sql_node)
workflow.add_node("execute_query", execute_query_node)

# Better control flow, error handling, human-in-loop
```

### 2. LangChain Chains (Simpler, More Direct)
```python
from langchain.chains import create_sql_query_chain

# More direct, less flexible than agents
chain = create_sql_query_chain(llm, db)
result = chain.invoke({"question": "How many patients?"})
```

### 3. Custom Frameworks
```python
# Semantic Kernel (Microsoft)
import semantic_kernel as sk

# AutoGen (Multi-agent conversations)
from autogen import AssistantAgent, UserProxyAgent

# CrewAI (Role-based agents)
from crewai import Agent, Task, Crew
```

## 🎯 Choosing the Right Approach

| Use Case | Recommended Approach |
|----------|---------------------|
| **Hackathons** | This tutorial setup (rapid prototyping) |
| **Simple Projects** | LangChain Chains |
| **Production Systems** | LangGraph (complex workflows, human oversight) |
| **Custom Requirements** | Direct API calls (maximum control) |
| **Learning** | Start here → LangGraph → Custom frameworks |

## 🔧 Extending the Codebase

### 📁 Suggested Directory Structure for Extensions

<details>
<summary><b>Basic Extension Structure</b></summary>

```
healthcare-sql-agent/
├── core/                     # Core agent functionality
│   ├── __init__.py
│   ├── agent.py             # Main agent class
│   ├── database.py          # Database utilities
│   ├── safety.py            # Security wrappers
│   └── config.py            # Configuration management
│
├── models/                   # LLM model integrations
│   ├── __init__.py
│   ├── base.py              # Abstract base classes
│   ├── ollama.py            # Ollama integration
│   ├── openai.py            # OpenAI integration
│   └── custom.py            # Custom model support
│
├── data/                     # Data management
│   ├── __init__.py
│   ├── schemas/             # Database schemas
│   ├── generators/          # Data generation tools
│   └── healthcare_hackathon.db
│
├── tools/                    # Agent tools and utilities
│   ├── __init__.py
│   ├── sql_tools.py         # SQL query tools
│   ├── validation.py        # Query validation
│   └── formatting.py       # Result formatting
│
├── examples/                 # Usage examples
│   ├── basic_usage.py
│   ├── advanced_queries.py
│   └── custom_domains.py
│
├── tests/                    # Test suite
│   ├── __init__.py
│   ├── test_agent.py
│   ├── test_safety.py
│   └── test_integration.py
│
└── scripts/                  # Setup and utility scripts
    ├── setup_database.py
    ├── benchmark_models.py
    └── generate_data.py
```
</details>

<details>
<summary><b>Production-Ready Structure</b></summary>

```
healthcare-sql-agent/
├── src/
│   ├── healthcare_agent/
│   │   ├── __init__.py
│   │   ├── core/            # Core business logic
│   │   │   ├── agent.py
│   │   │   ├── database.py
│   │   │   └── safety.py
│   │   ├── models/          # LLM integrations
│   │   ├── tools/           # Agent tools
│   │   ├── utils/           # Utilities
│   │   └── config/          # Configuration
│   │       ├── settings.py
│   │       └── models.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── performance/
│
├── docs/
│   ├── api/
│   ├── tutorials/
│   └── deployment/
│
├── deployment/
│   ├── docker/
│   ├── kubernetes/
│   └── terraform/
│
├── monitoring/
│   ├── prometheus/
│   └── grafana/
│
└── scripts/
    ├── setup/
    ├── maintenance/
    └── migration/
```
</details>

### 🛠️ Extension Patterns

#### 1. **Different Data Sources**
```python
# Replace SQLDatabase with:
src/healthcare_agent/datasources/
├── sql.py          # PostgreSQL/MySQL
├── nosql.py        # MongoDB  
├── api.py          # REST/GraphQL endpoints
├── vector.py       # Pinecone, Weaviate
└── warehouse.py    # Snowflake, BigQuery
```

#### 2. **Different Interaction Patterns**
```python
# Beyond Q&A:
src/healthcare_agent/interfaces/
├── chat.py         # Interactive chat
├── api.py          # REST API endpoints
├── batch.py        # Batch processing
├── streaming.py    # Real-time streaming
└── visualization.py # Chart generation
```

## 📖 Recommended Learning Path

1. **Complete this tutorial** - understand the basics
2. **Modify the healthcare domain** - try different medical scenarios
3. **Add new safety rules** - practice security thinking
4. **Try different models** - compare Ollama vs OpenAI performance
5. **Implement LangGraph version** - learn modern patterns
6. **Add observability** - practice production thinking
7. **Build your own domain** - apply to your specific use case

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
# Check available models and their status
python config.py

# Verify Ollama is running and model is available
ollama list
```

**"Database not found"**
```bash
# Create database
python setup_database.py
```

**"Import errors"**
```bash
# Install missing packages
pip install -r requirements.txt
```

**"Ollama connection failed"**
```bash
# Make sure Ollama is running
ollama serve

# Check if model is pulled
ollama pull llama3.2:latest
```

## 💡 Pro Tips

1. **Start with Ollama** - Best balance of free + performance
2. **Use simple queries first** - Build complexity gradually
3. **Check model status** - Run `python config.py` to see model availability and setup
4. **Switch easily** - Change `DEFAULT_MODEL` in config.py to try different models
5. **Local = Privacy** - Healthcare data never leaves your machine with local models
6. **Consistent model names** - Use `llama3.2:latest` to match config.py defaults

## 🤝 Contributing to Team Knowledge

After the hackathon, consider:
- Documenting patterns you discovered
- Sharing model performance comparisons
- Contributing safety improvements
- Building domain-specific extensions

## 📚 Learning Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Tutorials](https://langchain-ai.github.io/langgraph/)
- [Healthcare Data Analytics](https://www.healthcatalyst.com/insights/healthcare-analytics-101)
- [ICD-10 Codes](https://www.icd10data.com/)
- [NDC Drug Codes](https://www.fda.gov/drugs/drug-approvals-and-databases/national-drug-code-directory)

---

**Remember: This tutorial is your starting point, not your destination!**

*Build something amazing with healthcare AI!* 🚀
