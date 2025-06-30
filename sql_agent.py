import sqlite3
import os
from dotenv import load_dotenv

# Import your config system
from config import get_model_config, check_model_ready, DEFAULT_MODEL

# LLM imports  
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor

# Load environment variables
load_dotenv()

# Setup
DB_PATH = "healthcare_hackathon.db"

def setup_llm(model_name=None):
    """Initialize LLM using your config system"""
    if model_name is None:
        model_name = DEFAULT_MODEL
    
    # Get validated model config
    config = get_model_config(model_name)
    
    # Check if model is ready
    ready, status = check_model_ready(model_name)
    if not ready:
        raise ValueError(f"Model '{model_name}' not ready: {status}")
    
    # Initialize LLM based on config
    if config.type == "ollama":
        return OllamaLLM(
            model=config.model,
            base_url=config.base_url,
            temperature=0.1
        )
    elif config.type == "openai":
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ValueError(f"OpenAI requires API key. Set {config.api_key_env} in .env file")
        
        return ChatOpenAI(
            model=config.model_name,
            temperature=0.1,
            api_key=api_key
        )
    else:
        raise ValueError(f"Unknown model type: {config.type}")

@tool
def execute_sql_query(sql_query: str) -> str:
    """Execute a SQL query on the healthcare database and return results."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Safety check - only allow SELECT
        if not sql_query.strip().upper().startswith('SELECT'):
            return "Error: Only SELECT statements are allowed."
        
        cursor.execute(sql_query)
        results = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        
        if not results:
            return "No results found."
        
        # Format results nicely
        if len(results) == 1 and len(results[0]) == 1:
            return f"Result: {results[0][0]}"
        
        output = f"Found {len(results)} rows:\n"
        output += " | ".join(columns) + "\n"
        output += "-" * 50 + "\n"
        
        for row in results[:10]:  # Show first 10 rows
            output += " | ".join(str(x) for x in row) + "\n"
        
        if len(results) > 10:
            output += f"... and {len(results) - 10} more rows"
        
        return output
        
    except Exception as e:
        return f"SQL Error: {e}"
    finally:
        conn.close()

def get_schema():
    """Get database schema information"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    schema_info = []
    tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    
    for (table_name,) in tables:
        table_info = cursor.execute(f"PRAGMA table_info({table_name})").fetchall()
        schema_info.append(f"\nTable: {table_name}")
        for col in table_info:
            schema_info.append(f"  - {col[1]} ({col[2]})")
    
    conn.close()
    return "\n".join(schema_info)

# Initialize LLM using your config system
llm = setup_llm()

# Agent prompt
prompt = ChatPromptTemplate.from_template("""You are a healthcare data analyst. You have access to a healthcare database with this schema:

{schema}

When users ask questions about the data, write SQL queries to get the information and execute them using the execute_sql_query tool.

RULES:
1. Use ONLY columns that exist in the schema above
2. Use SQLite syntax
3. SELECT statements only
4. If you can't answer with available columns, explain what's missing

Common diagnosis codes: E1140 (diabetes), I2510 (heart disease), J449 (COPD), M545 (back pain), F329 (depression)

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}""")

# Create agent
tools = [execute_sql_query]
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=3)

def ask(question):
    """Ask a question about the healthcare data"""
    print(f"Question: {question}")
    try:
        response = agent_executor.invoke({
            "input": question,
            "schema": get_schema(),
            "tools": [tool.name + ": " + tool.description for tool in tools],
            "tool_names": [tool.name for tool in tools]
        })
        print(f"Answer: {response['output']}")
    except Exception as e:
        print(f"Error: {e}")
    print("-" * 60)

# Test queries
ask("How many unique patients do we have?")
ask("What are the top 5 most expensive claims?")
ask("Which specialty has the most patients?")
ask("Show me all diabetes patients and their medications")
ask("What's the average copay by medication type?")
