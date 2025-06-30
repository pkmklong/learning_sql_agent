import sqlite3
import os
from dotenv import load_dotenv

# Import your config system
from config import get_model_config, check_model_ready, DEFAULT_MODEL

# LLM imports  
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

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

# Simple prompt template for SQL generation
prompt = ChatPromptTemplate.from_template("""
You are a healthcare data analyst. Here's the database schema:

{schema}

Convert this question to a SQL query:
"{question}"

Rules:
- Use SQLite syntax (LIMIT not TOP)
- Only SELECT statements
- Use only columns that exist in the schema

Return only the SQL query, no explanation.
""")

def ask(question):
    """Ask a question about the healthcare data"""
    print(f"Question: {question}")
    
    try:
        # Generate SQL
        response = llm.invoke(prompt.format(schema=get_schema(), question=question))
        sql = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        
        # Clean up the SQL
        sql = sql.replace('```sql', '').replace('```', '').strip()
        
        print(f"Generated SQL: {sql}")
        
        # Execute SQL
        result = execute_sql_query(sql)
        print(f"Answer: {result}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    print("-" * 60)

# Test queries
ask("How many unique patients do we have?")
ask("What are the top 5 most expensive claims?")
ask("Which specialty has the most claims?")
ask("Show me all diabetes patients and their medications")
ask("What's the average copay by medication type?")
