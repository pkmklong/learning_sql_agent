"""
Healthcare SQL Agent - Simple and Reliable
"""

import sqlite3
import os
from dotenv import load_dotenv

# Config should always exist
from config import get_model_config, check_model_ready, list_models, DEFAULT_MODEL

# LLM imports
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM

# Load environment variables
load_dotenv()

class SimpleHealthcareAgent:
    """
    Simple healthcare agent that builds SQL and executes it safely
    """
    
    def __init__(self, db_path: str, model: str = None):
        self.db_path = db_path
        
        # Use default model if none specified
        if model is None:
            model = DEFAULT_MODEL
        
        # Get validated model config
        self.config = get_model_config(model)
        
        # Check if model is ready
        ready, status = check_model_ready(model)
        if not ready:
            raise ValueError(f"Model '{model}' not ready: {status}")
        
        # Initialize database connection
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        
        # Initialize LLM
        self.llm = self._setup_llm()
        
        # Get schema info once
        self.schema = self._get_schema()
    
    def _setup_llm(self):
        """Initialize the LLM based on validated config"""
        if self.config.type == "ollama":
            return OllamaLLM(
                model=self.config.model,
                base_url=self.config.base_url,
                temperature=0.1
            )
        
        elif self.config.type == "openai":
            api_key = os.getenv(self.config.api_key_env)
            if not api_key:
                raise ValueError(f"OpenAI requires API key. Set {self.config.api_key_env} in .env file")
            
            return ChatOpenAI(
                model=self.config.model_name,
                temperature=0.1,
                api_key=api_key
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.config.type}")
    
    def _get_schema(self) -> str:
        """Get database schema information"""
        schema_info = []
        
        # Get table names
        tables = self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        
        for (table_name,) in tables:
            # Get table info
            table_info = self.cursor.execute(f"PRAGMA table_info({table_name})").fetchall()
            schema_info.append(f"\nTable: {table_name}")
            for col in table_info:
                schema_info.append(f"  - {col[1]} ({col[2]})")
        
        return "\n".join(schema_info)
    
    def _is_safe_query(self, sql: str) -> bool:
        """Check if SQL query is safe (SELECT only)"""
        sql_upper = sql.upper().strip()
        
        # Must start with SELECT
        if not sql_upper.startswith('SELECT'):
            return False
        
        # Check for forbidden keywords
        forbidden = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE']
        for keyword in forbidden:
            if keyword in sql_upper:
                return False
        
        return True
    
    def query(self, question: str) -> str:
        """Query the database with natural language"""
        try:
            # Create a simple prompt
            prompt = f"""
You are a SQL expert. Convert this question to a SELECT query for a healthcare database.

Database Schema:
{self.schema}

Question: {question}

Rules:
- Only use SELECT statements
- Return just the SQL query, no explanation
- Use proper SQLite syntax
- Don't use backticks or markdown

SQL Query:"""
            
            # Get SQL from LLM
            response = self.llm.invoke(prompt)
            sql = response.strip()
            
            # Clean up the response (remove common formatting)
            sql = sql.replace('```sql', '').replace('```', '').strip()
            
            # Safety check
            if not self._is_safe_query(sql):
                return f"Unsafe query detected. Only SELECT statements allowed.\nGenerated: {sql}"
            
            # Execute query
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            
            # Format results nicely
            if not results:
                return "No results found."
            
            if len(results) == 1 and len(results[0]) == 1:
                # Single value result
                return f"Result: {results[0][0]}"
            
            # Multiple results - format as text
            result_text = f"Found {len(results)} results:\n"
            for i, row in enumerate(results[:10]):  # Limit to 10 rows
                result_text += f"{i+1}. {row}\n"
            
            if len(results) > 10:
                result_text += f"... and {len(results) - 10} more rows"
            
            return result_text
            
        except sqlite3.Error as e:
            return f"SQL Error: {e}\nGenerated SQL: {sql}"
        except Exception as e:
            return f"Error: {e}"
    
    def get_tables(self) -> str:
        """Get list of available tables"""
        tables = self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        return "Available tables: " + ", ".join([t[0] for t in tables])
    
    def close(self):
        """Close database connection"""
        self.conn.close()

def main():
    """Demo the simple healthcare agent"""
    
    print("🏥 Simple Healthcare SQL Agent")
    print("=" * 35)
    
    # Check if database exists
    db_path = "healthcare_hackathon.db"
    if not os.path.exists(db_path):
        print("❌ Database not found!")
        print("   Run: python setup_database.py")
        return
    
    print(f"📊 Using database: {db_path}")
    
    # Show available models
    print("\n🤖 Available Models:")
    list_models()
    
    try:
        print(f"\n🚀 Starting simple agent with {DEFAULT_MODEL}...")
        agent = SimpleHealthcareAgent(db_path)
        
        # Show tables
        print(f"\n📋 {agent.get_tables()}")
        
        # Test queries
        test_queries = [
            "How many diagnosis claims are there?",
            "How many prescriptions are there?",
            "How many providers are there?"
        ]
        
        for query in test_queries:
            print(f"\n❓ {query}")
            result = agent.query(query)
            print(f"✅ {result}")
        
        print("\n💡 Try these queries:")
        examples = [
            "What are the top 3 diagnosis codes?",
            "How many claims per specialty?",
            "What medications are prescribed most?",
            "Which providers have the most claims?",
        ]
        
        for i, example in enumerate(examples, 1):
            print(f"   {i}. agent.query('{example}')")
        
        print(f"\n🔧 Interactive Mode:")
        print("   agent = SimpleHealthcareAgent('{db_path}')")
        print("   result = agent.query('Your question here')")
        print("   agent.close()  # When done")
        
        # Clean up
        agent.close()
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")

if __name__ == "__main__":
    main()
