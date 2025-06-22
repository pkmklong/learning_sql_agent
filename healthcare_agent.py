"""
Healthcare SQL Agent - LLM Agent for querying medical data
Supports Ollama (free/local) and OpenAI (paid/cloud) models
"""

import sqlite3
import re
import os
from typing import Optional, List, Dict, Any, Union
from dotenv import load_dotenv

# Fixed LangChain imports for latest version
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase

# Config should always exist
from config import get_model_config, check_model_ready, list_models, DEFAULT_MODEL

# LLM imports - updated for latest versions
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM

# Load environment variables
load_dotenv()

class SafeSQLDatabase(SQLDatabase):
    """
    A safe wrapper around SQLDatabase that prevents dangerous operations
    """
    
    FORBIDDEN_KEYWORDS = [
        'DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 
        'TRUNCATE', 'REPLACE', 'MERGE', 'UPSERT', 'EXEC', 'EXECUTE'
    ]
    
    def run(self, command: str, fetch: str = "all", **kwargs) -> str:
        """Override run method to add safety checks"""
        # Check for forbidden keywords
        command_upper = command.upper()
        for keyword in self.FORBIDDEN_KEYWORDS:
            if keyword in command_upper:
                return f"Error: '{keyword}' operations are not allowed for security reasons."
        
        # If safe, execute the query (accept any additional kwargs)
        return super().run(command, fetch, **kwargs)

class HackathonSQLAgent:
    """
    Healthcare SQL agent with Ollama (free) and OpenAI (paid) options
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
        
        # Initialize database
        self.db = SafeSQLDatabase.from_uri(f"sqlite:///{db_path}")
        
        # Initialize LLM
        self.llm = self._setup_llm()
        
        # Create agent - simplified approach
        self.agent_executor = create_sql_agent(
            llm=self.llm,
            db=self.db,  # Pass db directly instead of toolkit
            verbose=True,
            agent_type="zero-shot-react-description",
            max_iterations=10,  # Prevent infinite loops
            prefix="""You are a healthcare data analyst. Query medical claims and prescription data safely.
            
            Available tables:
            - dx_claims: diagnosis claims (ICD-10 codes, procedures) 
            - rx_prescriptions: prescriptions (NDC codes, medications)
            - providers: healthcare provider information
            
            Important rules:
            - Only use SELECT queries
            - No UPDATE, DELETE, INSERT, DROP operations allowed
            - Always check table schemas before writing queries
            - Limit results to reasonable amounts
            """
        )
    
    def _setup_llm(self):
        """Initialize the LLM based on validated config"""
        if self.config.type == "ollama":
            return OllamaLLM(
                model=self.config.model,
                base_url=self.config.base_url,
                temperature=self.config.temperature
            )
        
        elif self.config.type == "openai":
            api_key = os.getenv(self.config.api_key_env)
            if not api_key:
                raise ValueError(f"OpenAI requires API key. Set {self.config.api_key_env} in .env file")
            
            return ChatOpenAI(
                model=self.config.model_name,
                temperature=self.config.temperature,
                api_key=api_key
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.config.type}")
    
    def query(self, question: str) -> str:
        """Query the database with natural language"""
        try:
            # Simple, direct approach - let the agent handle everything
            result = self.agent_executor.invoke({"input": question})
            
            # Extract the output from the result
            if isinstance(result, dict):
                return result.get("output", str(result))
            else:
                return str(result)
                
        except Exception as e:
            return f"Query failed: {str(e)}"
    
    def get_schema_info(self) -> str:
        """Get database schema information"""
        return self.db.get_table_info()

def main():
    """Demo the healthcare SQL agent"""
    
    print("🏥 Healthcare SQL Agent")
    print("=" * 30)
    
    # Check if database exists
    db_path = "healthcare_hackathon.db"
    if not os.path.exists(db_path):
        print("❌ Database not found!")
        print("   Run: python setup_database.py")
        print("   Then: python healthcare_agent.py")
        return
    
    print(f"📊 Using database: {db_path}")
    
    # Show available models
    print("\n🤖 Available Models:")
    list_models()
    
    # Initialize agent with default model
    try:
        print(f"\n🚀 Starting agent with {DEFAULT_MODEL}...")
        agent = HackathonSQLAgent(db_path)
        
        # Test query - simple and direct
        print("\n🧪 Testing query...")
        result = agent.query("How many diagnosis claims are there?")
        print(f"✅ Result: {result}")
        
        # Show example queries
        print("\n💡 Example Queries:")
        examples = [
            "How many prescriptions per specialty?",
            "What are the most common diagnosis codes?", 
            "Show me diabetes-related medications",
            "Which providers have the most claims?",
            "Find patients with both diagnosis and prescriptions"
        ]
        
        for i, example in enumerate(examples, 1):
            print(f"   {i}. agent.query('{example}')")
        
        print(f"\n🔧 Interactive Mode:")
        print("   agent = HackathonSQLAgent('{db_path}')")
        print("   result = agent.query('Your question here')")
            
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("   • Database: python setup_database.py")
        print("   • Ollama: ollama pull llama3.2:latest")  
        print("   • OpenAI: Add OPENAI_API_KEY to .env")
        print("   • Config: python config.py")

if __name__ == "__main__":
    main()
