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

# Modern prompt handling
from langchain.prompts import ChatPromptTemplate

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
        
        # Create modern prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a SQL expert for healthcare data analysis.
            Convert natural language questions to SELECT queries.
            
            Database Schema:
            {schema}
            
            IMPORTANT SQLite Syntax Rules:
            - Only use SELECT statements
            - Return clean SQL without backticks or markdown
            - Use SQLite-specific functions (NOT PostgreSQL/MySQL)
            - For dates, use: strftime('%Y-%m', date_column) for year-month
            - For month only: strftime('%m', date_column)
            - For year only: strftime('%Y', date_column)
            - Keep queries simple and efficient
            - Focus on the specific question asked
            
            Common JOIN Patterns:
            - Patient medications: FROM rx_prescriptions r JOIN providers p ON r.provider_id = p.provider_id
            - Provider claims: FROM dx_claims d JOIN providers p ON d.provider_id = p.provider_id
            - Patient journey: FROM dx_claims d JOIN rx_prescriptions r ON d.patient_id = r.patient_id
            
            Key Column Names (use exactly as shown):
            - rx_prescriptions.medication (NOT drug_name)
            - providers.specialty 
            - dx_claims.cost (NOT claim_cost)
            - dx_claims.diagnosis_code
            - All tables have patient_id and provider_id for joins
            
            Example date queries:
            - Monthly grouping: SELECT strftime('%Y-%m', service_date) as month, COUNT(*) FROM table GROUP BY strftime('%Y-%m', service_date)
            - Year grouping: SELECT strftime('%Y', service_date) as year, COUNT(*) FROM table GROUP BY strftime('%Y', service_date)"""),
            
            ("human", "{question}")
        ])
    
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
        """Get database schema information with relationships"""
        schema_info = []
        
        # Get table names
        tables = self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        
        for (table_name,) in tables:
            # Get table info
            table_info = self.cursor.execute(f"PRAGMA table_info({table_name})").fetchall()
            schema_info.append(f"\nTable: {table_name}")
            for col in table_info:
                schema_info.append(f"  - {col[1]} ({col[2]})")
        
        # Add explicit relationship and column information
        schema_info.append(f"\nKey Relationships:")
        schema_info.append(f"  - dx_claims.patient_id = rx_prescriptions.patient_id (same patient)")
        schema_info.append(f"  - dx_claims.provider_id = providers.provider_id (provider info)")
        schema_info.append(f"  - rx_prescriptions.provider_id = providers.provider_id (prescribing provider)")
        
        schema_info.append(f"\nImportant Columns:")
        schema_info.append(f"  - rx_prescriptions.medication (drug name)")
        schema_info.append(f"  - providers.specialty (provider specialty)")
        schema_info.append(f"  - dx_claims.cost (claim cost amount, NOT claim_cost)")
        schema_info.append(f"  - dx_claims.diagnosis_code (medical condition)")
        schema_info.append(f"  - dx_claims.patient_id, rx_prescriptions.patient_id (patient identifiers)")
        schema_info.append(f"  - providers.provider_id (provider identifier)")
        
        return "\n".join(schema_info)
    
    def get_actual_schema(self) -> str:
        """Get the actual database schema by examining the real table structure"""
        try:
            # Get actual column info for each table
            tables = self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            
            schema_details = []
            for (table_name,) in tables:
                schema_details.append(f"\n=== {table_name.upper()} TABLE ===")
                
                # Get column info
                columns = self.cursor.execute(f"PRAGMA table_info({table_name})").fetchall()
                for col in columns:
                    schema_details.append(f"{col[1]} ({col[2]})")
                
                # Show sample data to understand the structure
                try:
                    sample = self.cursor.execute(f"SELECT * FROM {table_name} LIMIT 2").fetchall()
                    if sample:
                        schema_details.append(f"Sample data: {sample[0]}")
                except:
                    pass
            
            return "\n".join(schema_details)
        except Exception as e:
            return f"Error getting schema: {e}"
    
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
            # Use ChatPromptTemplate for structured prompting
            messages = self.prompt_template.format_messages(
                schema=self.schema,
                question=question
            )
            
            # Get SQL from LLM
            response = self.llm.invoke(messages)
            
            # Extract content based on response type
            if hasattr(response, 'content'):
                sql = response.content.strip()
            else:
                sql = str(response).strip()
            
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
    
    def debug_query(self, question: str) -> str:
        """Debug version that shows actual schema"""
        print("🔍 ACTUAL DATABASE SCHEMA:")
        print(self.get_actual_schema())
        return self.query(question)
    
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
        
        # Debug: Show actual schema structure
        print("\n🔍 DEBUGGING: Actual database structure:")
        print(agent.get_actual_schema())
        
        print("\n💡 Example questions you can ask:")
        examples = [
            "How many diagnosis claims are there?",
            "What are the top 3 diagnosis codes?",
            "How many claims per specialty?",
            "What medications are prescribed most?",
            "Which providers have the most claims?",
        ]
        
        for i, example in enumerate(examples, 1):
            print(f"   {i}. {example}")
        
        # Interactive mode
        print("\n" + "="*50)
        print("🎮 ASK YOUR QUESTIONS!")
        print("Type 'quit', 'exit', or 'done' to stop")
        print("="*50)
        
        while True:
            try:
                # Get user input
                user_question = input("\n❓ Your question: ").strip()
                
                # Check for exit commands
                if user_question.lower() in ['quit', 'exit', 'done', 'q']:
                    print("👋 Thanks for using the healthcare agent!")
                    break
                
                # Skip empty questions
                if not user_question:
                    continue
                
                # Process the question
                print("🤖 Thinking...")
                result = agent.query(user_question)
                print(f"✅ {result}")
                
            except KeyboardInterrupt:
                print("\n👋 Exiting... Thanks for using the healthcare agent!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        # Clean up
        agent.close()
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")

if __name__ == "__main__":
    main()
