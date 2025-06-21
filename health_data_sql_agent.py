import sqlite3
import re
import os
from typing import Optional, List, Dict, Any, Union
from dotenv import load_dotenv
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.agents import AgentExecutor
from langchain.schema import AgentAction, AgentFinish
from langchain.callbacks.base import BaseCallbackHandler

# Import config
try:
    from config import get_llm_config, validate_llm_setup, list_available_models, print_setup_instructions, DEFAULT_LLM
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("⚠️  config.py not found. Using basic configuration.")

# Import different LLM options
try:
    from langchain.llms import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from langchain.llms import Ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from langchain.llms import LlamaCpp
    LLAMACPP_AVAILABLE = True
except ImportError:
    LLAMACPP_AVAILABLE = False

try:
    from langchain_community.llms import HuggingFacePipeline
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

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
    
    def run(self, command: str, fetch: str = "all") -> str:
        """Override run method to add safety checks"""
        # Check for forbidden keywords
        command_upper = command.upper()
        for keyword in self.FORBIDDEN_KEYWORDS:
            if keyword in command_upper:
                return f"Error: '{keyword}' operations are not allowed for security reasons."
        
        # Additional regex check for common dangerous patterns
        dangerous_patterns = [
            r'DROP\s+TABLE',
            r'DELETE\s+FROM',
            r'UPDATE\s+.*SET',
            r'INSERT\s+INTO',
            r'ALTER\s+TABLE',
            r'CREATE\s+TABLE'
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, command_upper):
                return "Error: This type of operation is not allowed."
        
        # If safe, execute the query
        return super().run(command, fetch)

def create_sample_database(db_path: str = "healthcare_hackathon.db"):
    """Create a sample healthcare database for the hackathon"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create diagnosis claims table (DX)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS dx_claims (
            claim_id INTEGER PRIMARY KEY,
            patient_id INTEGER NOT NULL,
            diagnosis_code TEXT NOT NULL,
            service_date DATE NOT NULL,
            provider_id INTEGER NOT NULL,
            provider_specialty TEXT NOT NULL,
            cpt_code TEXT,
            claim_amount DECIMAL(10,2)
        )
    ''')
    
    # Create prescription table (RX)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS rx_prescriptions (
            prescription_id INTEGER PRIMARY KEY,
            patient_id INTEGER NOT NULL,
            ndc_code TEXT NOT NULL,
            drug_name TEXT NOT NULL,
            generic_name TEXT NOT NULL,
            service_date DATE NOT NULL,
            provider_id INTEGER NOT NULL,
            provider_specialty TEXT NOT NULL,
            quantity INTEGER,
            days_supply INTEGER,
            copay DECIMAL(8,2)
        )
    ''')
    
    # Create providers reference table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS providers (
            provider_id INTEGER PRIMARY KEY,
            provider_name TEXT NOT NULL,
            specialty TEXT NOT NULL,
            practice_location TEXT
        )
    ''')
    
    # Insert sample diagnosis claims data
    dx_claims_data = [
        (1, 1001, 'E1140', '2024-03-15', 201, 'Endocrinology', '99213', 125.50),
        (2, 1002, 'I2510', '2024-03-16', 202, 'Cardiology', '93000', 89.25),
        (3, 1003, 'J449', '2024-03-17', 203, 'Pulmonology', '94010', 156.75),
        (4, 1001, 'Z7901', '2024-03-18', 204, 'Oncology', '77067', 245.00),
        (5, 1004, 'M545', '2024-03-19', 205, 'Orthopedics', '20610', 178.30),
        (6, 1005, 'F329', '2024-03-20', 206, 'Psychiatry', '90834', 95.00),
        (7, 1002, 'N183', '2024-03-21', 207, 'Nephrology', '36415', 67.80),
        (8, 1006, 'K219', '2024-03-22', 208, 'Gastroenterology', '43235', 312.45),
        (9, 1003, 'G309', '2024-03-23', 209, 'Neurology', '95860', 189.60),
        (10, 1007, 'L309', '2024-03-24', 210, 'Dermatology', '11100', 98.25),
        (11, 1008, 'H269', '2024-03-25', 211, 'Ophthalmology', '92004', 145.75),
        (12, 1009, 'N390', '2024-03-26', 212, 'Urology', '51798', 234.90)
    ]
    
    # Insert sample prescription data
    rx_prescriptions_data = [
        (1, 1001, '0088221947', 'Metformin HCl 500mg', 'Metformin', '2024-03-15', 201, 'Endocrinology', 60, 30, 10.00),
        (2, 1002, '0003084221', 'Lisinopril 10mg', 'Lisinopril', '2024-03-16', 202, 'Cardiology', 30, 30, 5.00),
        (3, 1003, '0173068220', 'Albuterol Sulfate 90mcg', 'Albuterol', '2024-03-17', 203, 'Pulmonology', 1, 30, 15.25),
        (4, 1004, '0093051556', 'Ibuprofen 600mg', 'Ibuprofen', '2024-03-19', 205, 'Orthopedics', 60, 10, 8.50),
        (5, 1005, '0378603093', 'Sertraline 50mg', 'Sertraline', '2024-03-20', 206, 'Psychiatry', 30, 30, 12.75),
        (6, 1002, '0054327599', 'Furosemide 40mg', 'Furosemide', '2024-03-21', 207, 'Nephrology', 30, 30, 6.80),
        (7, 1006, '0093515301', 'Omeprazole 20mg', 'Omeprazole', '2024-03-22', 208, 'Gastroenterology', 30, 30, 9.45),
        (8, 1003, '0093832568', 'Gabapentin 300mg', 'Gabapentin', '2024-03-23', 209, 'Neurology', 90, 30, 18.90),
        (9, 1007, '0168013631', 'Hydrocortisone Cream 1%', 'Hydrocortisone', '2024-03-24', 210, 'Dermatology', 1, 14, 11.25),
        (10, 1008, '0065015015', 'Latanoprost 0.005%', 'Latanoprost', '2024-03-25', 211, 'Ophthalmology', 1, 30, 45.60),
        (11, 1009, '0093511205', 'Tamsulosin 0.4mg', 'Tamsulosin', '2024-03-26', 212, 'Urology', 30, 30, 14.30),
        (12, 1001, '0088019747', 'Insulin Glargine 100units/ml', 'Insulin Glargine', '2024-03-28', 201, 'Endocrinology', 1, 30, 85.75)
    ]
    
    # Insert provider reference data
    providers_data = [
        (201, 'Dr. Sarah Chen', 'Endocrinology', 'Downtown Medical Center'),
        (202, 'Dr. Michael Rodriguez', 'Cardiology', 'Heart Care Clinic'),
        (203, 'Dr. Jennifer Kim', 'Pulmonology', 'Respiratory Health Center'),
        (204, 'Dr. David Thompson', 'Oncology', 'Cancer Treatment Center'),
        (205, 'Dr. Lisa Wang', 'Orthopedics', 'Bone & Joint Specialists'),
        (206, 'Dr. Robert Johnson', 'Psychiatry', 'Mental Health Associates'),
        (207, 'Dr. Maria Garcia', 'Nephrology', 'Kidney Care Center'),
        (208, 'Dr. James Wilson', 'Gastroenterology', 'Digestive Health Clinic'),
        (209, 'Dr. Emily Davis', 'Neurology', 'Neurological Institute'),
        (210, 'Dr. Andrew Miller', 'Dermatology', 'Skin Care Specialists'),
        (211, 'Dr. Rachel Brown', 'Ophthalmology', 'Eye Care Center'),
        (212, 'Dr. Kevin Lee', 'Urology', 'Urological Associates')
    ]
    
    cursor.executemany('INSERT OR REPLACE INTO dx_claims VALUES (?, ?, ?, ?, ?, ?, ?, ?)', dx_claims_data)
    cursor.executemany('INSERT OR REPLACE INTO rx_prescriptions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', rx_prescriptions_data)
    cursor.executemany('INSERT OR REPLACE INTO providers VALUES (?, ?, ?, ?)', providers_data)
    
    conn.commit()
    conn.close()
    print(f"Sample database created at {db_path}")

class HackathonSQLAgent:
    """
    A lightweight SQL agent for hackathon training with config-managed LLMs
    """
    
    def __init__(self, db_path: str, llm_config: Union[str, Dict[str, Any]] = None):
        self.db_path = db_path
        
        # Get LLM configuration
        if llm_config is None:
            if CONFIG_AVAILABLE:
                llm_config = DEFAULT_LLM
            else:
                llm_config = "openai"  # fallback
        
        # Handle string config (model name) vs dict config (custom)
        if isinstance(llm_config, str):
            if CONFIG_AVAILABLE:
                self.config = get_llm_config(llm_config)
            else:
                # Fallback basic config
                self.config = {"type": llm_config}
        else:
            self.config = llm_config
        
        # Validate setup
        if CONFIG_AVAILABLE:
            is_valid, error = validate_llm_setup(self.config)
            if not is_valid:
                print(f"❌ LLM setup issue: {error}")
                print_setup_instructions(self.config.get("type"))
                raise ValueError(f"LLM setup failed: {error}")
        
        # Initialize the safe database connection
        self.db = SafeSQLDatabase.from_uri(f"sqlite:///{db_path}")
        
        # Initialize LLM based on config
        self.llm = self._initialize_llm_from_config()
        
        # Create SQL toolkit
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        
        # Create the agent with healthcare-specific context
        self.agent_executor = create_sql_agent(
            llm=self.llm,
            toolkit=self.toolkit,
            verbose=True,
            agent_type="zero-shot-react-description",
            prefix="""You are a healthcare data analyst agent designed to query medical claims and prescription data.
            
            IMPORTANT HEALTHCARE CONTEXT:
            - dx_claims table contains diagnosis claims with ICD-10 codes (without decimals)
            - rx_prescriptions table contains prescription data with NDC codes
            - providers table contains provider reference information
            - Always consider patient privacy and data sensitivity
            - Focus on aggregate analysis rather than individual patient details when possible
            
            SAFETY RULES:
            - You can ONLY perform SELECT queries
            - NO UPDATE, DELETE, INSERT, DROP, or ALTER operations are allowed
            - Be careful with patient data - aggregate when possible
            """
        )
    
    def _initialize_llm_from_config(self):
        """Initialize LLM based on configuration"""
        llm_type = self.config.get("type", "openai")
        
        if llm_type == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI not available. Install with: pip install langchain-openai")
            
            api_key = os.getenv(self.config.get("api_key_env", "OPENAI_API_KEY"))
            if not api_key:
                raise ValueError(f"Missing {self.config.get('api_key_env', 'OPENAI_API_KEY')} environment variable")
            
            return OpenAI(
                openai_api_key=api_key,
                model_name=self.config.get("model_name", "gpt-3.5-turbo"),
                temperature=self.config.get("temperature", 0),
                max_tokens=self.config.get("max_tokens", 2000)
            )
        
        elif llm_type == "ollama":
            if not OLLAMA_AVAILABLE:
                raise ImportError("Ollama not available. Install with: pip install langchain-ollama")
            
            return Ollama(
                model=self.config.get("model", "llama3.1"),
                base_url=self.config.get("base_url", "http://localhost:11434"),
                temperature=self.config.get("temperature", 0)
            )
        
        elif llm_type == "llamacpp":
            if not LLAMACPP_AVAILABLE:
                raise ImportError("LlamaCpp not available. Install with: pip install llama-cpp-python")
            
            model_path = self.config.get("model_path")
            if not model_path or not os.path.exists(model_path):
                raise ValueError(f"Model file not found: {model_path}")
            
            return LlamaCpp(
                model_path=model_path,
                temperature=self.config.get("temperature", 0),
                max_tokens=self.config.get("max_tokens", 2000),
                n_ctx=self.config.get("n_ctx", 4096),
                verbose=self.config.get("verbose", False)
            )
        
        elif llm_type == "huggingface":
            if not HUGGINGFACE_AVAILABLE:
                raise ImportError("HuggingFace not available. Install with: pip install transformers torch")
            
            model_name = self.config.get("model_name", "microsoft/DialoGPT-medium")
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(model_name)
                
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=self.config.get("max_tokens", 512),
                    temperature=self.config.get("temperature", 0.1)
                )
                
                return HuggingFacePipeline(pipeline=pipe)
            except Exception as e:
                raise RuntimeError(f"Failed to load HuggingFace model {model_name}: {e}")
        
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")
    
    def get_config_info(self) -> str:
        """Get information about current configuration"""
        return f"Using {self.config.get('type')} with config: {self.config}"
    
    def get_schema_info(self) -> str:
        """Get database schema information"""
        return self.db.get_table_info()
    
    def query(self, question: str) -> str:
        """
        Query the database with natural language
        """
        try:
            # Add context about the healthcare database schema
            context = f"""
            You are querying a healthcare claims database with the following schema:
            {self.get_schema_info()}
            
            HEALTHCARE DATA CONTEXT:
            - dx_claims: Diagnosis claims with ICD-10 codes, CPT procedures, and provider info
            - rx_prescriptions: Prescription data with NDC codes, drug names, and pharmacy info  
            - providers: Reference table for provider details and specialties
            
            IMPORTANT SAFETY RULES:
            - You can ONLY perform SELECT queries
            - NO UPDATE, DELETE, INSERT, DROP, or ALTER operations are allowed
            - Focus on answering the user's question with read-only operations
            - Consider patient privacy - use aggregate data when possible
            
            User question: {question}
            """
            
            result = self.agent_executor.run(context)
            return result
            
        except Exception as e:
            return f"Error processing query: {str(e)}"
    
    def run_direct_sql(self, sql_query: str) -> str:
        """
        Run a direct SQL query (with safety checks)
        """
        try:
            result = self.db.run(sql_query)
            return result
        except Exception as e:
            return f"Error executing SQL: {str(e)}"

# Example usage and setup
def main():
    """Main function to demonstrate the SQL agent with config management"""
    
    print("🏥 Healthcare SQL Agent Hackathon")
    print("=" * 50)
    
    # Create sample database
    print("📊 Creating healthcare database...")
    create_sample_database()
    
    if CONFIG_AVAILABLE:
        print("\n🤖 Available Models:")
        list_available_models()
        
        print(f"\n🎯 Using default model: {DEFAULT_LLM}")
        print("   (Change DEFAULT_LLM in config.py to switch)")
        
        # Initialize with default config
        try:
            agent = HackathonSQLAgent("healthcare_hackathon.db")
            print(f"✅ Agent initialized: {agent.get_config_info()}")
            
            # Test a simple query
            print("\n🧪 Testing query...")
            result = agent.query("How many diagnosis claims are there?")
            print(f"Result: {result}")
            
        except Exception as e:
            print(f"❌ Agent initialization failed: {e}")
            print("\n💡 Try these solutions:")
            print("1. Check config.py settings")
            print("2. Run setup instructions for your chosen model")
            print("3. Use 'python config.py' to see model status")
    
    else:
        print("\n⚠️  config.py not found - using basic setup")
        print("Create config.py for advanced model management")

def demo_model_switching():
    """Demonstrate how to switch between models"""
    if not CONFIG_AVAILABLE:
        print("config.py required for model switching demo")
        return
    
    print("\n🔄 Model Switching Demo")
    print("=" * 30)
    
    models_to_try = ["free", "fast", "offline"]  # Aliases from config
    
    for model_alias in models_to_try:
        try:
            print(f"\n📍 Trying {model_alias} model...")
            agent = HackathonSQLAgent("healthcare_hackathon.db", llm_config=model_alias)
            print(f"✅ {model_alias} model working: {agent.get_config_info()}")
        except Exception as e:
            print(f"❌ {model_alias} model failed: {e}")

if __name__ == "__main__":
    # Run the main demo
    main()
    
    # Optional: Demo model switching
    # demo_model_switching()
    
    print("\n" + "="*60)
    print("🚀 QUICK START EXAMPLES")
    print("="*60)
    print("""
# Use default model (set in config.py)
agent = HackathonSQLAgent("healthcare_hackathon.db")

# Use specific model
agent = HackathonSQLAgent("healthcare_hackathon.db", llm_config="ollama")

# Use alias for easy switching
agent = HackathonSQLAgent("healthcare_hackathon.db", llm_config="free")

# Check what models are available
python config.py

# Query examples
agent.query("How many prescriptions were filled?")
agent.query("Which specialty has the most patients?")
agent.query("Show me diabetes-related prescriptions")
""")
    
    # Print schema information
    print("Database Schema:")
    print("=" * 50)
    print(agent.get_schema_info())
    print("\n")
    
    # Example queries
    example_queries = [
        "How many diagnosis claims are there by specialty?",
        "What are the most common diagnosis codes?",
        "Show me prescription patterns for diabetes medications",
        "Which providers have the highest claim volumes?",
        "What's the average copay by drug type?",
        "Find patients with both diabetes diagnosis and diabetes medications",
        "Show me the total claim amounts by specialty"
    ]
    
    print("Example Queries:")
    print("=" * 50)
    
    for i, query in enumerate(example_queries, 1):
        print(f"\n{i}. {query}")
        try:
            result = agent.query(query)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 30)

if __name__ == "__main__":
    # For hackathon participants to run
    main()

# Additional utility functions for hackathon participants

def quick_setup():
    """Quick setup function for hackathon participants"""
    print("🚀 Setting up your SQL Agent for the hackathon!")
    print("\n1. Install required packages:")
    print("pip install langchain openai sqlite3")
    print("\n2. Set your OpenAI API key in the main() function")
    print("\n3. Run the script to create sample database and test queries")
    print("\n4. Try your own queries using agent.query('Your question here')")
    
def show_example_queries():
    """Show example queries for participants"""
    examples = [
        "How many claims does each provider specialty have?",
        "What are the top 5 most prescribed medications?",
        "Show me patients who had claims in multiple specialties",
        "What's the average days supply for chronic condition medications?",
        "Find the most expensive claims by specialty",
        "Which diagnosis codes appear most frequently?",
        "Show prescription and diagnosis data for the same patients"
    ]
    
    print("Example Natural Language Queries for Healthcare Data:")
    print("=" * 50)
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example}")

# Healthcare Hackathon Challenge Ideas
"""
🏥 HEALTHCARE HACKATHON CHALLENGES:

1. BASIC: Find patterns in prescription vs diagnosis data
2. INTERMEDIATE: Build a provider performance dashboard
3. ADVANCED: Implement medication adherence analysis
4. EXPERT: Create a cost analysis tool with visualization

🛡️ SECURITY & PRIVACY FEATURES:
- SQL injection prevention
- Read-only operations enforced
- Patient data protection
- HIPAA-conscious design patterns

📊 HEALTHCARE SAMPLE DATA:
- dx_claims table (12 diagnosis records with ICD-10 codes)
- rx_prescriptions table (12 prescription records with NDC codes)
- providers table (12 healthcare providers across specialties)

🔧 HEALTHCARE CUSTOMIZATION IDEAS:
- Add patient demographics table (age, gender, zip)
- Implement drug interaction checking
- Create specialty-specific analysis tools
- Add insurance and formulary data
- Build medication adherence tracking

📋 COMMON HEALTHCARE QUERIES:
- Polypharmacy analysis (patients with multiple medications)
- Chronic disease management patterns
- Provider utilization patterns
- Cost per episode analysis
- Medication compliance tracking

💡 LEARNING OBJECTIVES:
- Understanding healthcare data structures
- ICD-10 and NDC code systems
- Provider specialty analysis
- Claims data relationships
- Healthcare analytics fundamentals
"""
