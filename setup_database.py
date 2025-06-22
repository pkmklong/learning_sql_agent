"""
Healthcare Database Setup Script
Creates sample healthcare database with realistic medical data
"""

import sqlite3
import os

def create_healthcare_database(db_path: str = "healthcare_hackathon.db"):
    """Create a sample healthcare database for the hackathon"""
    
    # Remove existing database if it exists
    if os.path.exists(db_path):
        print(f"🗑️  Removing existing database: {db_path}")
        os.remove(db_path)
    
    print(f"📊 Creating healthcare database: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create diagnosis claims table (DX)
    print("   Creating dx_claims table...")
    cursor.execute('''
        CREATE TABLE dx_claims (
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
    print("   Creating rx_prescriptions table...")
    cursor.execute('''
        CREATE TABLE rx_prescriptions (
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
    print("   Creating providers table...")
    cursor.execute('''
        CREATE TABLE providers (
            provider_id INTEGER PRIMARY KEY,
            provider_name TEXT NOT NULL,
            specialty TEXT NOT NULL,
            practice_location TEXT
        )
    ''')
    
    # Insert sample diagnosis claims data
    print("   Inserting diagnosis claims data...")
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
    print("   Inserting prescription data...")
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
    print("   Inserting provider data...")
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
    
    # Execute all inserts
    cursor.executemany('INSERT INTO dx_claims VALUES (?, ?, ?, ?, ?, ?, ?, ?)', dx_claims_data)
    cursor.executemany('INSERT INTO rx_prescriptions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', rx_prescriptions_data)
    cursor.executemany('INSERT INTO providers VALUES (?, ?, ?, ?)', providers_data)
    
    # Commit and close
    conn.commit()
    conn.close()
    
    print(f"✅ Database created successfully!")
    print(f"   📍 Location: {os.path.abspath(db_path)}")
    print(f"   📊 Tables: dx_claims (12 records), rx_prescriptions (12 records), providers (12 records)")
    
    return db_path

def show_database_info(db_path: str = "healthcare_hackathon.db"):
    """Show information about the database"""
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print(f"\n📊 Database Information: {db_path}")
    print("=" * 50)
    
    # Get table info
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    for table in tables:
        table_name = table[0]
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"📋 {table_name}: {count} records")
    
    # Show sample data
    print("\n🔍 Sample Data:")
    print("-" * 30)
    
    cursor.execute("SELECT COUNT(DISTINCT patient_id) FROM dx_claims")
    unique_patients = cursor.fetchone()[0]
    print(f"👥 Unique patients: {unique_patients}")
    
    cursor.execute("SELECT COUNT(DISTINCT provider_specialty) FROM dx_claims")
    specialties = cursor.fetchone()[0]
    print(f"🏥 Medical specialties: {specialties}")
    
    cursor.execute("SELECT COUNT(DISTINCT generic_name) FROM rx_prescriptions")
    medications = cursor.fetchone()[0]
    print(f"💊 Unique medications: {medications}")
    
    conn.close()

if __name__ == "__main__":
    print("🏥 Healthcare Database Setup")
    print("=" * 30)
    
    # Create the database
    db_path = create_healthcare_database()
    
    # Show info about what was created
    show_database_info(db_path)
    
    print("\n🚀 Ready for hackathon!")
    print("   Run: python healthcare_agent.py")
