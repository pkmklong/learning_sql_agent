import sqlite3
import os
import random
from datetime import datetime, timedelta

def create_database(db_path="healthcare.db", num_records=1000):
    if os.path.exists(db_path):
        os.remove(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE dx_claims (
            claim_id INTEGER PRIMARY KEY,
            patient_id INTEGER,
            diagnosis_code TEXT,
            service_date DATE,
            provider_id INTEGER,
            provider_specialty TEXT,
            claim_amount DECIMAL(10,2)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE rx_prescriptions (
            prescription_id INTEGER PRIMARY KEY,
            patient_id INTEGER,
            drug_name TEXT,
            generic_name TEXT,
            service_date DATE,
            provider_id INTEGER,
            quantity INTEGER,
            copay DECIMAL(8,2)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE providers (
            provider_id INTEGER PRIMARY KEY,
            provider_name TEXT,
            specialty TEXT
        )
    ''')
    
    # Generate sample data
    dx_data = generate_dx_claims(num_records)
    rx_data = generate_rx_prescriptions(num_records)
    provider_data = generate_providers()
    
    cursor.executemany('INSERT INTO dx_claims VALUES (?, ?, ?, ?, ?, ?, ?)', dx_data)
    cursor.executemany('INSERT INTO rx_prescriptions VALUES (?, ?, ?, ?, ?, ?, ?, ?)', rx_data)
    cursor.executemany('INSERT INTO providers VALUES (?, ?, ?)', provider_data)
    
    conn.commit()
    conn.close()

def generate_providers():
    """Generate provider data"""
    first_names = ['Sarah', 'Michael', 'Jennifer', 'David', 'Lisa', 'Robert', 'Maria', 'James', 
                   'Emily', 'Andrew', 'Rachel', 'Kevin', 'Amanda', 'Daniel', 'Jessica', 'Ryan']
    last_names = ['Chen', 'Rodriguez', 'Kim', 'Thompson', 'Wang', 'Johnson', 'Garcia', 'Wilson',
                  'Davis', 'Miller', 'Brown', 'Lee', 'Taylor', 'Anderson', 'Jackson', 'White']
    specialties = ['Endocrinology', 'Cardiology', 'Pulmonology', 'Oncology', 'Orthopedics', 
                   'Psychiatry', 'Nephrology', 'Gastroenterology', 'Neurology', 'Dermatology']
    
    providers = []
    for i, specialty in enumerate(specialties, 201):
        name = f"Dr. {random.choice(first_names)} {random.choice(last_names)}"
        providers.append((i, name, specialty))
    
    return providers

def generate_dx_claims(num_records):
    """Generate diagnosis claims data"""
    diagnosis_codes = ['E1140', 'I2510', 'J449', 'Z7901', 'M545', 'F329', 'N183', 'K219', 'G309', 'L309']
    specialties = ['Endocrinology', 'Cardiology', 'Pulmonology', 'Oncology', 'Orthopedics', 
                   'Psychiatry', 'Nephrology', 'Gastroenterology', 'Neurology', 'Dermatology']
    
    dx_claims = []
    start_date = datetime(2024, 1, 1)
    
    for i in range(1, num_records + 1):
        patient_id = random.randint(1001, 1001 + num_records // 10)  # ~10 claims per patient
        diagnosis_code = random.choice(diagnosis_codes)
        service_date = start_date + timedelta(days=random.randint(0, 180))
        provider_id = random.randint(201, 210)
        specialty = specialties[provider_id - 201]
        claim_amount = round(random.uniform(50.0, 500.0), 2)
        
        dx_claims.append((i, patient_id, diagnosis_code, service_date.strftime('%Y-%m-%d'), 
                         provider_id, specialty, claim_amount))
    
    return dx_claims

def generate_rx_prescriptions(num_records):
    """Generate prescription data"""
    medications = [
        ('Metformin HCl 500mg', 'Metformin'), ('Lisinopril 10mg', 'Lisinopril'),
        ('Albuterol Sulfate 90mcg', 'Albuterol'), ('Ibuprofen 600mg', 'Ibuprofen'),
        ('Sertraline 50mg', 'Sertraline'), ('Furosemide 40mg', 'Furosemide'),
        ('Omeprazole 20mg', 'Omeprazole'), ('Gabapentin 300mg', 'Gabapentin'),
        ('Atorvastatin 20mg', 'Atorvastatin'), ('Amlodipine 5mg', 'Amlodipine'),
        ('Levothyroxine 50mcg', 'Levothyroxine'), ('Hydrochlorothiazide 25mg', 'HCTZ')
    ]
    
    rx_prescriptions = []
    start_date = datetime(2024, 1, 1)
    
    for i in range(1, num_records + 1):
        patient_id = random.randint(1001, 1001 + num_records // 10)
        drug_name, generic_name = random.choice(medications)
        service_date = start_date + timedelta(days=random.randint(0, 180))
        provider_id = random.randint(201, 210)
        quantity = random.choice([30, 60, 90, 1])  # 1 for inhalers/creams
        copay = round(random.uniform(5.0, 50.0), 2)
        
        rx_prescriptions.append((i, patient_id, drug_name, generic_name, 
                               service_date.strftime('%Y-%m-%d'), provider_id, quantity, copay))
    
    return rx_prescriptions

if __name__ == "__main__":
    create_database(num_records=1000)  # Generate 1000 records per table
