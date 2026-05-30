"""
Synthetic Data Generator for Customer Churn Prediction
Generates realistic customer data when real dataset is not available
"""

import pandas as pd
import numpy as np
from faker import Faker
import os
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import DATA_DIR, RANDOM_SEED

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
fake = Faker()
Faker.seed(RANDOM_SEED)


def generate_churn_dataset(n_samples=10000, churn_rate=0.27):
    """
    Generate synthetic customer churn dataset similar to Telco dataset
    
    Args:
        n_samples (int): Number of customer records to generate
        churn_rate (float): Proportion of customers who churn
    
    Returns:
        pd.DataFrame: Synthetic customer dataset
    """
    
    print(f"Generating {n_samples} synthetic customer records...")
    
    # Initialize lists to store data
    data = {
        'customerID': [],
        'gender': [],
        'SeniorCitizen': [],
        'Partner': [],
        'Dependents': [],
        'tenure': [],
        'PhoneService': [],
        'MultipleLines': [],
        'InternetService': [],
        'OnlineSecurity': [],
        'OnlineBackup': [],
        'DeviceProtection': [],
        'TechSupport': [],
        'StreamingTV': [],
        'StreamingMovies': [],
        'Contract': [],
        'PaperlessBilling': [],
        'PaymentMethod': [],
        'MonthlyCharges': [],
        'TotalCharges': [],
        'Churn': []
    }
    
    for i in range(n_samples):
        # Basic demographics
        customer_id = fake.uuid4()[:8]
        gender = np.random.choice(['Male', 'Female'])
        senior_citizen = np.random.choice([0, 1], p=[0.84, 0.16])
        partner = np.random.choice(['Yes', 'No'], p=[0.48, 0.52])
        dependents = np.random.choice(['Yes', 'No'], p=[0.30, 0.70])
        
        # Tenure (months with company)
        # Churners tend to have shorter tenure
        if np.random.random() < churn_rate:
            tenure = int(np.random.exponential(scale=10))
            will_churn = 'Yes'
        else:
            tenure = int(np.random.exponential(scale=35))
            will_churn = 'No'
        tenure = max(0, min(tenure, 72))  # Cap at 72 months
        
        # Services
        phone_service = np.random.choice(['Yes', 'No'], p=[0.90, 0.10])
        
        if phone_service == 'Yes':
            multiple_lines = np.random.choice(['Yes', 'No', 'No phone service'], p=[0.42, 0.48, 0.10])
        else:
            multiple_lines = 'No phone service'
        
        internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], p=[0.34, 0.44, 0.22])
        
        # Internet-dependent services
        if internet_service == 'No':
            online_security = 'No internet service'
            online_backup = 'No internet service'
            device_protection = 'No internet service'
            tech_support = 'No internet service'
            streaming_tv = 'No internet service'
            streaming_movies = 'No internet service'
        else:
            online_security = np.random.choice(['Yes', 'No'], p=[0.28, 0.72])
            online_backup = np.random.choice(['Yes', 'No'], p=[0.34, 0.66])
            device_protection = np.random.choice(['Yes', 'No'], p=[0.34, 0.66])
            tech_support = np.random.choice(['Yes', 'No'], p=[0.29, 0.71])
            streaming_tv = np.random.choice(['Yes', 'No'], p=[0.38, 0.62])
            streaming_movies = np.random.choice(['Yes', 'No'], p=[0.39, 0.61])
        
        # Contract type (churners more likely to have month-to-month)
        if will_churn == 'Yes':
            contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], p=[0.75, 0.15, 0.10])
        else:
            contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], p=[0.40, 0.30, 0.30])
        
        paperless_billing = np.random.choice(['Yes', 'No'], p=[0.59, 0.41])
        
        # Payment method (electronic check users churn more)
        if will_churn == 'Yes':
            payment_method = np.random.choice([
                'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
            ], p=[0.45, 0.20, 0.20, 0.15])
        else:
            payment_method = np.random.choice([
                'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
            ], p=[0.25, 0.25, 0.25, 0.25])
        
        # Monthly charges based on services
        base_charge = 20
        if phone_service == 'Yes':
            base_charge += 10
        if multiple_lines == 'Yes':
            base_charge += 10
        if internet_service == 'DSL':
            base_charge += 20
        elif internet_service == 'Fiber optic':
            base_charge += 40  # Fiber optic is more expensive
        if online_security == 'Yes':
            base_charge += 5
        if online_backup == 'Yes':
            base_charge += 5
        if device_protection == 'Yes':
            base_charge += 5
        if tech_support == 'Yes':
            base_charge += 5
        if streaming_tv == 'Yes':
            base_charge += 8
        if streaming_movies == 'Yes':
            base_charge += 8
        
        # Add some random variation
        monthly_charges = base_charge + np.random.uniform(-5, 15)
        monthly_charges = round(max(18.25, min(monthly_charges, 118.75)), 2)
        
        # Total charges = monthly charges * tenure (with some variation)
        if tenure == 0:
            total_charges = monthly_charges
        else:
            total_charges = round(monthly_charges * tenure * np.random.uniform(0.95, 1.05), 2)
        
        # Append to data dictionary
        data['customerID'].append(customer_id)
        data['gender'].append(gender)
        data['SeniorCitizen'].append(senior_citizen)
        data['Partner'].append(partner)
        data['Dependents'].append(dependents)
        data['tenure'].append(tenure)
        data['PhoneService'].append(phone_service)
        data['MultipleLines'].append(multiple_lines)
        data['InternetService'].append(internet_service)
        data['OnlineSecurity'].append(online_security)
        data['OnlineBackup'].append(online_backup)
        data['DeviceProtection'].append(device_protection)
        data['TechSupport'].append(tech_support)
        data['StreamingTV'].append(streaming_tv)
        data['StreamingMovies'].append(streaming_movies)
        data['Contract'].append(contract)
        data['PaperlessBilling'].append(paperless_billing)
        data['PaymentMethod'].append(payment_method)
        data['MonthlyCharges'].append(monthly_charges)
        data['TotalCharges'].append(total_charges)
        data['Churn'].append(will_churn)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    print(f"\nDataset generated successfully!")
    print(f"Total records: {len(df)}")
    print(f"Churn rate: {(df['Churn'] == 'Yes').sum() / len(df) * 100:.2f}%")
    
    return df


def save_dataset(df, filename='telco_churn.csv'):
    """
    Save dataset to CSV file
    
    Args:
        df (pd.DataFrame): Dataset to save
        filename (str): Name of the output file
    """
    output_path = os.path.join(DATA_DIR, filename)
    df.to_csv(output_path, index=False)
    print(f"\nDataset saved to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.2f} KB")
    
    return output_path


def display_dataset_info(df):
    """Display basic information about the dataset"""
    print("\n" + "=" * 80)
    print("DATASET INFORMATION")
    print("=" * 80)
    
    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nData types:")
    print(df.dtypes)
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
    print("\nChurn distribution:")
    print(df['Churn'].value_counts())
    print(f"Churn rate: {(df['Churn'] == 'Yes').sum() / len(df) * 100:.2f}%")
    
    print("\nNumerical statistics:")
    print(df[['tenure', 'MonthlyCharges', 'TotalCharges']].describe())
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    # Generate dataset
    print("=" * 80)
    print("SYNTHETIC CUSTOMER CHURN DATASET GENERATOR")
    print("=" * 80)
    
    # Generate 10,000 customer records
    df = generate_churn_dataset(n_samples=10000, churn_rate=0.27)
    
    # Display information
    display_dataset_info(df)
    
    # Save to CSV
    output_path = save_dataset(df)
    
    print("\n✓ Dataset generation complete!")
    print(f"✓ You can now run the main pipeline with this dataset")
    print(f"✓ File location: {output_path}")
