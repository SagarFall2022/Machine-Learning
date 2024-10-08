import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib
import os
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
import time

# Loading the environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Function for generating the synthetic data
def generate_synthetic_data(num_samples=5000):
    np.random.seed(42)
    data = {
        'Suspension_Travel': np.random.uniform(10, 30, num_samples),  # mm
        'Load_Weight': np.random.uniform(1000, 3000, num_samples),  # kg
        'Shock_Absorber_Pressure': np.random.uniform(100, 400, num_samples),  # psi
        'Vibration_Intensity': np.random.uniform(0.1, 2.0, num_samples),  # g
        'Temperature': np.random.uniform(50, 100, num_samples),  # °F
        'Mileage': np.random.uniform(10000, 200000, num_samples),  # km
    }
    df = pd.DataFrame(data)
    df['Performance_Metric'] = (0.5 * df['Suspension_Travel'] + 
                                 0.2 * df['Load_Weight'] - 
                                 0.3 * df['Shock_Absorber_Pressure'] + 
                                 np.random.normal(0, 10, num_samples))
    return df

# Loading the data
def load_data():
    data_dir = 'data'
    data_file_path = os.path.join(data_dir, 'synthetic_data.csv')

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    if not os.path.exists(data_file_path):
        df = generate_synthetic_data(num_samples=8000)  
        df.to_csv(data_file_path, index=False)
    else:
        df = pd.read_csv(data_file_path)

    features = ['Suspension_Travel', 'Load_Weight', 'Shock_Absorber_Pressure', 
                'Vibration_Intensity', 'Temperature', 'Mileage']
    target = 'Performance_Metric'

    X = df[features]
    y = df[target]
    return X, y

def train_model():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    #Predicting the test dataset :)
    y_pred = model.predict(X_test)

    
    r2 = r2_score(y_test, y_pred)

    print("Model Coefficients:")
    for feature, coef in zip(X.columns, model.coef_):
        print(f"{feature}: {coef:.4f}")
    
    print(f"\nR² Score: {r2:.4f}")

    #Saving the trained model
    joblib.dump(model, 'model.joblib')
    return model


def load_model():
    return joblib.load('model.joblib')

# Predicting the performance
def predict_performance(suspension_travel, load_weight, shock_pressure, vibration_intensity, temperature, mileage):
    model = load_model()

    input_data = pd.DataFrame({
        'Suspension_Travel': [suspension_travel],
        'Load_Weight': [load_weight],
        'Shock_Absorber_Pressure': [shock_pressure],
        'Vibration_Intensity': [vibration_intensity],
        'Temperature': [temperature],
        'Mileage': [mileage]
    })

    prediction = model.predict(input_data)
    return prediction[0]

def generate_maintenance_insights(performance_metric):
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    prompt = (f"The truck's suspension performance is rated at {performance_metric:.2f}. "
              f"Based on this performance metric, provide detailed maintenance recommendations, "
              f"including potential issues that might arise, suggested inspection intervals, "
              f"and any specific components that should be prioritized for maintenance. "
              f"Also, include any best practices for maintaining optimal suspension performance.")

    max_retries = 5
    base_delay = 1
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise e
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"Rate limit exceeded. Retrying in {delay:.2f} seconds...")
            time.sleep(delay)
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return "Unable to generate insights due to an error."

    return "Unable to generate insights due to persistent rate limit errors."
