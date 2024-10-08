# Truck Suspension Performance Prediction

This project aims to develop a predictive model for truck suspension performance using linear regression. 
The model takes various input parameters related to truck specifications and predicts a performance metric. 
Additionally, the project includes a Streamlit web application that allows users to interactively input data and receive predictions. 
The application also leverages OpenAI's API to provide maintenance insights based on the predicted performance.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Model Performance](#model-performance)
- [Deployment](#deployment)

## Features

- Predicts truck suspension performance based on user inputs.
- Provides maintenance recommendations using OpenAI's API.
- User-friendly interface built with Streamlit.

## Technologies Used

- **Python**: Programming language used for development.
- **Streamlit**: Framework for building the web application.
- **Scikit-learn**: Library for machine learning algorithms.
- **OpenAI API**: For generating maintenance insights.
- **Pandas**: For data manipulation and analysis.
- **Joblib**: For saving and loading the trained model.

## Installation

1. **Clone this repository**:

   ```bash
   git clone https://github.com/SagarFall2022/Machine-Learning.git
   cd Truck Suspension Performance Prediction/
   ```

2. **Create a virtual environment and activate it**:

   ```bash
   python -m venv venv
   ```

   - On Windows use:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux use:
     ```bash
     source venv/bin/activate
     ```

3. **Install the required packages**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set your OpenAI API key in your environment variables**:

   - On macOS/Linux:
     ```bash
     export OPENAI_API_KEY='your_openai_api_key'
     ```
   - On Windows use:
     ```bash
     set OPENAI_API_KEY='your_openai_api_key'
     ```

## Usage

1. **Open your terminal or command prompt.**
2. **Navigate to the project directory**:

   ```bash
   cd path/to/my_truck_project/
   ```

3. **Run the Streamlit app**:

   ```bash
   streamlit run app.py
   ```

4. **Interact with the application**:
   - A new tab will open in your default web browser displaying the Streamlit application.
   - Input your truck parameters and click the "Predict Performance" button to receive predictions and maintenance insights.

## Model Training

The model is trained using synthetic data generated within the project. The training process includes:

- Data generation
- Model training using Linear Regression
- Saving the trained model for future predictions

## Model Performance

During training, the model's coefficients and R² score are printed to provide insights into its performance.

## Deployment

This application can be deployed using platforms like Streamlit Sharing, Heroku, or AWS. 
Ensure all environment variables are correctly set in your deployment environment.
