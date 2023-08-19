import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import warnings
import os
from matplotlib import pyplot as plt

# Number of years we are predicting for:
prediction_steps = 100

# Suppress ARIMA warnings
warnings.filterwarnings("ignore")

# Load your data into a DataFrame
inputData = pd.read_csv('RecyclePythonInput.csv')

if not os.path.exists('TotalPrediction'):
    os.makedirs('TotalPrediction')

arr = inputData.to_numpy()
data = arr[:,1:].sum(axis=-1)
# import pdb; pdb.set_trace()
# Generate some example data for demonstration
years = np.arange(2006, 2022)

# Create a DataFrame with the data
df = pd.DataFrame({'Year': years, 'Value': data})

# Loop through the regression models
regressors = [
    LinearRegression(),
    Ridge(),
    Lasso(),
    RandomForestRegressor(),
    SVR(),
    KNeighborsRegressor(),
    DecisionTreeRegressor(),
    GradientBoostingRegressor()
]

results = {}

for regressor in regressors:
    model_name = regressor.__class__.__name__
    model = regressor.fit(df[['Year']], df['Value'])
    
    # Predict for the next 100 years
    future_years = np.arange(2022, 2122)
    predictions = model.predict(future_years.reshape(-1, 1))
    results[model_name] = predictions
    
    # Save predictions to CSV
    predictions_df = pd.DataFrame({'Year': future_years, 'Prediction': predictions})
    predictions_df.to_csv(f'TotalPrediction/{model_name}_predictions.csv', index=False)
    
    # Calculate error (you might need to adjust this depending on the regression type)
    error = np.mean((df['Value'] - model.predict(df[['Year']]))**2)
    
    # Save error to CSV
    error_df = pd.DataFrame({'Model': [model_name], 'Error': [error]})
    if not os.path.exists('TotalPrediction/error.csv'):
        error_df.to_csv('TotalPrediction/error.csv', index=False)
    else:
        error_df.to_csv('TotalPrediction/error.csv', mode='a', header=False, index=False)
    
    # Plot and save the predictions
    plt.figure()
    plt.plot(df['Year'], df['Value'], label='Actual')
    plt.plot(future_years, predictions, label='Predicted')
    plt.title(model_name)
    plt.legend()
    plt.savefig(f'TotalPrediction/{model_name}_plot.png')
    plt.close()

# ARIMA and Seasonal ARIMA
order = (5, 1, 0)  # Example order (you need to tune this based on your data)
seasonal_order = (1, 1, 1, 12)  # Example seasonal order

arima_model = ARIMA(df['Value'], order=order)
sarima_model = SARIMAX(df['Value'], order=order, seasonal_order=seasonal_order)

arima_result = arima_model.fit()
sarima_result = sarima_model.fit()

arima_predictions = arima_result.predict(start=len(df), end=len(df) + 99)
sarima_predictions = sarima_result.predict(start=len(df), end=len(df) + 99)

results['ARIMA'] = arima_predictions
results['SARIMA'] = sarima_predictions

for model_name, predictions in results.items():
    predictions_df = pd.DataFrame({'Year': future_years, 'Prediction': predictions})
    predictions_df.to_csv(f'TotalPrediction/{model_name}_predictions.csv', index=False)
    
    plt.figure()
    plt.plot(df['Year'], df['Value'], label='Actual')
    plt.plot(future_years, predictions, label='Predicted')
    plt.title(model_name)
    plt.legend()
    plt.savefig(f'TotalPrediction/{model_name}_plot.png')
    plt.close()
