import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
from matplotlib import pyplot
from pandas import read_csv
import pmdarima as pm

# Suppress ARIMA warnings
warnings.filterwarnings("ignore")

# Load your data into a DataFrame
# Replace 'RecyclePythonInput.csv' with your actual file path
inputData = pd.read_csv('RecyclePythonInput.csv')

# Plotting the series
series = read_csv('wasteprojectionmodelPython.csv', header=0, index_col=0)
print(series.head())
series.plot()
pyplot.show()

# Since there is no clear trend for each of the individual materials 
# So a regression was fit to the total and proportions separately:

# List of columns (excluding the first column, assuming it's 'years')
column_list = inputData.columns[1:].tolist()

total = inputData.sum(numeric_only=True)
propData = inputData.div(inputData.sum(axis=1), axis=0)
print(total)
print(propData)

# Specify the regression models you want to use
regression_models = [
    LinearRegression(),
    Ridge(),
    Lasso(),
    RandomForestRegressor()
]
for model in regression_models:
    errorList = 0
    for column_name in column_list:
        material_data = propData[[column_name]]  # Remove 'years' from the selection

        # Splitting into train and test sets (80-20 split)
        X = material_data.index  # Use the index as the feature
        y = material_data[column_name]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model_name = model.__class__.__name__  # Get the name of the current model

        # Model training and evaluation
        model.fit(X_train.to_numpy().reshape(-1, 1), y_train)  # Reshape X_train for compatibility

        # Prediction and evaluation
        y_pred = model.predict(X_test.to_numpy().reshape(-1, 1))  # Reshape X_test for compatibility
        mse = mean_squared_error(y_test, y_pred)
        errorList += mse
        # print(f'{model_name} - Mean Squared Error for {column_name}: {mse}')
    print(f'{model_name} - Mean Squared Error: {errorList}')

# ARIMA model
errorARIMA = 0 
for column_name in column_list:
    material_data = propData[[column_name]]  # Remove 'years' from the selection

    # Splitting into train and test sets (80-20 split)
    X = material_data.index  # Use the index as the feature
    y = material_data[column_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Convert index to datetime for ARIMA
    X_train.index = pd.to_datetime(X_train)
    X_test.index = pd.to_datetime(X_test)

    # Fit ARIMA model
    model_arima = ARIMA(y_train, order=(5, 1, 0))  # Example order, tune as needed
    model_arima_fit = model_arima.fit()

    # Forecast using ARIMA model
    y_pred_arima = model_arima_fit.forecast(steps=len(y_test))

    # Print ARIMA results
    mse_arima = mean_squared_error(y_test, y_pred_arima)
    errorARIMA += mse_arima

print(f'ARIMA - Mean Squared Error: {errorARIMA}')


# Loop through each material
sArimaError = 0
for column_name in column_list:
    material_data = propData[[column_name, 'years']]  # Swap column order here

    # Splitting into train and test sets (80-20 split)
    X = material_data['years']
    y = material_data[column_name]

    # Convert index to datetime for SARIMA
    X.index = pd.to_datetime(X)

    # Fit SARIMA model
    model_sarima = pm.auto_arima (y, test='adf', seasonal=True, m=5, 
        trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
    model_sarima_fit = model_sarima.fit(disp=0)

    # Forecast using SARIMA model
    y_pred_sarima = model_sarima_fit.predict(start=len(y), end=len(y) + len(y_test) - 1)

    # Print SARIMA results
    mse_sarima = mean_squared_error(y_test, y_pred_sarima)
    sArimaError += mse_sarima
print(f'SARIMA - Mean Squared Error: {sArimaError}')
