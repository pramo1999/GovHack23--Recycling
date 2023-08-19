import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
from matplotlib import pyplot as plt
from pandas import read_csv
import pmdarima as pm
import numpy as np
from statsmodels.tsa.stattools import adfuller
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
import os
# Number of years we are predicting for: 
prediction_steps = 100

# Suppress ARIMA warnings
warnings.filterwarnings("ignore")

# Load your data into a DataFrame
# Replace 'RecyclePythonInput.csv'
inputData = pd.read_csv('RecyclePythonInput.csv')

# Plotting the series
series = read_csv('wasteprojectionmodelPython.csv', header=0, index_col=0)
print(series.head())
series.plot()
# pyplot.show()
plt.savefig(f'Recycling_plot.png')
plt.close()
# import pdb; pdb.set_trace()
# Since there is no clear trend for each of the individual materials 
# So a regression was fit to the total and proportions separately:

# List of columns (excluding the first column, assuming it's 'years')
column_list = inputData.columns[1:].tolist()
# import pdb; pdb.set_trace()
arr = inputData.to_numpy()

total = arr[:,1:].sum(axis=-1)
propData = inputData
propData[column_list] = inputData[column_list].div(inputData[column_list].sum(axis=1), axis=0)

print(total)
print(propData)

#combining the total column into the prop data column 


#outputdata
output_data = {}#propData.copy()

# clears the output file for the errors 
with open('MSE_models','w') as outfile:
    outfile.write('')

best_models = {} 
errorList = []
for column_name in column_list:
    # import pdb;pdb.set_trace()
    min_mse = float('inf')  # Initialize with a large value
    best_model = None
    
    # Specify the regression models you want to use
    regression_models = [
        LinearRegression(),
        Lasso(),
        RandomForestRegressor(),
        SVR(),
        KNeighborsRegressor(),
        DecisionTreeRegressor(),
        GradientBoostingRegressor()
    ]

    for model in regression_models: 
            material_data = propData[[column_name]]  # Remove 'years' from the selection

            # Splitting into train and test sets (80-20 split)
            X = material_data.index  # Use the index as the feature
            y = material_data[column_name]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=False)

            model_name = model.__class__.__name__  # Get the name of the current model

            # Model training and evaluation
            model.fit(X_train.to_numpy().reshape(-1, 1), y_train)  # Reshape X_train for compatibility

            # Prediction and evaluation
            y_pred = model.predict(X_test.to_numpy().reshape(-1, 1))  # Reshape X_test for compatibility
            y_pred = np.clip(y_pred, 0, 1.5)
            mse = mean_squared_error(y_test, y_pred)
            errorSum = mse
            print(f'{model_name} - Mean Squared Error for {column_name}: {mse}')
            if errorSum < min_mse:
                print(f'errorSum {errorSum}, min_mse {min_mse}')
                min_mse = errorSum
                # Forecast using RandomForestRegressor model
                X = inputData['years']                
                future_years = np.arange(max(X) + 1, max(X) + prediction_steps + 1)
                future_data = pd.DataFrame({'years': future_years})
                future_predictions = model.predict(future_data)
                future_predictions = np.clip(future_predictions, 0, 1.5)
                # import pdb; pdb.set_trace()
                best_model = model_name
                print(column_name, best_model, future_predictions)
            errorList.append(errorSum)    
            with open('MSE_models','a') as outfile:
                outfile.write(f'{model_name} - Mean Squared Error: {errorSum}\n')

    # ================================================================================
    # ARIMA model
    errorARIMA = 0 
    material_data = propData[[column_name]]  # Remove 'years' from the selection

    # Splitting into train and test sets (80-20 split)
    X = material_data.index  # Use the index as the feature
    y = material_data[column_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=True)

    # Convert index to datetime for ARIMA
    X_train.index = pd.to_datetime(X_train)
    X_test.index = pd.to_datetime(X_test)

    # Fit ARIMA model
    model_arima = ARIMA(y_train, order=(5, 1, 0))  # Example order, tune as needed
    model_arima_fit = model_arima.fit()

    # Forecast using ARIMA model
    y_pred_arima = model_arima_fit.forecast(steps=len(y_test))
    y_pred_arima = np.clip(y_pred_arima, 0, 1.5) 

    # Print ARIMA results
    mse_arima = mean_squared_error(y_test, y_pred_arima)
    errorARIMA += mse_arima
    errorList.append(errorARIMA)
    with open('MSE_models','a') as outfile:
        outfile.write(f'ARIMA - Mean Squared Error: {errorARIMA}\n')


    if errorARIMA < min_mse:
        min_mse = errorARIMA
        best_model = 'ARIMA'
        # Forecast using RandomForestRegressor model
        X = inputData['years']             
        future_years = np.arange(max(X) + 1, max(X) + prediction_steps + 1)
        future_data = pd.DataFrame({'years': future_years})
        future_predictions = model.predict(future_data)
        future_predictions = np.clip(future_predictions, 0, 1.5)
        print(column_name, best_model, future_predictions)
    output_data[column_name] = future_predictions
    best_models[column_name] = best_model
    if column_name == 'Concrete':
        import pdb; pdb.set_trace()


#Exporting the list of errors: 


# import pdb; pdb.set_trace()
#Exporting the list of errors:
output_data_df = pd.DataFrame(output_data)
for i in range(len(output_data_df)):
    output_data_df.iloc[i] = output_data_df.iloc[i] / output_data_df.iloc[i].sum()
# Save the output_data DataFrame to a CSV file
output_data_df.to_csv('output_predictions.csv', index=False) 


# Create a DataFrame from the best_models dictionary
best_model_df = pd.DataFrame(best_models.items(), columns=['Column', 'Best_Model'])

# Save the best_model_df DataFrame to a CSV file
best_model_df.to_csv('best_models.csv', index=False)

# # Save the output_data DataFrame to a new CSV file
# output_data.to_csv('output_predictions.csv', index=False)


# # =============================================================================
# # Seasonal Arima
# # Loop through each material
#     sArimaError = 0  # Initialize sArimaError inside the loop
#     material_data = propData[[column_name, 'years']]  # Swap column order here
#     # Splitting into train and test sets (80-20 split)
#     x = material_data['years']
#     y = material_data[column_name]

#     # Convert index to datetime for SARIMA
#     x.index = pd.to_datetime(x)  # Corrected variable name
#     # =============================================================================
#     # checking for stationarity: Data needs to be stationary to apply the Seasonal 
#     # Arima model 

#     # Perform the Augmented Dickey-Fuller (ADF) test
#     result = adfuller(y)

#     # Extract the p-value from the test result
#     p_value = result[1]

#     # Set the significance level (e.g., 0.05)
#     alpha = 0.05

#     # Check if the p-value is less than the significance level
#     if p_value > alpha:
#         print("The time series is non-stationary (null hypothesis not rejected)")
#     else:
#         # # Fit SARIMA model
#         # model_sarima = pm.auto_arima(y, test='adf', seasonal=True, m=5, 
#         #                             trace=True, error_action='ignore',
#         #                              suppress_warnings=True, stepwise=True)
#         # model_sarima_fit = model_sarima.fit(y)

#         # # Forecast using SARIMA model
#         # forecast_horizon = len(y_test)
#         # y_pred_sarima = model_sarima_fit.predict(n_periods=forecast_horizon)

#         # # Print SARIMA results
#         # mse_sarima = mean_squared_error(y_test, y_pred_sarima)
#         # sArimaError += mse_sarima
#         # print(f'SARIMA - Mean Squared Error for {column_name}: {sArimaError}')  # Print error for each material
#         # Fit SARIMA 
#         import pdb; pdb.set_trace()
#         model_sarima = pm.auto_arima(y, test='ch', seasonal=True, m=5,
#                                     trace=True, error_action='ignore',
#                                     suppress_warnings=True, stepwise=True)
#         model_sarima_fit = model_sarima.fit(y_train)  # Use y_train instead of y

#         # Forecast using SARIMA model
#         forecast_horizon = len(y_test)
#         y_pred_sarima = model_sarima_fit.predict(n_periods=forecast_horizon)

#         # Print SARIMA results
#         mse_sarima = mean_squared_error(y_test, y_pred_sarima)
#         sArimaError += mse_sarima
#         print(f'SARIMA - Mean Squared Error for {column_name}: {mse_sarima}')  # Print error for each material
