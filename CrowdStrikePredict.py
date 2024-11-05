import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def calculateVal(data):
    data['Date'] = pd.to_datetime(data['Date'])
    data['Time'] = (data['Date'] - data['Date'].min()).dt.days  # Convert dates to a numeric value

    X = sm.add_constant(data['Time'])  # Adds a constant term for intercept
    model = sm.OLS(data['Close'], X).fit()  # Fit the model
    
    # Generate future predictions
    print(model.summary())
    
    # Generate predictions for the next 7 days
    future_time = data['Time'].max() + np.arange(1, 8)  # Predict for next 7 days
    future_dates = data['Date'].max() + pd.to_timedelta(future_time - data['Time'].max(), unit='D')
    
    # Calculate future predictions using model parameters
    future_predictions = model.predict(sm.add_constant(future_time))
    
    # Add predicted future data to the DataFrame
    future_data = pd.DataFrame({'Date': future_dates, 'Close': future_predictions})
    data = pd.concat([data, future_data], ignore_index=True)
    
    return data, model

def main():
    # Read stock data
    cs = pd.read_csv('CrowdStrike Stock.csv')

    print(cs.head())

    # Calculate trend
    cs_new, model = calculateVal(cs)

    # downsampling (reducing data points)
    cs_downsampled = cs_new.iloc[::7] # take every 7th row

    # Plot actual vs predicted
    plt.figure(figsize=(14, 7))

    sns.scatterplot(data=cs_downsampled, x='Date', y='Close', label='Downsampled Data')
    sns.lineplot(data=cs_new, x='Date', y='Close', label='Actual Data', color='blue')

    # Plot future predictions
    sns.lineplot(data=cs_new.iloc[-7:], x='Date', y='Close', label='Predicted Data', color='red', linestyle='--')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()  # Displays plot
    

if __name__ == "__main__":
    main()
