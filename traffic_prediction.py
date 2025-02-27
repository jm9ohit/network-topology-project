"""
Traffic Matrix Estimation and Prediction Module.
This module implements time series forecasting for network traffic using ARIMA models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

def generate_sample_data():
    """
    Generate sample network traffic data for demonstration purposes.
    
    Returns:
        DataFrame: Sample traffic data with timestamps
    """
    data = {
        'Timestamp': pd.date_range(start='2024-08-01 00:00', periods=30, freq='10T'),
        'Path_A': [200, 195, 210, 220, 215, 225, 230, 240, 235, 245, 250, 255, 
                  260, 270, 275, 280, 290, 295, 300, 310, 315, 320, 330, 335, 
                  340, 345, 350, 355, 360, 365],
        'Path_B': [100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 
                  160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215, 
                  220, 225, 230, 235, 240, 245],
        'Path_C': [50, 55, 53, 60, 58, 62, 64, 66, 68, 72, 74, 76, 78, 80, 82, 
                  84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112]
    }

    df = pd.DataFrame(data)
    df.set_index('Timestamp', inplace=True)
    return df

def fit_arima_and_predict(data, p, d, q, steps):
    """
    Fit an ARIMA model to the data and make predictions.
    
    Args:
        data: Time series data to model
        p: AR order
        d: Differencing order
        q: MA order
        steps: Number of steps to forecast
        
    Returns:
        Series: Forecasted values
    """
    model = ARIMA(data, order=(p, d, q))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

def visualize_data(df, title="Network Traffic Over Time"):
    """
    Visualize the time series data.
    
    Args:
        df: DataFrame containing the data
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    df.plot()
    plt.title(title)
    plt.xlabel('Timestamp')
    plt.ylabel('Traffic (Mbps)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to run the traffic prediction pipeline.
    """
    # Load or generate sample data
    df = generate_sample_data()
    print("Sample data head:")
    print(df.head())
    
    # Visualize the data
    visualize_data(df)
    
    # ARIMA model parameters
    p, d, q = 2, 1, 2  # Example parameters, can be optimized
    forecast_steps = 10  # Number of steps to predict
    
    # Fit models and make predictions
    predictions = {}
    for path in df.columns:
        print(f"Fitting ARIMA for {path}...")
        predictions[path] = fit_arima_and_predict(df[path], p, d, q, forecast_steps)
    
    # Create DataFrame with predictions
    pred_df = pd.DataFrame(
        predictions, 
        index=pd.date_range(
            start=df.index[-1] + pd.Timedelta(minutes=10), 
            periods=forecast_steps, 
            freq='10T'
        )
    )
    
    print("\nPredictions:")
    print(pred_df)
    
    # Visualize the predictions with actual data
    combined_df = pd.concat([df, pred_df])
    visualize_data(combined_df, title="Network Traffic with Predictions")

if __name__ == "__main__":
    main()
