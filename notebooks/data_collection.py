from src.data_loader import fetch_stock_data
import pandas as pd

# Fetch AAPL stock data
data = fetch_stock_data('AAPL', '2020-01-01', '2023-12-31')

# Save data
data.to_csv('data/AAPL_stock_data.csv')
print("Data saved successfully!")