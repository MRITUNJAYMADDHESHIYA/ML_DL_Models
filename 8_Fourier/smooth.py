import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft

# Step 1: Load CSV
df = pd.read_csv("C:/Users/Mritunjay Maddhesiya/OneDrive/Desktop/MT5/8_Fourier/SOL_1H.csv")   # replace with your file
df['Date'] = pd.to_datetime(df['open_time'])
df.set_index('Date', inplace=True)

# Use closing prices
prices = df['close'].values
n = len(prices)

# Step 2: Apply Fourier Transform
fft_values = fft(prices)

# Step 3: Filter (low-pass) - keep only first k frequencies
k = 20  # adjust for smoothness
fft_filtered = np.copy(fft_values)
fft_filtered[k:-k] = 0

# Step 4: Inverse FFT to get smoothed prices
smoothed_prices = np.real(ifft(fft_filtered))

# Step 5: Add to DataFrame
df['Smoothed'] = smoothed_prices

# Step 6: Plot
plt.figure(figsize=(14,6))
plt.plot(df.index, df['close'], label="Raw Prices", alpha=0.6)
plt.plot(df.index, df['Smoothed'], label=f"Fourier Smoothed (k={k})", linewidth=2)
plt.legend()
plt.title("Fourier Filtering of Market Data")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()



# Simple trading rule:
# Buy when price crosses above smoothed
# Sell when price crosses below smoothed

df['Signal'] = 0
df['Signal'][df['close'] > df['Smoothed']] = 1   # Buy
df['Signal'][df['close'] < df['Smoothed']] = -1  # Sell

plt.figure(figsize=(14,6))
plt.plot(df.index, df['close'], label="Raw Prices", alpha=0.6)
plt.plot(df.index, df['Smoothed'], label=f"Fourier Smoothed (k={k})", linewidth=2)

# Mark signals
buy_signals = df[df['Signal'] == 1]
sell_signals = df[df['Signal'] == -1]
plt.scatter(buy_signals.index, buy_signals['close'], marker='^', color='g', label='Buy', alpha=1)
plt.scatter(sell_signals.index, sell_signals['close'], marker='v', color='r', label='Sell', alpha=1)

plt.legend()
plt.title("Fourier Smoothed Trading Strategy")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()
