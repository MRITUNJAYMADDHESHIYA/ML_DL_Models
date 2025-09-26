import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft

######################### Price Data() ############################
np.random.seed(42)
n = 200
t = np.arange(n)
# Price = trend + cycle + noise
prices = 50 + 0.1*t + 2*np.sin(2*np.pi*t/20) + np.random.normal(0, 0.5, n)

# Step 1: Apply Fast Fourier Transform
fft_values = fft(prices)

# Step 2: Filter - keep only first 'k' low frequencies
k = 10   # adjust this for smoothness
fft_filtered = np.copy(fft_values)
fft_filtered[k:-k] = 0

# Step 3: Inverse FFT to get smoothed signal
smoothed_prices = np.real(ifft(fft_filtered))

# Step 4: Plot
plt.figure(figsize=(12,6))
plt.plot(prices, label="Raw Prices", alpha=0.6)
plt.plot(smoothed_prices, label=f"Smoothed (k={k})", linewidth=2)
plt.legend()
plt.title("Fourier Filtering of Prices")
plt.xlabel("Time")
plt.ylabel("Price")
plt.show()




# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from numpy.fft import fft, fftfreq

# #############################Lets take some closing price
# prices = pd.Series([100, 101, 102, 101, 100, 99, 98, 99, 100, 101, 103, 102])

# #####################Convert prices to returns (stationary)
# returns = prices.pct_change().dropna()

# #####################Apply Fourier Transform
# N          = len(returns)
# freqs      = fftfreq(N)  # frequency bins
# fft_values = fft(returns)

# # Step 3: Magnitude spectrum
# magnitude = np.abs(fft_values)

# # Plot
# plt.figure(figsize=(12,5))
# plt.stem(freqs[:N//2], magnitude[:N//2], use_line_collection=True)
# plt.title("Frequency Spectrum of Returns")
# plt.xlabel("Frequency")
# plt.ylabel("Magnitude")
# plt.show()
