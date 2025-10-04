import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta

# Replace with your account details
login_number = 274242894
password = "Mritunjay@76519"
server = "Exness-MT5Trial6"

############################ MT5 initialize ##############################
if not mt5.initialize(login=login_number, password=password, server=server):
    print("initialize() failed, error code =", mt5.last_error())
    mt5.shutdown()
    quit()

account_info = mt5.account_info()
if account_info is None:
    print("Login failed:", mt5.last_error())
    mt5.shutdown()
else:
    print(f"Connected to account {account_info.login} on {account_info.server}")

############################ Symbol #####################################
symbol1 = "XAUUSDm"
timeframe = mt5.TIMEFRAME_D1
start_time = datetime.now() - timedelta(days=1440)
end_time = datetime.now()

######################################### values #########################
def get_ohlc(symbol):
    rates = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)
    rates_df = pd.DataFrame(rates)
    rates_df['time'] = pd.to_datetime(rates_df['time'], unit='s')
    return rates_df

symbol1_df = get_ohlc(symbol1)
symbol1_df.to_csv(f'{symbol1.split(".")[0]}_1D.csv', index=False)
print(symbol1_df.head())
