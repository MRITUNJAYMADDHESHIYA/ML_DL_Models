import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
from config import login_number, path


############################ mt5 initialize ##############################
result = mt5.initialize(path)
if login_number == mt5.account_info().login and result is True:
    print('Connection with MT5 established')
else:
    print('Connection failed')
    mt5.shutdown()


############################ Symbol #####################################
symbol1 = 'DBK.ETR'
timeframe = mt5.TIMEFRAME_H1
start_time = datetime.now() - timedelta(days=720)
end_time = datetime.now()

######################################### values #########################
def get_ohlc(symbol):
    rates = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)
    rates_df = pd.DataFrame(rates)
    rates_df['time'] = pd.to_datetime(rates_df['time'], unit='s')
    return rates_df

symbol1_df = get_ohlc(symbol1)
symbol1_df.to_csv(f'data/{symbol1.split(".")[0]}.csv', index=False)
symbol1_df

