import pandas as pd
from time import sleep
from datetime import datetime, time
import MetaTrader5 as mt5
import matplotlib.pyplot as plt   

################################### Global ###########################
symbol1          = 'US30m'
symbol2          = 'USTECm'
timeframe        = mt5.TIMEFRAME_M1
period           = 500
TRADE_START      = time(19, 0)    # IST 19:00
TRADE_END        = time(23, 50)
ratio            = 0.94           # cointegration ratio
bollinger_period = 50
num_std          = 1.0           # threshold
max_trade        = 3
strategy_id      = 12345
volume           = 0.01
max_exposure     = 0.02

############################ get ohlc ################################
def get_ohlc(symbol):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, period)
    rates_df = pd.DataFrame(rates)
    rates_df['time'] = pd.to_datetime(rates_df['time'], unit='s')
    return rates_df

####################### Spread & Indicators ##########################
def get_spread_indicators():
    rates1 = get_ohlc(symbol1)
    rates2 = get_ohlc(symbol2)

    spread_df = rates1[['time', 'close']].merge(rates2[['time', 'close']], on='time', suffixes=['_1', '_2'])

    # Spread = close1 - ratio * close2
    spread_df['spread'] = spread_df['close_1'] - ratio * spread_df['close_2']

    sma = spread_df['spread'].rolling(bollinger_period).mean().iloc[-1]
    std = spread_df['spread'].rolling(bollinger_period).std().iloc[-1]

    lower_band = sma - num_std * std
    upper_band = sma + num_std * std

    return round(sma, 6), round(lower_band, 6), round(upper_band, 6), spread_df

######################## Live Spread ##################################
def get_live_spread():
    tick1 = mt5.symbol_info_tick(symbol1)
    tick2 = mt5.symbol_info_tick(symbol2)

    mid1 = (tick1.bid + tick1.ask) / 2
    mid2 = (tick2.bid + tick2.ask) / 2

    spread = mid1 - ratio * mid2
    return round(spread, 6)

######################## Exposures ####################################
def get_exposure_by_symbol(symbol):
    exposure = 0.0
    positions = mt5.positions_get(symbol=symbol)
    if positions:
        for pos in positions:
            if pos.type == 0:  # Buy
                exposure += pos.volume
            elif pos.type == 1:  # Sell
                exposure -= pos.volume
    return exposure

def get_exposure():
    exposure1 = get_exposure_by_symbol(symbol1)
    exposure2 = get_exposure_by_symbol(symbol2)
    return exposure1, exposure2

def check_exposure(exposure1, exposure2):
    # Check hedge neutrality
    if exposure1 == 0 and exposure2 == 0:
        return True
    elif abs(exposure1 + exposure2 * ratio) < 1e-6:
        return True
    else:
        print(f'Exposure mismatch: {exposure1}, {exposure2}')
        return False

######################## Market Orders ################################
def round_volume(symbol, vol):
    info = mt5.symbol_info(symbol)
    step = info.volume_step
    return round(vol / step) * step

def send_market_order(symbol, order_type, volume, sl=0.0, tp=0.0,
                      deviation=100, comment='', magic=int(strategy_id),
                      type_filling=mt5.ORDER_FILLING_IOC):
    tick = mt5.symbol_info_tick(symbol)
    order_dict = {'buy': mt5.ORDER_TYPE_BUY, 'sell': mt5.ORDER_TYPE_SELL}
    price_dict = {'buy': tick.ask, 'sell': tick.bid}

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": round_volume(symbol, volume),
        "type": order_dict[order_type],
        "price": price_dict[order_type],
        "sl": sl,
        "tp": tp,
        "deviation": deviation,
        "magic": magic,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": type_filling,
    }

    order_result = mt5.order_send(request)
    return order_result

############################### Close positions ######################
def close_all_positions(symbol):
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return None

    results = []
    for pos in positions:
        order_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(pos.symbol).bid if pos.type == 0 else mt5.symbol_info_tick(pos.symbol).ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": pos.ticket,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": order_type,
            "price": price,
            "deviation": 100,
            "magic": strategy_id,
            "comment": "Exit Hedge",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        results.append(result)
    return results

######################## Plot Spread + SMA + Bands ####################
import matplotlib.pyplot as plt
plt.ion()  # turn on interactive mode
fig, ax = plt.subplots(figsize=(12,6))

def plot_spread_realtime(spread_df):
    ax.clear()
    spread_df['sma']   = spread_df['spread'].rolling(bollinger_period).mean()
    spread_df['std']   = spread_df['spread'].rolling(bollinger_period).std()
    spread_df['upper'] = spread_df['sma'] + num_std * spread_df['std']
    spread_df['lower'] = spread_df['sma'] - num_std * spread_df['std']

    ax.plot(spread_df['time'], spread_df['spread'], label='Spread', color='blue')
    ax.plot(spread_df['time'], spread_df['sma'], label='SMA', color='orange')
    ax.plot(spread_df['time'], spread_df['upper'], label='Upper Band', color='green', linestyle='--')
    ax.plot(spread_df['time'], spread_df['lower'], label='Lower Band', color='red', linestyle='--')

    ax.legend()
    ax.set_title("Spread with SMA & Bollinger Bands")

    plt.draw()
    plt.pause(0.01)  # brief pause to refresh the plot



######################## Main function ###############################
if __name__ == '__main__':
    ################### Initialize ############################
    login_number = 274242894
    password     = "Mritunjay@76519"
    server       = "Exness-MT5Trial6"

    result = mt5.initialize(login=login_number, password=password, server=server)
    if result and mt5.account_info() and login_number == mt5.account_info().login:
        print('Connection with MT5 established')
    else:
        print('Connection failed:', mt5.last_error())
        mt5.shutdown()
        exit()

    ################### safety variables ######################
    entries = 0
    exits   = 0

    while True:
        sleep(2)

        if datetime.now().time() < TRADE_START:
            print(f'Waiting for trading start {TRADE_START}')
            sleep(5)
            continue
        elif datetime.now().time() > TRADE_END:
            print("Trading time over. Closing session.")
            mt5.shutdown()
            exit()

        sma, lower_band, upper_band, spread_df = get_spread_indicators()
        spread                                 = get_live_spread()
        exposure1, exposure2                   = get_exposure()

        if not check_exposure(exposure1, exposure2):
            sleep(5)
            continue

        ################### Trade Signal ########################
        entry_signal = None
        exit_signal  = None

        if spread < lower_band:
            entry_signal = 1   # Long spread (Buy symbol1, Sell symbol2)
        elif spread > upper_band:
            entry_signal = -1  # Short spread (Sell symbol1, Buy symbol2)

        ########## Exit signal
        if spread > sma:
            exit_signal = -1
        elif spread < sma:
            exit_signal = 1

        ################ Trade logic #########################
        res1 = res2 = None

        # ENTRY
        if entry_signal == 1 and abs(exposure1) < max_exposure and entries < max_trade:
            res1 = send_market_order(symbol1, 'buy', volume)
            res2 = send_market_order(symbol2, 'sell', volume * ratio)
            entries += 1

        elif entry_signal == -1 and abs(exposure1) < max_exposure and entries < max_trade:
            res1 = send_market_order(symbol1, 'sell', volume)
            res2 = send_market_order(symbol2, 'buy', volume * ratio)
            entries += 1

        # EXIT
        if exit_signal == 1 and exposure1 < 0:
            res1 = close_all_positions(symbol1)
            res2 = close_all_positions(symbol2)
            exits += 1

        elif exit_signal == -1 and exposure1 > 0:
            res1 = close_all_positions(symbol1)
            res2 = close_all_positions(symbol2)
            exits += 1

        if res1 and res2:
            print("Orders executed:", res1, res2)

        ################ Log ################
        print('Account:', mt5.account_info().login, mt5.account_info().equity)
        print('Exposures:', exposure1, exposure2)
        print('SMA / Bands:', sma, lower_band, upper_band)
        print('Spread:', spread)
        print('Entry Signal:', entry_signal, '| Exit Signal:', exit_signal)
        print('Entries:', entries, 'Exits:', exits)
        print('---\n')

        if datetime.now().second % 10 == 0:  # update every 10 sec
            plot_spread_realtime(spread_df)
