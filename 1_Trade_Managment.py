import MetaTrader5 as mt5
import pandas as pd

####################### Initialization #########################
mt5.initialize()
login=106402323
password="@@@@@@@@@19"
server="E@@@@@@@@@@@@@@a@@@@@@6"

mt5.login(login, password=password, server=server)


########################## Global variables #########################
symbol = "XAUUSD"
timeframe = mt5.TIMEFRAME_M1  # 1-minute timeframe
lot_size  = 0.1
deviation = 20                # maximum allowed price slippage (in points)
magic = 1000
comment = "Python Order"


######################### Account Information #########################
def account_info():
    account_info = mt5.account_info()
    if account_info is not None:
        return account_info
    else:
        print("Failed to get account info")
        return None


######################### Terminal Information #########################
def terminal_info():
    terminal_info = mt5.terminal_info()
    if terminal_info is not None:
        return terminal_info
    else:
        print("Failed to get terminal info")
        return None

######################### Symbol Information #########################
def symbol_total():
    symbols = mt5.symbols_get()
    if symbols is None:
        print("symbols_get() failed:", mt5.last_error())
        return None
    if len(symbols) > 0:
        return pd.DataFrame(symbols)
    else:
        print("No symbols found")
        return []

def get_symbol_info(symbol="EURUSD"):
    info = mt5.symbol_info(symbol)
    if info is None:
        print(f"Failed to get symbol info for {symbol}, error: {mt5.last_error()}")
        return None
    
    print(f"Symbol Info for {symbol}:")
    print(f" Spread   = {info.spread}")
    print(f" Digits   = {info.digits}")
    print(f" Point    = {info.point}")
    print(f" Bid      = {info.bid}")
    print(f" Ask      = {info.ask}")
    print(f" Session High = {info.session_high}")
    print(f" Session Low  = {info.session_low}")
    
    print("\nAll properties:")
    info_dict = info._asdict()
    for prop, val in info_dict.items():
        print(f"{prop} = {val}")
    
    return info_dict




################################ Orders Functions #########################
def place_order(symbol, order_type, volum=lot_size, price=None, sl=None, tp=None, deviation=20, magic=magic, comment="Python Order"):
    # Ensure symbol is ready
    if not mt5.symbol_select(symbol, True):
        print(f"Failed to select {symbol}")
        return None

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Symbol not found: {symbol}")
        return None

    # Define order types
    order_dict = {
        "buy": mt5.ORDER_TYPE_BUY,
        "sell": mt5.ORDER_TYPE_SELL,
        "buy_limit": mt5.ORDER_TYPE_BUY_LIMIT,
        "sell_limit": mt5.ORDER_TYPE_SELL_LIMIT,
        "buy_stop": mt5.ORDER_TYPE_BUY_STOP,
        "sell_stop": mt5.ORDER_TYPE_SELL_STOP,
    }
    if order_type not in order_dict:
        print("Invalid order type")
        return None

    order_type_mt5 = order_dict[order_type]

    # Get market prices
    tick = mt5.symbol_info_tick(symbol)
    ask, bid = tick.ask, tick.bid

    # Market orders use bid/ask price, pending orders use provided price
    if order_type == "buy":
        order_price = ask
    elif order_type == "sell":
        order_price = bid
    else:
        if price is None:
            print("Price required for limit/stop orders")
            return None
        order_price = price

    # Prepare request
    request = {
        "action": mt5.TRADE_ACTION_DEAL if order_type in ["buy", "sell"] else mt5.TRADE_ACTION_PENDING,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type_mt5,
        "price": order_price,
        "sl": sl,
        "tp": tp,
        "deviation": deviation,
        "magic": magic,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,  # Good Till Cancelled
        "type_filling": mt5.ORDER_FILLING_IOC,  # Immediate or Cancel
    }

    # Send order
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order failed: {result.retcode}, {result.comment}")
    else:
        print(f"Order placed successfully: {result}")
    return result



