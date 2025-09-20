"""
EMA Crossover Bot for XAUUSD on 1-minute timeframe using MetaTrader5 Python API
- OOP-based implementation for cleaner structure and reusability
- Trades only between 19:00 and 21:00 IST (Asia/Kolkata)
- Uses EMA fast/slow crossover (default 9 / 21)
- Uses ATR(14) on 1-min to set SL distance (1 * ATR) and TP = 3 * SL (1:3 R)
- Risk = 2% of account balance per trade
- Only one position at a time for the symbol
- Places market orders with configurable deviation (slippage in points)
"""

import time
import math
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo
import MetaTrader5 as mt5
import pandas as pd


############################### Global Variable ###############################################
symbol        = "XAUUSDm"
timeframe     = mt5.TIMEFRAME_M1
risk          = 0.02             #2% per trade
deviation     = 20               #Max slippage
ema_fast      = 9
ema_slow      = 21
atr_period    = 14
check_interval = 10              #bot will check conditions every 10 seconds.

################################# class method ##########################################
class EMACrossoverBot:
    def __init__(self, symbol=symbol, timeframe=timeframe, fast=ema_fast, slow=ema_slow, atr_period=atr_period, risk=risk, deviation=deviation, check_interval=check_interval):
        self.symbol         = symbol
        self.timeframe      = timeframe
        self.fast           = fast
        self.slow           = slow
        self.atr_period     = atr_period
        self.risk           = risk
        self.deviation      = deviation
        self.check_interval = check_interval

        self.trade_start    = dt_time(19, 0)  #7:00 PM ist
        self.trade_end      = dt_time(23, 59)  #2:00 AM ist
        self.IST            = ZoneInfo("Asia/Kolkata")

    def initialize(self):
        if not mt5.initialize():
            raise RuntimeError(f"MT5 initialize() failed, code={mt5.last_error()}")
        print("MT5 initialized")

    def shutdown(self):
        mt5.shutdown()
        print("MT5 shutdown")

    def account_info():
        account_info = mt5.account_info()
        if account_info is not None:
            return account_info
        else:
            print("Failed to get account info")
            return None

##################################### latest candle data #####################################
    def get_rates(self, count=500):
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, count)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No rates for {self.symbol}")
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

###################################### Indicator Information ##################################
    @staticmethod
    def ema(series, span):
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def atr(df, period=14):
        df = df.copy()
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = (df['high'] - df['close'].shift()).abs()
        df['low_close'] = (df['low'] - df['close'].shift()).abs()
        df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        return df['tr'].rolling(period).mean()

###################################### Trading Time ######################################
    def in_trading_window(self):
        now_ist = datetime.now(self.IST)
        return self.trade_start <= now_ist.time() <= self.trade_end

    def has_open_position(self):
        positions = mt5.positions_get(symbol=self.symbol)
        return positions is not None and len(positions) > 0

    def calculate_lot_size(self, stop_price_diff, risk_amount):
        info = mt5.symbol_info(self.symbol)
        if info is None:
            raise RuntimeError(f"symbol_info for {self.symbol} not available")

        tick_value = None
        for attr in ("trade_tick_value", "trade_tick_value_d", "tick_value", "trade_tick_value1"):
            tick_value = getattr(info, attr, None)
            if tick_value:
                break

        contract_size = getattr(info, 'trade_contract_size', None) or getattr(info, 'contract_size', None)
        point = getattr(info, 'point', None)

        if tick_value and point:
            tick_size = getattr(info, 'trade_tick_size', None) or getattr(info, 'tick_size', None) or point
            money_per_price = tick_value / tick_size
            lots = risk_amount / (money_per_price * stop_price_diff)
        elif contract_size and point:
            money_per_price = contract_size
            lots = risk_amount / (money_per_price * stop_price_diff)
        else:
            print("Warning: lot size calculation fallback to minimal lot.")
            lots = info.volume_min if hasattr(info, 'volume_min') else 0.01

        vol_step = getattr(info, 'volume_step', 0.01)
        vol_min = getattr(info, 'volume_min', 0.01)
        vol_max = getattr(info, 'volume_max', 100.0)

        if math.isnan(lots) or lots <= 0:
            return vol_min

        lots = max(vol_min, min(lots, vol_max))
        rounded = math.floor(lots / vol_step) * vol_step
        return max(round(rounded, 8), vol_min)

######################################## Market Order ####################################################
    def send_market_order(self, order_type, volume, price, sl_price, tp_price, comment="ema_cross"):
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": self.deviation,
            "magic": 123456,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        return mt5.order_send(request)


################################################# ema strategy #######################################
    def check_signal_and_trade(self):
        df = self.get_rates(count=200)
        df['ema_fast'] = self.ema(df['close'], self.fast)
        df['ema_slow'] = self.ema(df['close'], self.slow)
        df['atr'] = self.atr(df, self.atr_period)

        latest = df.iloc[-2]
        print("\n===== Latest Candle Info =====")
        print(f"Time   : {latest['time']}")
        print(f"Open   : {latest['open']}")
        print(f"High   : {latest['high']}")
        print(f"Low    : {latest['low']}")
        print(f"Close  : {latest['close']}")
        print(f"EMA{self.fast}: {latest['ema_fast']:.2f}")
        print(f"EMA{self.slow}: {latest['ema_slow']:.2f}")
        print(f"ATR{self.atr_period}: {latest['atr']:.2f}")
        prev = df.iloc[-3]

        bullish = (prev['ema_fast'] <= prev['ema_slow']) and (latest['ema_fast'] > latest['ema_slow'])
        bearish = (prev['ema_fast'] >= prev['ema_slow']) and (latest['ema_fast'] < latest['ema_slow'])

        atr_val = df['atr'].iloc[-2]
        if math.isnan(atr_val) or atr_val <= 0:
            atr_val = df['high'].iloc[-2] - df['low'].iloc[-2]

        sl_price_diff = atr_val
        balance = mt5.account_info().balance
        risk_amount = balance * self.risk

        tick = mt5.symbol_info_tick(self.symbol)
        ask, bid = tick.ask, tick.bid

        if bullish:
            entry_price = ask
            sl_price = entry_price - sl_price_diff
            tp_price = entry_price + 3 * sl_price_diff
            lot = self.calculate_lot_size(sl_price_diff, risk_amount)
            print(f"Bullish: Entry {entry_price}, SL {sl_price}, TP {tp_price}, lot {lot}")
            res = self.send_market_order(mt5.ORDER_TYPE_BUY, lot, entry_price, sl_price, tp_price)
            print("Result:", res)

        elif bearish:
            entry_price = bid
            sl_price = entry_price + sl_price_diff
            tp_price = entry_price - 3 * sl_price_diff
            lot = self.calculate_lot_size(sl_price_diff, risk_amount)
            print(f"Bearish: Entry {entry_price}, SL {sl_price}, TP {tp_price}, lot {lot}")
            res = self.send_market_order(mt5.ORDER_TYPE_SELL, lot, entry_price, sl_price, tp_price)
            print("Result:", res)

###################################### Main function ########################################
    def run(self):
        self.initialize()
        print(f"Bot started for {self.symbol}. Trading window {self.trade_start}-{self.trade_end} IST. Risk: {self.risk*100}%")
        try:
            while True:
                if not self.in_trading_window():
                    time.sleep(self.check_interval)
                    continue

                if self.has_open_position():
                    time.sleep(self.check_interval)
                    continue

                self.check_signal_and_trade()
                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            print("Bot stopped manually.")
        finally:
            self.shutdown()


if __name__ == "__main__":
    bot = EMACrossoverBot()
    bot.run()
