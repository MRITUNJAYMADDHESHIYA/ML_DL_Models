import time
import math
import threading
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo
import MetaTrader5 as mt5
import pandas as pd



SYMBOL         = "USTECm"
TIMEFRAME      = mt5.TIMEFRAME_M1
RISK           = 0.02              # 2% per trade
DEVIATION      = 20                # slippage in points
CHECK_INTERVAL = 5                 # seconds between checks
MAX_POSITIONS  = 5
TRADE_START    = dt_time(19, 0)    # IST 19:00
TRADE_END      = dt_time(23, 40)
IST_ZONE       = ZoneInfo("Asia/Kolkata")
MAGIC          = 123456

class CandleBot:
    def __init__(self, symbol=SYMBOL, timeframe=TIMEFRAME, risk=RISK, deviation=DEVIATION,
                 check_interval=CHECK_INTERVAL, max_positions=MAX_POSITIONS):
        self.symbol = symbol
        self.timeframe = timeframe
        self.risk = risk
        self.deviation = deviation
        self.check_interval = check_interval
        self.max_positions = max_positions

        self.trade_start = TRADE_START
        self.trade_end = TRADE_END
        self.IST = IST_ZONE

        # threading primitives
        self.stop_event = threading.Event()
        self.mt5_lock = threading.Lock()    # guard MT5 calls

        # ensure only one trade per candle
        self.last_trade_candle = None  

    # ------------------ MT5 helpers (thread-safe wrappers) ------------------
    def mt5_initialize(self):
        with self.mt5_lock:
            if not mt5.initialize():
                raise RuntimeError(f"MT5 initialize() failed, code={mt5.last_error()}")
        print("MT5 initialized")

    def mt5_shutdown(self):
        with self.mt5_lock:
            mt5.shutdown()
        print("MT5 shutdown")

    def account_info(self):
        with self.mt5_lock:
            return mt5.account_info()

    def symbol_info(self):
        with self.mt5_lock:
            return mt5.symbol_info(self.symbol)

    def symbol_tick(self):
        with self.mt5_lock:
            return mt5.symbol_info_tick(self.symbol)

    def copy_rates(self, count=10):
        with self.mt5_lock:
            rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, count)
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    def positions_get(self):
        with self.mt5_lock:
            return mt5.positions_get(symbol=self.symbol)

    def order_send(self, request):
        with self.mt5_lock:
            return mt5.order_send(request)

    # ------------------ Trading logic ------------------
    def in_trading_window(self):
        now_ist = datetime.now(self.IST)
        return self.trade_start <= now_ist.time() <= self.trade_end

    def calculate_lot_size(self, stop_price_diff, risk_amount):
        info = self.symbol_info()
        if info is None:
            print("symbol_info not available, using fallback lot")
            return getattr(info, 'volume_min', 0.01) if info else 0.01

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
            print("Warning: falling back to minimal lot")
            return getattr(info, 'volume_min', 0.01)

        vol_step = getattr(info, 'volume_step', 0.01)
        vol_min = getattr(info, 'volume_min', 0.01)
        vol_max = getattr(info, 'volume_max', 100.0)

        if math.isnan(lots) or lots <= 0:
            return vol_min

        lots = max(vol_min, min(lots, vol_max))
        rounded = math.floor(lots / vol_step) * vol_step
        return max(round(rounded, 8), vol_min)

    def send_market_order(self, order_type, volume, price, sl_price=None, tp_price=None, comment="candle_bot"):
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "deviation": self.deviation,
            "magic": MAGIC,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        if sl_price is not None:
            req["sl"] = sl_price
        if tp_price is not None and tp_price != 0:
            req["tp"] = tp_price

        res = self.order_send(req)
        return res

    def close_position(self, position):
        tick = self.symbol_tick()
        if tick is None:
            print("No tick available to close position")
            return None

        close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask

        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": position.volume,
            "type": close_type,
            "position": int(position.ticket),
            "price": price,
            "deviation": self.deviation,
            "magic": MAGIC,
            "comment": "dynamic_tp_close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        res = self.order_send(req)
        return res

    # ------------------ Workers ------------------
    def entry_worker(self):
        print("Entry thread started")
        while not self.stop_event.is_set():
            try:
                if not self.in_trading_window():
                    time.sleep(self.check_interval)
                    continue

                positions = self.positions_get()
                pos_count = len(positions) if positions is not None else 0
                if pos_count >= self.max_positions:
                    time.sleep(self.check_interval)
                    continue

                df = self.copy_rates(count=5)
                if df is None or len(df) < 3:
                    time.sleep(self.check_interval)
                    continue

                prev = df.iloc[-2]
                candle_time = prev['time']

                # prevent multiple trades in same candle
                if self.last_trade_candle == candle_time:
                    time.sleep(self.check_interval)
                    continue

                bullish_prev = prev['close'] > prev['open']
                bearish_prev = prev['close'] < prev['open']

                if bullish_prev:
                    tick = self.symbol_tick()
                    if tick:
                        entry_price = tick.ask
                        sl_price = float(prev['low'])
                        stop_diff = abs(entry_price - sl_price)
                        balance = float(self.account_info().balance)
                        risk_amount = balance * self.risk
                        lot = self.calculate_lot_size(stop_diff, risk_amount)
                        print(f"[ENTRY] BUY signal. Entry {entry_price}, SL {sl_price}, lot {lot}")
                        res = self.send_market_order(mt5.ORDER_TYPE_BUY, lot, entry_price, sl_price, None, comment="candle_buy")
                        print("Order send result:", res)
                        self.last_trade_candle = candle_time

                elif bearish_prev:
                    tick = self.symbol_tick()
                    if tick:
                        entry_price = tick.bid
                        sl_price = float(prev['high'])
                        stop_diff = abs(sl_price - entry_price)
                        balance = float(self.account_info().balance)
                        risk_amount = balance * self.risk
                        lot = self.calculate_lot_size(stop_diff, risk_amount)
                        print(f"[ENTRY] SELL signal. Entry {entry_price}, SL {sl_price}, lot {lot}")
                        res = self.send_market_order(mt5.ORDER_TYPE_SELL, lot, entry_price, sl_price, None, comment="candle_sell")
                        print("Order send result:", res)
                        self.last_trade_candle = candle_time
            except Exception as e:
                print("Entry thread error:", e)

            time.sleep(self.check_interval)

        print("Entry thread stopped")

    def manager_worker(self):
        print("Manager thread started")
        while not self.stop_event.is_set():
            try:
                if not self.in_trading_window():
                    time.sleep(self.check_interval)
                    continue

                df = self.copy_rates(count=3)
                if df is None or len(df) < 2:
                    time.sleep(self.check_interval)
                    continue

                prev = df.iloc[-2]
                bullish = prev['close'] > prev['open']
                bearish = prev['close'] < prev['open']

                positions = self.positions_get()
                if not positions:
                    time.sleep(self.check_interval)
                    continue

                for pos in positions:
                    if pos.type == mt5.ORDER_TYPE_BUY and bearish:
                        print(f"[MANAGER] Closing BUY pos ticket {pos.ticket} because bearish candle formed")
                        res = self.close_position(pos)
                        print("Close result:", res)

                    elif pos.type == mt5.ORDER_TYPE_SELL and bullish:
                        print(f"[MANAGER] Closing SELL pos ticket {pos.ticket} because bullish candle formed")
                        res = self.close_position(pos)
                        print("Close result:", res)

            except Exception as e:
                print("Manager thread error:", e)

            time.sleep(self.check_interval)

        print("Manager thread stoppe")

    def run(self):
        self.mt5_initialize()
        print(f"Bot started for {self.symbol}. Trading window {self.trade_start} - {self.trade_end} IST. Max positions: {self.max_positions}")

        self.entry_thread = threading.Thread(target=self.entry_worker, name="EntryThread", daemon=True)
        self.manager_thread = threading.Thread(target=self.manager_worker, name="ManagerThread", daemon=True)
        self.entry_thread.start()
        self.manager_thread.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stop requested by user (KeyboardInterrupt). Shutting down...")
            self.stop_event.set()
            self.entry_thread.join(timeout=5)
            self.manager_thread.join(timeout=5)
        finally:
            self.mt5_shutdown()


if __name__ == '__main__':
    bot = CandleBot()
    bot.run()
