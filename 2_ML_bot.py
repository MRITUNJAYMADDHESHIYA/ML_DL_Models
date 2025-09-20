import time
import math
import threading
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import xgboost as xgb
import ta
from sklearn.model_selection import train_test_split

# ---------------- CONFIG ----------------
SYMBOL         = "XAUUSDm"
TIMEFRAME      = mt5.TIMEFRAME_M1
RISK           = 0.02
DEVIATION      = 20
CHECK_INTERVAL = 5
TRADE_START    = dt_time(19, 0)
TRADE_END      = dt_time(23, 40)
IST_ZONE       = ZoneInfo("Asia/Kolkata")
MAGIC          = 123456
N_BARS         = 5000   # training candles
# ----------------------------------------

class MLBot:
    def __init__(self):
        self.model = None
        self.stop_event = threading.Event()
        self.mt5_lock = threading.Lock()

    # --------- MT5 helpers ----------
    def mt5_initialize(self):
        with self.mt5_lock:
            if not mt5.initialize():
                raise RuntimeError("MT5 init failed")
        print("MT5 initialized")

    def mt5_shutdown(self):
        with self.mt5_lock:
            mt5.shutdown()
        print("MT5 shutdown")

    def symbol_info(self):
        with self.mt5_lock:
            return mt5.symbol_info(SYMBOL)

    def symbol_tick(self):
        with self.mt5_lock:
            return mt5.symbol_info_tick(SYMBOL)

    def copy_rates(self, count=1000):
        with self.mt5_lock:
            rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, count)
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df

    def positions_get(self):
        with self.mt5_lock:
            return mt5.positions_get(symbol=SYMBOL)

    def order_send(self, request):
        with self.mt5_lock:
            return mt5.order_send(request)
        
#################################### Train Model ################################
    def train_model(self):
        print("Fetching training data...")
        df = self.copy_rates(N_BARS)

        # Features
        for lag in range(1, 6):
            df[f"lag_close_{lag}"] = df["close"].shift(lag)
        df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
        df["ema_10"] = ta.trend.EMAIndicator(df["close"], window=10).ema_indicator()
        df["ema_30"] = ta.trend.EMAIndicator(df["close"], window=30).ema_indicator()
        df["atr"] = ta.volatility.AverageTrueRange(
            df["high"], df["low"], df["close"], window=14
        ).average_true_range()

        df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
        df = df.dropna()

        features = [c for c in df.columns if c not in ["time", "target"]]
        X, y = df[features], df["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        print("Training XGBoost...")
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)
        acc = (self.model.predict(X_test) == y_test).mean()
        print(f"Validation Accuracy: {acc:.4f}")

        self.features = features

    # --------- Trading ----------
    def in_trading_window(self):
        now_ist = datetime.now(IST_ZONE)
        return TRADE_START <= now_ist.time() <= TRADE_END

    def calculate_lot_size(self, stop_diff, risk_amount):
        info = self.symbol_info()
        vol_min = getattr(info, "volume_min", 0.01)
        vol_max = getattr(info, "volume_max", 100.0)
        vol_step = getattr(info, "volume_step", 0.01)

        contract_size = getattr(info, "trade_contract_size", 100.0)
        lots = risk_amount / (contract_size * stop_diff)
        lots = max(vol_min, min(lots, vol_max))
        return round(math.floor(lots / vol_step) * vol_step, 2)

    def send_order(self, order_type, volume, price, sl_price):
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": volume,
            "type": order_type,
            "price": price,
            "deviation": DEVIATION,
            "magic": MAGIC,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
            "sl": sl_price,
        }
        return self.order_send(req)

############################################## Entry Position ##########################################
    def entry_worker(self):
        print("Entry thread started")
        while not self.stop_event.is_set():
            try:
                if not self.in_trading_window():
                    time.sleep(CHECK_INTERVAL)
                    continue

                df = self.copy_rates(50)
                if df is None or len(df) < 30:
                    time.sleep(CHECK_INTERVAL)
                    continue

                row = df.iloc[-1:].copy()
                for lag in range(1, 6):
                    row[f"lag_close_{lag}"] = df["close"].shift(lag).iloc[-1]
                row["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi().iloc[-1]
                row["ema_10"] = ta.trend.EMAIndicator(df["close"], window=10).ema_indicator().iloc[-1]
                row["ema_30"] = ta.trend.EMAIndicator(df["close"], window=30).ema_indicator().iloc[-1]
                row["atr"] = ta.volatility.AverageTrueRange(
                    df["high"], df["low"], df["close"], window=14
                ).average_true_range().iloc[-1]

                row = row[self.features]

                pred = self.model.predict(row)[0]
                tick = self.symbol_tick()
                if tick is None:
                    time.sleep(CHECK_INTERVAL)
                    continue

                balance = float(mt5.account_info().balance)
                risk_amount = balance * RISK

                if pred == 1:  # bullish -> BUY
                    sl = float(df.iloc[-2]["low"])
                    stop_diff = abs(tick.ask - sl)
                    lot = self.calculate_lot_size(stop_diff, risk_amount)
                    print(f"[ENTRY] BUY {lot} lots at {tick.ask}, SL {sl}")
                    res = self.send_order(mt5.ORDER_TYPE_BUY, lot, tick.ask, sl)
                    print("Order result:", res)

                elif pred == 0:  # bearish -> SELL
                    sl = float(df.iloc[-2]["high"])
                    stop_diff = abs(sl - tick.bid)
                    lot = self.calculate_lot_size(stop_diff, risk_amount)
                    print(f"[ENTRY] SELL {lot} lots at {tick.bid}, SL {sl}")
                    res = self.send_order(mt5.ORDER_TYPE_SELL, lot, tick.bid, sl)
                    print("Order result:", res)

            except Exception as e:
                print("Entry error:", e)

            time.sleep(CHECK_INTERVAL)
        print("Entry thread stopped")

    def run(self):
        self.mt5_initialize()
        self.train_model()

        entry_thread = threading.Thread(target=self.entry_worker, daemon=True)
        entry_thread.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping bot...")
            self.stop_event.set()
            entry_thread.join(timeout=5)
        finally:
            self.mt5_shutdown()


if __name__ == "__main__":
    bot = MLBot()
    bot.run()
