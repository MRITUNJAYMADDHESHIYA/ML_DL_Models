#!/usr/bin/env python3
"""
MLBot for XAUUSD / BTCUSD 1-min using XGBoost with MT5 execution.
Modified to place real trades (PAPER_TRADING=False) and improved MT5 init & order handling.
"""

import os
import time
import math
import threading
import logging
from datetime import datetime, time as dt_time, timedelta, timezone
from zoneinfo import ZoneInfo

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import xgboost as xgb
import ta
from sklearn.model_selection import train_test_split
from joblib import dump, load

# ----------------------------- CONFIG -----------------------------
SYMBOL            = "BTCUSDm"            # adapt to your broker symbol (check exact name)
TIMEFRAME         = mt5.TIMEFRAME_M1
RISK_FRACTION     = 0.022                 # fraction of account balance to risk per trade
DEVIATION         = 20
CHECK_INTERVAL    = 5                    # seconds between checks in entry loop
TRADE_START       = dt_time(19, 0)       # IST trading window start
TRADE_END         = dt_time(23, 55)      # IST trading window end
IST_ZONE          = ZoneInfo("Asia/Kolkata")
MAGIC             = 123456
N_BARS            = 5000                 # number of historical bars for training
MODEL_PATH        = "xauusd_xgb_model.joblib"
FEATURES_PATH     = "xauusd_features.joblib"
PAPER_TRADING     = False                # False = send real orders to MT5
RR                = 2.0                  # Risk:Reward ratio for TP
MAX_POS_LOTS      = 5.0                  # absolute cap on lots
MIN_TIME_BETWEEN_TRADES_SEC = 10         # cooldown between placed trades
LOG_FILE          = "mlbot.log"
# ------------------------------------------------------------------

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger("MLBot")

class MLBot:
    def __init__(self):
        self.model = None
        self.features = None
        self.stop_event = threading.Event()
        self.mt5_lock = threading.Lock()
        self.last_trade_time = datetime.now(timezone.utc) - timedelta(seconds=60)

    def mt5_initialize(self):
        with self.mt5_lock:
            if not mt5.initialize():
                raise RuntimeError(f"MT5 initialize() failed, code={mt5.last_error()}")
        print("MT5 initialized")

    def mt5_shutdown(self):
        with self.mt5_lock:
            try:
                mt5.shutdown()
            except Exception:
                logger.exception("Error shutting down MT5")
        logger.info("MT5 shutdown")

    def ensure_symbol_selected(self):
        with self.mt5_lock:
            info = mt5.symbol_info(SYMBOL)
            if info is None:
                logger.error("Symbol %s not found in Market Watch", SYMBOL)
                return False
            if not info.visible:
                ok = mt5.symbol_select(SYMBOL, True)
                if not ok:
                    logger.error("Failed to select symbol %s", SYMBOL)
                    return False
                logger.info("Selected symbol %s in Market Watch", SYMBOL)
            return True

    def symbol_info(self):
        with self.mt5_lock:
            return mt5.symbol_info(SYMBOL)

    def symbol_tick(self):
        with self.mt5_lock:
            return mt5.symbol_info_tick(SYMBOL)

    def copy_rates(self, count=1000):
        with self.mt5_lock:
            rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, count)
        if rates is None:
            return None
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df

    def positions_get(self):
        with self.mt5_lock:
            return mt5.positions_get(symbol=SYMBOL)

    def order_send_mt5(self, request):
        # send order to MT5 with locking and basic error handling
        with self.mt5_lock:
            try:
                res = mt5.order_send(request)
                return res
            except Exception:
                logger.exception("order_send exception")
                return None

    # ---------------- Feature engineering (train & live must match) ----------------
    def build_features(self, df):
        # expect df has time, open, high, low, close, tick_volume (or volume)
        df = df.copy()
        # create lag close values (lags 1..5)
        for lag in range(1, 6):
            df[f"lag_close_{lag}"] = df["close"].shift(lag)
        # indicators
        df["rsi"]    = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
        df["ema_10"] = ta.trend.EMAIndicator(df["close"], window=10).ema_indicator()
        df["ema_30"] = ta.trend.EMAIndicator(df["close"], window=30).ema_indicator()
        # simple ATR
        df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
        df = df.dropna()
        return df

    # ---------------- Training / loading ----------------
    def train_model(self):
        logger.info("Fetching training data...")
        df = self.copy_rates(N_BARS)
        if df is None or len(df) < 200:
            raise RuntimeError("Not enough bars for training")

        df = self.build_features(df)
        # target: next-minute up (1) or not (0)
        df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
        df = df.dropna()

        features = [c for c in df.columns if c not in ["time", "target"]]
        X = df[features]
        y = df["target"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        logger.info("Training XGBoost model...")
        model = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        model.fit(X_train, y_train)
        acc = (model.predict(X_test) == y_test).mean()
        logger.info(f"Validation Accuracy: {acc:.4f}")

        # save model and features
        dump(model, MODEL_PATH)
        dump(features, FEATURES_PATH)
        logger.info(f"Saved model to {MODEL_PATH} and features to {FEATURES_PATH}")

        self.model = model
        self.features = features

    def load_model(self):
        if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
            logger.info("Model or features not found on disk")
            return False
        self.model = load(MODEL_PATH)
        self.features = load(FEATURES_PATH)
        logger.info(f"Loaded model {MODEL_PATH} and features list")
        return True

    # ---------------- Trading helpers ----------------
    def in_trading_window(self):
        now_ist = datetime.now(IST_ZONE)
        return TRADE_START <= now_ist.time() <= TRADE_END

    def has_open_position(self):
        pos = self.positions_get()
        return (pos is not None and len(pos) > 0)

    def calculate_lot_size(self, stop_diff_price, risk_amount):
        """
        stop_diff_price: price distance from entry to SL in quote currency (e.g. USD)
        risk_amount: money to risk in account currency
        """
        info = self.symbol_info()
        if info is None:
            logger.warning("symbol_info returned None, using defaults for lot sizing")
            vol_min, vol_max, vol_step, contract_size = 0.01, 100.0, 0.01, 100.0
        else:
            vol_min = getattr(info, "volume_min", 0.01)
            vol_max = getattr(info, "volume_max", 100.0)
            vol_step = getattr(info, "volume_step", 0.01)
            # contract size may be named trade_contract_size or lot_size per broker
            contract_size = getattr(info, "trade_contract_size", None) or getattr(info, "lot_size", None) or 100.0

        if stop_diff_price <= 0 or math.isnan(stop_diff_price):
            logger.warning("Invalid stop_diff_price, returning minimum lot")
            return vol_min

        # lots = risk / (contract_size * stop_diff)
        try:
            lots = risk_amount / (contract_size * stop_diff_price)
        except Exception:
            logger.exception("Error calculating lots, returning vol_min")
            return vol_min

        lots = max(vol_min, min(lots, vol_max))
        # floor to vol_step
        lots = math.floor(lots / vol_step) * vol_step
        lots = round(lots, 2)
        # enforce absolute cap too
        lots = min(lots, MAX_POS_LOTS)
        return lots

    def send_order(self, signal, lot_size, sl, tp):
        """
        signal: "BUY" or "SELL"
        """
        # Use SYMBOL constant
        symbol = SYMBOL
        tick = self.symbol_tick()
        if tick is None:
            logger.error("symbol_tick returned None when placing order")
            return None

        price = float(tick.ask) if signal == "BUY" else float(tick.bid)
        order_type = mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot_size),
            "type": order_type,
            "price": price,
            "sl": float(sl) if sl is not None else 0.0,
            "tp": float(tp) if tp is not None else 0.0,
            "deviation": DEVIATION,
            "magic": MAGIC,
            "comment": "xgboost_trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        if PAPER_TRADING:
            logger.info("[PAPER] Simulated order: %s", request)
            class SimRes:
                def __init__(self):
                    self.retcode = 10009
                    self.comment = "PAPER_SIM"
            return SimRes()

        # send real order
        res = self.order_send_mt5(request)
        if res is None:
            logger.error("order_send returned None")
            return None

        # log result details
        retcode = getattr(res, "retcode", None)
        logger.info("Order send result: retcode=%s, comment=%s, order=%s", retcode, getattr(res, "comment", ""), request)
        # check success code (MT5 codes vary; trade succeeded when retcode indicates done)
        if retcode is None or (retcode != mt5.TRADE_RETCODE_DONE and retcode != 10009):
            logger.error("Order failed: retcode=%s, detail=%s", retcode, res)
        else:
            logger.info("Trade placed: %s %s lots @ %s (sl=%s tp=%s)", signal, lot_size, price, sl, tp)
        return res

    # ---------------- Entry thread ----------------
    def entry_worker(self):
        logger.info("Entry thread started")
        while not self.stop_event.is_set():
            try:
                if not self.in_trading_window():
                    time.sleep(CHECK_INTERVAL)
                    continue

                # Ensure symbol is selected and visible
                if not self.ensure_symbol_selected():
                    time.sleep(CHECK_INTERVAL)
                    continue

                # Avoid overtrading: only one position at a time
                if self.has_open_position():
                    logger.debug("Position already open, sleeping")
                    time.sleep(CHECK_INTERVAL)
                    continue

                df = self.copy_rates(200)  # fetch 200 bars for features
                if df is None or len(df) < 60:
                    logger.warning("Not enough bars fetched")
                    time.sleep(CHECK_INTERVAL)
                    continue

                df_feat = self.build_features(df)
                if df_feat is None or len(df_feat) == 0:
                    logger.warning("No features available yet")
                    time.sleep(CHECK_INTERVAL)
                    continue

                latest = df_feat.iloc[-1:].copy()
                # ensure all model features present
                for f in self.features:
                    if f not in latest.columns:
                        logger.error("Feature missing in live: %s", f)
                        raise RuntimeError(f"Feature {f} missing in live features")

                X_live = latest[self.features].values
                pred = int(self.model.predict(X_live)[0])
                prob = None
                try:
                    prob = float(self.model.predict_proba(X_live)[0,1])
                except Exception:
                    prob = None

                tick = self.symbol_tick()
                if tick is None:
                    logger.warning("symbol_tick returned None")
                    time.sleep(CHECK_INTERVAL)
                    continue

                # compute risk amount from account balance
                acc_info = mt5.account_info()
                if acc_info is None:
                    logger.warning("account_info returned None, using zero balance")
                    balance = 0.0
                else:
                    balance = float(acc_info.balance)
                risk_amount = balance * RISK_FRACTION

                # Determine entry logic
                prev_bar = df.iloc[-2]

                SL_EXTRA_POINTS = 100 #sl points
                if pred == 1:
                    # BUY
                    sl_price = float(prev_bar["low"]) - SL_EXTRA_POINTS * mt5.symbol_info(SYMBOL).point
                    entry_price = float(tick.ask)
                    stop_diff = abs(entry_price - sl_price)
                    if stop_diff <= 0:
                        logger.warning("Computed non-positive stop_diff for BUY; skipping")
                        time.sleep(CHECK_INTERVAL)
                        continue
                    lot = self.calculate_lot_size(stop_diff, risk_amount)
                    tp_price = entry_price + RR * stop_diff
                    logger.info("[ENTRY] BUY signal prob=%s lot=%s entry=%s sl=%s tp=%s", prob, lot, entry_price, sl_price, tp_price)

                    # Rate-limit trades
                    if (datetime.now(timezone.utc) - self.last_trade_time).total_seconds() < MIN_TIME_BETWEEN_TRADES_SEC:
                        logger.info("Cooldown active, skipping trade")
                    else:
                        res = self.send_order("BUY", lot, sl_price, tp_price)
                        logger.info("Order result: %s", res)
                        self.last_trade_time = datetime.now(timezone.utc)

                elif pred == 0:
                    # SELL
                    sl_price = float(prev_bar["high"]) + SL_EXTRA_POINTS * mt5.symbol_info(SYMBOL).point
                    entry_price = float(tick.bid)
                    stop_diff = abs(sl_price - entry_price)
                    if stop_diff <= 0:
                        logger.warning("Computed non-positive stop_diff for SELL; skipping")
                        time.sleep(CHECK_INTERVAL)
                        continue
                    lot = self.calculate_lot_size(stop_diff, risk_amount)
                    tp_price = entry_price - RR * stop_diff
                    logger.info("[ENTRY] SELL signal prob=%s lot=%s entry=%s sl=%s tp=%s", prob, lot, entry_price, sl_price, tp_price)

                    if (datetime.now(timezone.utc) - self.last_trade_time).total_seconds() < MIN_TIME_BETWEEN_TRADES_SEC:
                        logger.info("Cooldown active, skipping trade")
                    else:
                        res = self.send_order("SELL", lot, sl_price, tp_price)
                        logger.info("Order result: %s", res)
                        self.last_trade_time = datetime.now(timezone.utc)

                else:
                    logger.debug("No actionable prediction")

            except Exception:
                logger.exception("Entry loop exception")

            time.sleep(CHECK_INTERVAL)
        logger.info("Entry thread stopped")

    # ---------------- Main run ----------------
    def run(self):
        try:
            self.mt5_initialize()
        except Exception:
            logger.exception("Failed to initialize MT5")
            return

        try:
            # load model if present otherwise train
            if not self.load_model():
                logger.info("Training model because loading failed")
                self.train_model()
            # final safety check
            if self.model is None or self.features is None:
                raise RuntimeError("Model or features not available after load/train")

            entry_thread = threading.Thread(target=self.entry_worker, daemon=True)
            entry_thread.start()

            # Main loop simply waits; expand with monitoring/commands as needed
            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt: stopping bot")
            self.stop_event.set()
            entry_thread.join(timeout=5)

        except Exception:
            logger.exception("Unhandled exception in run loop")
            self.stop_event.set()
            try:
                entry_thread.join(timeout=5)
            except Exception:
                pass

        finally:
            try:
                self.mt5_shutdown()
            except Exception:
                logger.exception("Error during MT5 shutdown")


if __name__ == "__main__":
    bot = MLBot()
    bot.run()



























































# import time
# import math
# import threading
# from datetime import datetime, time as dt_time
# from zoneinfo import ZoneInfo
# import MetaTrader5 as mt5
# import pandas as pd
# import numpy as np
# import xgboost as xgb
# import ta
# from sklearn.model_selection import train_test_split

# ################################################# Global variable ##################################################
# SYMBOL         = "XAUUSDm"
# TIMEFRAME      = mt5.TIMEFRAME_M1
# RISK           = 0.02
# DEVIATION      = 20
# CHECK_INTERVAL = 5
# TRADE_START    = dt_time(19, 0)
# TRADE_END      = dt_time(23, 40)
# IST_ZONE       = ZoneInfo("Asia/Kolkata")
# MAGIC          = 123456
# N_BARS         = 5000   # training candles

# ####################################################### ML Bot #####################################################
# class MLBot:
#     def __init__(self):
#         self.model      = None
#         self.stop_event = threading.Event()
#         self.mt5_lock   = threading.Lock()

# ############################################## MT5 Initialization ###################################################
#     def mt5_initialize(self):
#         with self.mt5_lock:
#             if not mt5.initialize():
#                 raise RuntimeError("MT5 init failed")
#         print("MT5 initialized")

#     def mt5_shutdown(self):
#         with self.mt5_lock:
#             mt5.shutdown()
#         print("MT5 shutdown")

#     def symbol_info(self):
#         with self.mt5_lock:
#             return mt5.symbol_info(SYMBOL)

#     def symbol_tick(self):
#         with self.mt5_lock:
#             return mt5.symbol_info_tick(SYMBOL)

#     def copy_rates(self, count=1000):
#         with self.mt5_lock:
#             rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, count)
#         df = pd.DataFrame(rates)
#         df["time"] = pd.to_datetime(df["time"], unit="s")
#         return df

#     def positions_get(self):
#         with self.mt5_lock:
#             return mt5.positions_get(symbol=SYMBOL)

#     def order_send(self, request):
#         with self.mt5_lock:
#             return mt5.order_send(request)
        
# #################################### Train Model ################################
#     def train_model(self):
#         print("Fetching training data...")
#         df = self.copy_rates(N_BARS)

#         # Features
#         for lag in range(1, 6):
#             df[f"lag_close_{lag}"] = df["close"].shift(lag)
#         df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
#         df["ema_10"] = ta.trend.EMAIndicator(df["close"], window=10).ema_indicator()
#         df["ema_30"] = ta.trend.EMAIndicator(df["close"], window=30).ema_indicator()
#         df["atr"] = ta.volatility.AverageTrueRange(
#             df["high"], df["low"], df["close"], window=14
#         ).average_true_range()

#         df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
#         df = df.dropna()

#         features = [c for c in df.columns if c not in ["time", "target"]]
#         X, y = df[features], df["target"]

#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, shuffle=False
#         )

#         print("Training XGBoost...")
#         self.model = xgb.XGBClassifier(
#             n_estimators=200,
#             max_depth=6,
#             learning_rate=0.05,
#             subsample=0.8,
#             colsample_bytree=0.8,
#             random_state=42,
#             n_jobs=-1,
#         )
#         self.model.fit(X_train, y_train)
#         acc = (self.model.predict(X_test) == y_test).mean()
#         print(f"Validation Accuracy: {acc:.4f}")

#         self.features = features

# ######################################## Trading Time ##########################################
#     def in_trading_window(self):
#         now_ist = datetime.now(IST_ZONE)
#         return TRADE_START <= now_ist.time() <= TRADE_END
# ######################################## Lot Size ##############################################
#     def calculate_lot_size(self, stop_diff, risk_amount):
#         info          = self.symbol_info()
#         vol_min       = getattr(info, "volume_min", 0.01)
#         vol_max       = getattr(info, "volume_max", 100.0)
#         vol_step      = getattr(info, "volume_step", 0.01)
#         contract_size = getattr(info, "trade_contract_size", 100.0) #### depending on assets

#         lots = risk_amount / (contract_size * stop_diff)
#         lots = max(vol_min, min(lots, vol_max))
#         return round(math.floor(lots / vol_step) * vol_step, 2)
# ######################################### Send Orders #########################################
#     def send_order(self, order_type, volume, price, sl_price):
#         req = {
#             "action": mt5.TRADE_ACTION_DEAL,
#             "symbol": SYMBOL,
#             "volume": volume,
#             "type": order_type,
#             "price": price,
#             "deviation": DEVIATION,
#             "magic": MAGIC,
#             "type_time": mt5.ORDER_TIME_GTC,
#             "type_filling": mt5.ORDER_FILLING_IOC,
#             "sl": sl_price,
#         }
#         return self.order_send(req)

# ############################################## Entry Position ##########################################
#     def entry_worker(self):
#         print("Entry thread started")
#         while not self.stop_event.is_set():
#             try:
#                 if not self.in_trading_window():
#                     time.sleep(CHECK_INTERVAL)
#                     continue

#                 df = self.copy_rates(50)
#                 if df is None or len(df) < 30:
#                     time.sleep(CHECK_INTERVAL)
#                     continue

#                 row = df.iloc[-1:].copy()
#                 for lag in range(1, 6):
#                     row[f"lag_close_{lag}"] = df["close"].shift(lag).iloc[-1]
#                 row["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi().iloc[-1]
#                 row["ema_10"] = ta.trend.EMAIndicator(df["close"], window=10).ema_indicator().iloc[-1]
#                 row["ema_30"] = ta.trend.EMAIndicator(df["close"], window=30).ema_indicator().iloc[-1]
#                 row["atr"] = ta.volatility.AverageTrueRange(
#                     df["high"], df["low"], df["close"], window=14
#                 ).average_true_range().iloc[-1]

#                 row = row[self.features]

#                 pred = self.model.predict(row)[0]
#                 tick = self.symbol_tick()
#                 if tick is None:
#                     time.sleep(CHECK_INTERVAL)
#                     continue

#                 balance = float(mt5.account_info().balance)
#                 risk_amount = balance * RISK

#                 if pred == 1:  # bullish -> BUY
#                     sl = float(df.iloc[-2]["low"])
#                     stop_diff = abs(tick.ask - sl)
#                     lot = self.calculate_lot_size(stop_diff, risk_amount)
#                     print(f"[ENTRY] BUY {lot} lots at {tick.ask}, SL {sl}")
#                     res = self.send_order(mt5.ORDER_TYPE_BUY, lot, tick.ask, sl)
#                     print("Order result:", res)

#                 elif pred == 0:  # bearish -> SELL
#                     sl = float(df.iloc[-2]["high"])
#                     stop_diff = abs(sl - tick.bid)
#                     lot = self.calculate_lot_size(stop_diff, risk_amount)
#                     print(f"[ENTRY] SELL {lot} lots at {tick.bid}, SL {sl}")
#                     res = self.send_order(mt5.ORDER_TYPE_SELL, lot, tick.bid, sl)
#                     print("Order result:", res)

#             except Exception as e:
#                 print("Entry error:", e)

#             time.sleep(CHECK_INTERVAL)
#         print("Entry thread stopped")

# ########################################### Run Main ##########################################################
#     def run(self):
#         self.mt5_initialize()
#         self.train_model()

#         entry_thread = threading.Thread(target=self.entry_worker, daemon=True)
#         entry_thread.start()

#         try:
#             while True:
#                 time.sleep(1)
#         except KeyboardInterrupt:
#             print("Stopping bot...")
#             self.stop_event.set()
#             entry_thread.join(timeout=5)
#         finally:
#             self.mt5_shutdown()


# if __name__ == "__main__":
#     bot = MLBot()
#     bot.run()
