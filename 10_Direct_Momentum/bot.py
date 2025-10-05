############# Strategy ###############
# 1. dm+ = high(t) - high(t-1)
# -> bullish = +di > -di and gradient boosting uptrend probability > 0.7 then long
# -> bearish = -di > +di and gradient downtrend probability > 0.7 then short

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


############################ Global Variables ############################
SYMBOL          = "XAUUSDm"
TIMEFRAME       = mt5.TIMEFRAME_M5
LOOKBACK_BARS   = 1000
ATR_PERIOD      = 14
DMI_PERIOD      = 14
PREDICT_HORIZON = 30  # bars look-ahead for TP/SL simulation

########################## Risk Factors ###################################
LOT            = 0.01
SL_PIPS        = 100
TP_PIPS        = 50
MAX_POS        = 1
RISK_PER_TRADE = 0.01

############################ Account Info #################################
MT5_LOGIN    = 274242894
MT5_PASSWORD = "Mritunjay@76519"
MT5_SERVER   = "Exness-MT5Trial6"

########################### Logging Configuration #########################
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


############################ MT5 Connector ################################
class MT5Connector:
    def __init__(self, login=None, password=None, server=None):
        self.login    = login
        self.password = password
        self.server   = server
        self._init_mt5()

    def _init_mt5(self):
        if not mt5.initialize():
            ok = mt5.initialize(login=self.login, password=self.password, server=self.server)
            if not ok:
                raise RuntimeError(f"MT5 initialize/login failed, error code: {mt5.last_error()}")
        logger.info("MT5 initialized successfully.")

    def shutdown(self):
        mt5.shutdown()

    def fetch_ohlc(self, symbol, timeframe, n_bars=1000):
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"Failed to fetch rates for {symbol}.")
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    def account_info(self):
        return mt5.account_info()

    def send_order(self, symbol, order_type, volume, sl_pips, tp_pips):
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            raise RuntimeError(f"Symbol {symbol} not found.")

        tick  = mt5.symbol_info_tick(symbol)
        price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid

        point = symbol_info.point
        sl = price - sl_pips * point if order_type == mt5.ORDER_TYPE_BUY else price + sl_pips * point
        tp = price + tp_pips * point if order_type == mt5.ORDER_TYPE_BUY else price - tp_pips * point

        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       float(volume),
            "type":         int(order_type),
            "price":        float(price),
            "sl":           float(sl),
            "tp":           float(tp),
            "deviation":    20,
            "magic":        123456,
            "comment":      "GradientBoost_Auto",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.warning(f"Order failed: {result.comment}, code={result.retcode}")
        else:
            logger.info(f"Order placed successfully: {result}")
        return result

    def close_all_positions(self, symbol):
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return
        for pos in positions:
            order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            tick       = mt5.symbol_info_tick(symbol)
            price      = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask

            close_request = {
                "action":       mt5.TRADE_ACTION_DEAL,
                "position":     pos.ticket,
                "symbol":       symbol,
                "volume":       pos.volume,
                "type":         order_type,
                "price":        price,
                "deviation":    20,
                "magic":        123456,
                "comment":      "Auto Close",
                "type_time":    mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC
            }
            result = mt5.order_send(close_request)
            logger.info(f"Closed position {pos.ticket}, result: {result}")


############################ Feature Engineering ##########################
class FeatureEngineer:
    def __init__(self, atr_period=ATR_PERIOD, dmi_period=DMI_PERIOD):
        self.atr_period = atr_period
        self.dmi_period = dmi_period

    def calculate_indicators(self, df):
        ######################## True Range calculation ########################
        df['tr'] = np.maximum.reduce([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift()).abs(),
            (df['low'] - df['close'].shift()).abs()
        ])
        df['atr'] = df['tr'].rolling(self.atr_period).mean()

        ################ +DM and -DM Calculation#################
        up_move            = df['high'].diff()
        down_move          = -df['low'].diff()
        df['dm_plus']      = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        df['dm_minus']     = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        df['atr'].replace(0, np.nan, inplace=True)
        df['di_plus']      = 100 * (df['dm_plus'].rolling(self.dmi_period).sum() / df['atr'])
        df['di_minus']     = 100 * (df['dm_minus'].rolling(self.dmi_period).sum() / df['atr'])
        df['price_change'] = df['close'].pct_change()
        df['di_diff']      = df['di_plus'] - df['di_minus']
        return df.dropna()


############################ Gradient Boosting Model ######################
class GradientBoostModel:
    def __init__(self):
        self.model = GradientBoostingClassifier(n_estimators=500, random_state=42)
        self.scaler = StandardScaler()

    def train(self, df):
        features = ['price_change', 'di_diff', 'atr']
        X        = df[features]
        y        = (df['di_plus'].shift(-1) > df['di_minus'].shift(-1)).astype(int).iloc[:-1]
        X        = X.iloc[:-1]
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        logger.info("Model trained successfully on latest data window.")

    def predict(self, df):
        latest             = df.iloc[-1]
        X_pred             = np.array([[latest['price_change'], latest['di_diff'], latest['atr']]])
        X_scaled           = self.scaler.transform(X_pred)
        prob               = self.model.predict_proba(X_scaled)[0]
        up_prob, down_prob = prob[1], prob[0]

        if latest['di_plus'] > latest['di_minus'] and up_prob > 0.7:
            return "BUY"
        elif latest['di_minus'] > latest['di_plus'] and down_prob > 0.7:
            return "SELL"
        return "HOLD"


############################ Trading Bot ##################################
class LiveTrader:
    def __init__(self, mt5_connector, feature_engineer, model):
        self.mt5   = mt5_connector
        self.fe    = feature_engineer
        self.model = model

    def run(self):
        logger.info("Starting DMI + Gradient Boosting bot...")
        df = self.mt5.fetch_ohlc(SYMBOL, TIMEFRAME, LOOKBACK_BARS)
        df = self.fe.calculate_indicators(df)
        self.model.train(df)

        while True:
            df     = self.mt5.fetch_ohlc(SYMBOL, TIMEFRAME, LOOKBACK_BARS)
            df     = self.fe.calculate_indicators(df)
            signal = self.model.predict(df)
            logger.info(f"Signal: {signal}")

            positions    = mt5.positions_get(symbol=SYMBOL)
            has_position = len(positions) > 0 if positions else False

            if signal == "BUY" and not has_position:
                self.mt5.close_all_positions(SYMBOL)
                self.mt5.send_order(SYMBOL, mt5.ORDER_TYPE_BUY, LOT, SL_PIPS, TP_PIPS)

            elif signal == "SELL" and not has_position:
                self.mt5.close_all_positions(SYMBOL)
                self.mt5.send_order(SYMBOL, mt5.ORDER_TYPE_SELL, LOT, SL_PIPS, TP_PIPS)

            time.sleep(60 * 5)  # run every 5 minutes

############################################## Backtester ##############################################
class SimpleBacktester:
    def __init__(self, df, feature_cols, model, scaler=None, horizon=PREDICT_HORIZON):
        self.df           = df.reset_index(drop=True).copy()
        self.feature_cols = feature_cols
        self.model        = model
        self.scaler       = scaler
        self.horizon      = horizon

    def run(self, threshold=0.7, save_path=None, plot=True):
        X        = self.df[self.feature_cols]
        X_scaled = self.scaler.transform(X)
        probs    = self.model.predict_proba(X_scaled)
        preds    = (probs[:, 1] >= threshold).astype(int)

        trades = []
        for i in range(len(preds) - self.horizon):
            pred  = preds[i]
            entry = self.df.loc[i, "close"]
            #####for PIP_VALUE calculation when connected to MT5
            # symbol_info = mt5.symbol_info(SYMBOL)
            # PIP_VALUE = symbol_info.point if symbol_info else 0.01
            PIP_VALUE = 0.01  # for XAUUSDm

            sl = SL_PIPS * PIP_VALUE
            tp = TP_PIPS * PIP_VALUE
            if pred == 1:  # BUY
                exit_price = entry + tp
                stop_price = entry - sl
            else:  # SELL
                exit_price = entry - tp
                stop_price = entry + sl

            window = self.df.loc[i+1 : i + self.horizon]
            hit, hit_price = None, None

            for _, r in window.iterrows():
                high, low = r["high"], r["low"]
                if pred == 1:
                    if low <= stop_price:
                        hit, hit_price = "SL", stop_price
                        break
                    elif high >= exit_price:
                        hit, hit_price = "TP", exit_price
                        break
                else:
                    if high >= stop_price:
                        hit, hit_price = "SL", stop_price
                        break
                    elif low <= exit_price:
                        hit, hit_price = "TP", exit_price
                        break

            if hit is None:
                hit = "Close"
                hit_price = self.df.loc[i + self.horizon, "close"]

            pnl = hit_price - entry if pred == 1 else entry - hit_price
            trades.append({
                "time":   self.df.loc[i, "time"],
                "pred": "BUY" if pred == 1 else "SELL",
                "entry":  entry,
                "exit":   hit_price,
                "result": hit,
                "pnl":    pnl
            })

        trades_df = pd.DataFrame(trades)
        trades_df["cum_pnl"] = trades_df["pnl"].cumsum()
        win_rate = (trades_df["pnl"] > 0).mean() * 100

        logger.info(f"Backtest Complete | Win Rate: {win_rate:.2f}% | Total Trades: {len(trades_df)}")

        if save_path:
            trades_df.to_csv(save_path, index=False)
            logger.info(f"Saved trades to {save_path}")

        if plot:
            plt.figure(figsize=(12, 6))
            plt.plot(trades_df["time"], trades_df["cum_pnl"], label="Equity Curve")
            plt.axhline(0, color="red", linestyle="--")
            plt.xlabel("Time")
            plt.ylabel("Cumulative PnL")
            plt.title("Backtest Equity Curve (DMI + Gradient Boost)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return trades_df
    

############################ MAIN #########################################
if __name__ == "__main__":

    ################### Live Trading ##############################
    # mt5_conn = MT5Connector(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER)
    # fe       = FeatureEngineer(ATR_PERIOD, DMI_PERIOD)
    # gb_model = GradientBoostModel()
    # bot      = LiveTrader(mt5_conn, fe, gb_model)
    # try:
    #     bot.run()
    # except KeyboardInterrupt:
    #     logger.info("Bot stopped by user.")
    # finally:
    #     mt5_conn.shutdown()


    ############### Backtest before live trading ################
    #df = mt5_conn.fetch_ohlc(SYMBOL, TIMEFRAME, LOOKBACK_BARS)
    df         = pd.read_csv("C:/Users/Mritunjay Maddhesiya/OneDrive/Desktop/MT5/Data/XAUUSDm_5m.csv")
    df['time'] = pd.to_datetime(df['time'])
    fe         = FeatureEngineer(ATR_PERIOD, DMI_PERIOD)
    df         = fe.calculate_indicators(df)
    gb_model   = GradientBoostModel()
    gb_model.train(df)
    backtest   = SimpleBacktester(df=df, feature_cols=['price_change', 'di_diff', 'atr'], model=gb_model.model, scaler=gb_model.scaler)
    results    = backtest.run(save_path="backtest_results.csv")
    
