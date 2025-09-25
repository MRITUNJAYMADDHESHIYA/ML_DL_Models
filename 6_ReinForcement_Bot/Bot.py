#!/usr/bin/env python3
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import ta
import time
import datetime
import sys
import os

####################### Global variable ############################
SYMBOL          = "XAUUSDm"   # will validate after login
TIMEFRAME       = mt5.TIMEFRAME_M1
LOOKBACK        = 50
INITIAL_BALANCE = 100000
LOT_SIZE        = 0.01
SL_PIPS         = 1000
TP_PIPS         = 5000
MODEL_PATH      = "xauusd_drl_model"

########################### Data Loader ##################################
def get_data(symbol, timeframe, n=1000):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    if rates is None:
        raise RuntimeError("No rates received from MT5")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df['returns'] = df['close'].pct_change()
    df['sma'] = ta.trend.sma_indicator(df['close'], window=10)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df.fillna(0, inplace=True)
    return df

####################################### Gym Env ################################
class TradingEnv(gym.Env):
    def __init__(self, df, lookback=LOOKBACK, initial_balance=INITIAL_BALANCE):
        super().__init__()
        self.df                = df
        self.lookback          = lookback
        self.initial_balance   = initial_balance
        self.balance           = initial_balance
        self.position          = 0
        self.index             = lookback
        self.done              = False

        self.action_space      = spaces.Discrete(3)  # 0 hold, 1 buy, 2 sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(lookback, df.shape[1]), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.position = 0
        self.index = self.lookback
        self.done = False
        obs = self.df.iloc[self.index-self.lookback:self.index].values
        return obs, {}

    def step(self, action):
        prev_price = self.df['close'].iloc[self.index-1]
        price = self.df['close'].iloc[self.index]

        # Action handling
        if action == 1:   # buy
            self.position = 1
        elif action == 2: # sell
            self.position = -1

        # Reward as PnL difference
        reward = self.position * (price - prev_price)

        self.index += 1
        terminated = self.index >= len(self.df)
        truncated = False

        obs = self.df.iloc[self.index-self.lookback:self.index].values
        return obs, reward, terminated, truncated, {}

############################## Train this model #####################################
def train_agent():
    df = get_data(SYMBOL, TIMEFRAME, n=5000)
    env = DummyVecEnv([lambda: TradingEnv(df)])
    model = PPO("MlpPolicy", env, verbose=1,
                learning_rate=0.0003, n_steps=128,
                batch_size=64, gamma=0.99)
    model.learn(total_timesteps=50000)
    model.save(MODEL_PATH)
    print("Model trained and saved")

################################## Close positions #################################
def close_all_positions(symbol):
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return
    for pos in positions:
        opposite_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(symbol)
        price = tick.ask if opposite_type == mt5.ORDER_TYPE_BUY else tick.bid
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": pos.volume,
            "type": opposite_type,
            "price": price,
            "deviation": 20,
            "magic": 123456,
            "comment": "Close by DRL",
            "type_filling": mt5.ORDER_FILLING_IOC
        }
        result = mt5.order_send(request)
        print("Close result:", result)

################################ Open trade ########################################
def execute_trade(action, symbol, lot=LOT_SIZE):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print("No tick data")
        return
    point = mt5.symbol_info(symbol).point
    if action == 1:  # BUY
        price = tick.ask
        sl = price - SL_PIPS * point
        tp = price + TP_PIPS * point
        order_type = mt5.ORDER_TYPE_BUY
    elif action == 2:  # SELL
        price = tick.bid
        sl = price + SL_PIPS * point
        tp = price - TP_PIPS * point
        order_type = mt5.ORDER_TYPE_SELL
    else:
        return
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 123456,
        "comment": "DRL trade",
        "type_filling": mt5.ORDER_FILLING_IOC
    }
    result = mt5.order_send(request)
    print("Trade result:", result)

#################################### Live Trade ######################################
def live_trading():
    if not os.path.exists(MODEL_PATH + ".zip"):
        print("No trained model found. Train first.")
        return
    model = PPO.load(MODEL_PATH)

    positions = mt5.positions_get(symbol=SYMBOL)
    if not positions:
        position_open = False
        current_position = None  # store dict: {'type': ..., 'sl': ..., 'tp': ...}

    try:
        while True:
            df_live = get_data(SYMBOL, TIMEFRAME, n=LOOKBACK+1)
            obs = df_live.values[-LOOKBACK:]
            action, _ = model.predict(obs, deterministic=True)
            act_name = ["HOLD", "BUY", "SELL"][action]
            last_price = df_live['close'].iloc[-1]
            print(f"{datetime.datetime.now()} | Action: {act_name} | Price: {last_price:.2f}")

            tick = mt5.symbol_info_tick(SYMBOL)
            price = tick.ask if action == 1 else tick.bid

            # Check if existing position hit SL/TP
            if position_open and current_position:
                if (current_position['type'] == mt5.ORDER_TYPE_BUY and 
                    (price <= current_position['sl'] or price >= current_position['tp'])) \
                or (current_position['type'] == mt5.ORDER_TYPE_SELL and
                    (price >= current_position['sl'] or price <= current_position['tp'])):
                    print("SL/TP hit. Closing position...")
                    close_all_positions(SYMBOL)
                    position_open = False
                    current_position = None

            # Open new position only if no position is open
            if not position_open and action in [1, 2]:
                execute_trade(action, SYMBOL)
                # Save current position info
                tick = mt5.symbol_info_tick(SYMBOL)
                point = mt5.symbol_info(SYMBOL).point
                if action == 1:  # BUY
                    sl = tick.ask - SL_PIPS * point
                    tp = tick.ask + TP_PIPS * point
                    order_type = mt5.ORDER_TYPE_BUY
                else:  # SELL
                    sl = tick.bid + SL_PIPS * point
                    tp = tick.bid - TP_PIPS * point
                    order_type = mt5.ORDER_TYPE_SELL
                current_position = {'type': order_type, 'sl': sl, 'tp': tp}
                position_open = True

            # Log trade/action
            with open("drl_trade_log.csv", "a") as f:
                f.write(f"{datetime.datetime.now()},{act_name},{last_price}\n")

            time.sleep(60)

    except KeyboardInterrupt:
        print("\nLive trading stopped by user")
        mt5.shutdown()


################################## Main ##########################################
if __name__ == "__main__":
    login_number = 274242894
    password     = "Mritunjay@76519"
    server       = "Exness-MT5Trial6"

    ###################### MT5 initial
    if not mt5.initialize(login=login_number, password=password, server=server):
        print("Connection failed:", mt5.last_error())
        sys.exit(1)
    if mt5.account_info():
        print('Connection with MT5 established:', mt5.account_info().login)

    ##################### Verify symbol exists
    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info is None:
        print(f"{SYMBOL} not found. Available XAU symbols:")
        for s in mt5.symbols_get():
            if "XAU" in s.name:
                print(" ->", s.name)
        mt5.shutdown()
        sys.exit(1)
    if not symbol_info.visible:
        mt5.symbol_select(SYMBOL, True)

    ####################Choose mode
    #MODE = "train"
    MODE = "live"
    if MODE == "train":
        train_agent()
    elif MODE == "live":
        live_trading()
